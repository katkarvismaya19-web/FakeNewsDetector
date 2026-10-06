"""Unified, calibrated predictor for rnn / lstm / bert / llm."""
import json, os, pickle
import numpy as np
import torch

from config import LOW_MEMORY, MAX_LEN, MODEL_REPO, OUT_DIR
from data import clean_text
from models_rnn import SeqClassifier, Vocab  # noqa: F401  (Vocab needed to unpickle)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _temp(name):
    f = f"{OUT_DIR}/{name}_meta.json"
    return json.load(open(f)).get("temperature", 1.0) if os.path.exists(f) else 1.0


def ensure_models():
    """Download trained models from the Hugging Face Hub if they aren't on disk (used when deployed)."""
    if os.path.exists(f"{OUT_DIR}/bert/config.json") or not MODEL_REPO:
        return
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=MODEL_REPO, local_dir=OUT_DIR)


class Detector:
    def __init__(self, models=("rnn", "lstm", "bert")):
        ensure_models()
        self.models, self.temps = {}, {}
        for n in models:
            self.temps[n] = _temp(n)
            if n in ("rnn", "lstm"):
                vocab = pickle.load(open(f"{OUT_DIR}/{n}_vocab.pkl", "rb"))
                m = SeqClassifier(len(vocab), n)
                m.load_state_dict(torch.load(f"{OUT_DIR}/{n}.pt", map_location=DEV))
                self.models[n] = (m.to(DEV).eval(), vocab)
            elif n == "bert":
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                tok = AutoTokenizer.from_pretrained(f"{OUT_DIR}/bert")
                m = AutoModelForSequenceClassification.from_pretrained(
                    f"{OUT_DIR}/bert", torch_dtype=torch.bfloat16 if LOW_MEMORY else torch.float32)
                self.models[n] = (m.to(DEV).eval(), tok)
            elif n == "llm":
                self.models[n] = None  # lazy-loaded on first call

    @torch.no_grad()
    def _logits(self, n, texts, bs=32):
        m, aux = self.models[n]
        out = []
        for i in range(0, len(texts), bs):
            b = texts[i:i + bs]
            if n == "bert":
                enc = aux(b, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(DEV)
                out.append(m(**enc).logits.float().cpu())
            else:
                out.append(m(torch.tensor([aux.encode(t, MAX_LEN) for t in b]).to(DEV)).cpu())
        return torch.cat(out)

    def raw_llm(self, texts):
        from llm_zeroshot import llm_prob_fake
        return llm_prob_fake(texts)

    def prob_fake(self, n, texts):
        """Calibrated probability that each text is FAKE."""
        texts = [clean_text(t) for t in texts]
        if n == "llm":
            p = np.clip(self.raw_llm(texts), 1e-6, 1 - 1e-6)
            return 1 / (1 + np.exp(-np.log(p / (1 - p)) / self.temps[n]))
        return torch.softmax(self._logits(n, texts) / self.temps[n], -1)[:, 1].numpy()

    def predict_all(self, text):
        return {n: float(self.prob_fake(n, [text])[0]) for n in self.models}

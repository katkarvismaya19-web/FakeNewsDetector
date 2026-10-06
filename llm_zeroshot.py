"""Zero-shot LLM baseline (no training). P(FAKE) = softmax over the LLM's next-token logits for FAKE vs REAL."""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import LLM_MODEL

PROMPT = ("You are a fact-checker for Indian news. Read the news below (Hindi or English) and decide if it is "
          "fake/misleading or real. Answer with exactly one word: FAKE or REAL.\n\nNews: {text}")
_tok = _model = None


def _load():
    global _tok, _model
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(LLM_MODEL)
        _model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto").eval()
    return _tok, _model


@torch.no_grad()
def llm_prob_fake(texts):
    tok, m = _load()
    fake_id = tok.encode("FAKE", add_special_tokens=False)[0]
    real_id = tok.encode("REAL", add_special_tokens=False)[0]
    probs = []
    for t in texts:
        msgs = [{"role": "user", "content": PROMPT.format(text=t[:1000])}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(m.device)
        logits = m(ids).logits[0, -1].float()
        probs.append(torch.softmax(torch.stack([logits[fake_id], logits[real_id]]), 0)[0].item())
    return np.array(probs)

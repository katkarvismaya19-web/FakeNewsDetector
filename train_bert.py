"""Fine-tune MuRIL (BERT for Indian languages) on Hindi fake news. GPU strongly recommended (Colab T4 is fine)."""
import json, time
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

from calibration import fit_temperature
from config import BERT_MODEL, LABEL_COL, MAX_LEN, OUT_DIR, SEED, TEXT_COL
from data import load_splits


def main():
    tr, va, _ = load_splits()
    tok = AutoTokenizer.from_pretrained(BERT_MODEL)

    def to_ds(df):
        d = Dataset.from_pandas(df.rename(columns={TEXT_COL: "text", LABEL_COL: "labels"}))
        return d.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN), batched=True)

    dtr, dva = to_ds(tr), to_ds(va)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL, num_labels=2, id2label={0: "REAL", 1: "FAKE"}, label2id={"REAL": 0, "FAKE": 1})

    args = TrainingArguments(
        output_dir=f"{OUT_DIR}/bert_ckpt", eval_strategy="epoch", save_strategy="epoch",
        learning_rate=2e-5, per_device_train_batch_size=16, per_device_eval_batch_size=32,
        num_train_epochs=5, weight_decay=0.01, warmup_ratio=0.1,
        load_best_model_at_end=True, metric_for_best_model="f1", save_total_limit=1,
        fp16=torch.cuda.is_available(), seed=SEED, report_to="none")

    def metrics(p):
        return {"f1": f1_score(p.label_ids, np.argmax(p.predictions, -1), average="macro")}

    trainer = Trainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    t0 = time.time()
    trainer.train()
    dt = time.time() - t0

    T = fit_temperature(torch.tensor(trainer.predict(dva).predictions), torch.tensor(va[LABEL_COL].values))
    trainer.save_model(f"{OUT_DIR}/bert")
    tok.save_pretrained(f"{OUT_DIR}/bert")
    json.dump({"temperature": T, "train_time_s": round(dt, 1)}, open(f"{OUT_DIR}/bert_meta.json", "w"))
    print(f"[bert] saved. temperature={T:.3f}")


if __name__ == "__main__":
    main()

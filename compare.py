"""Compare RNN vs LSTM vs BERT vs LLM on the held-out test set.
Usage: python compare.py --models rnn,lstm,bert,llm --llm_samples 300
Outputs: outputs/results.csv, outputs/comparison.png, outputs/confusion_matrices.png
"""
import argparse, json, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

from calibration import ece, fit_temperature
from config import LABEL_COL, OUT_DIR, SEED, TEXT_COL
from data import load_splits
from predict import Detector

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="rnn,lstm,bert,llm")
ap.add_argument("--llm_samples", type=int, default=300, help="LLM is slow; evaluated on a stratified sample")
a = ap.parse_args()
names = a.models.split(",")

_, va, te = load_splits()
det = Detector(names)


def sample(df, n):
    if len(df) <= n:
        return df
    return df.groupby(LABEL_COL, group_keys=False).apply(
        lambda g: g.sample(int(round(n * len(g) / len(df))), random_state=SEED))


if "llm" in names:  # calibrate the LLM's confidence on validation data, same as the trained models
    vs = sample(va, a.llm_samples)
    p = np.clip(det.raw_llm(vs[TEXT_COL].tolist()), 1e-6, 1 - 1e-6)
    lg = torch.tensor(np.log(p / (1 - p)), dtype=torch.float)
    T = fit_temperature(torch.stack([torch.zeros_like(lg), lg], 1), torch.tensor(vs[LABEL_COL].values))
    json.dump({"temperature": T}, open(f"{OUT_DIR}/llm_meta.json", "w"))
    det.temps["llm"] = T

rows = []
fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4))
for ax, n in zip(np.atleast_1d(axes), names):
    df = sample(te, a.llm_samples) if n == "llm" else te
    y = df[LABEL_COL].values
    t0 = time.time()
    p = det.prob_fake(n, df[TEXT_COL].tolist())
    ms = (time.time() - t0) / len(df) * 1000
    pred = (p >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    meta = json.load(open(f"{OUT_DIR}/{n}_meta.json"))
    rows.append({"model": n.upper(), "accuracy": accuracy_score(y, pred), "precision": pr, "recall": rc,
                 "f1_macro": f1, "roc_auc": roc_auc_score(y, p), "ece (lower=better)": ece(p, y),
                 "train_time_s": meta.get("train_time_s", 0), "ms_per_sample": ms, "n_test": len(df)})
    ConfusionMatrixDisplay(confusion_matrix(y, pred, labels=[0, 1]),
                           display_labels=["REAL", "FAKE"]).plot(ax=ax, colorbar=False)
    ax.set_title(n.upper())
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/confusion_matrices.png", dpi=150)

res = pd.DataFrame(rows)

lang_rows = []  # same metrics split by language: shows whether a model handles Hindi and English equally well
for n in names:
    df = sample(te, a.llm_samples) if n == "llm" else te
    p = det.prob_fake(n, df[TEXT_COL].tolist())
    for lg in ("hi", "en"):
        m = (df["lang"] == lg).values
        if m.sum() == 0 or len(set(df[LABEL_COL].values[m])) < 2:
            continue
        y, pred = df[LABEL_COL].values[m], (p[m] >= 0.5).astype(int)
        lang_rows.append({"model": n.upper(), "language": {"hi": "Hindi", "en": "English"}[lg], "n": int(m.sum()),
                          "accuracy": accuracy_score(y, pred),
                          "f1_macro": precision_recall_fscore_support(y, pred, average="macro", zero_division=0)[2],
                          "roc_auc": roc_auc_score(y, p[m])})
by_lang = pd.DataFrame(lang_rows)
by_lang.to_csv(f"{OUT_DIR}/results_by_language.csv", index=False)
res.to_csv(f"{OUT_DIR}/results.csv", index=False)
print(res.round(4).to_string(index=False))
print("\nBy language:\n" + by_lang.round(4).to_string(index=False))

ax = res.set_index("model")[["accuracy", "f1_macro", "roc_auc"]].plot.bar(figsize=(8, 4.5), rot=0)
ax.set_ylim(0, 1)
ax.set_title("Fake news detection (Hindi + English): RNN vs LSTM vs BERT vs LLM")
for c in ax.containers:
    ax.bar_label(c, fmt="%.2f", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/comparison.png", dpi=150)
print(f"\nSaved {OUT_DIR}/results.csv, results_by_language.csv, comparison.png, confusion_matrices.png")

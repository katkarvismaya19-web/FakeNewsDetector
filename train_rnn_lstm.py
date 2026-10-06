"""Train RNN and LSTM baselines.  Usage: python train_rnn_lstm.py [rnn|lstm|both]"""
import json, pickle, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from calibration import fit_temperature
from config import LABEL_COL, MAX_LEN, OUT_DIR, SEED, TEXT_COL
from data import load_splits
from models_rnn import SeqClassifier, Vocab

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def to_ds(df, vocab):
    X = torch.tensor([vocab.encode(t, MAX_LEN) for t in df[TEXT_COL]])
    return TensorDataset(X, torch.tensor(df[LABEL_COL].values))


@torch.no_grad()
def get_logits(model, loader):
    model.eval()
    return torch.cat([model(X.to(DEV)).cpu() for X, _ in loader])


def train(cell, epochs=15, patience=3):
    torch.manual_seed(SEED)
    tr, va, _ = load_splits()
    vocab = Vocab(tr[TEXT_COL])
    tl = DataLoader(to_ds(tr, vocab), batch_size=64, shuffle=True)
    vl = DataLoader(to_ds(va, vocab), batch_size=128)
    yv = torch.tensor(va[LABEL_COL].values)

    model = SeqClassifier(len(vocab), cell).to(DEV)
    counts = np.bincount(tr[LABEL_COL], minlength=2)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (2 * counts), dtype=torch.float).to(DEV))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    best, bad, t0 = float("inf"), 0, time.time()
    for ep in range(epochs):
        model.train()
        for X, y in tl:
            opt.zero_grad()
            loss = loss_fn(model(X.to(DEV)), y.to(DEV))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        vloss = nn.functional.cross_entropy(get_logits(model, vl), yv).item()
        print(f"[{cell}] epoch {ep + 1}  val_loss={vloss:.4f}")
        if vloss < best:
            best, bad = vloss, 0
            torch.save(model.state_dict(), f"{OUT_DIR}/{cell}.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"[{cell}] early stop")
                break

    model.load_state_dict(torch.load(f"{OUT_DIR}/{cell}.pt", map_location=DEV))
    T = fit_temperature(get_logits(model, vl), yv)
    pickle.dump(vocab, open(f"{OUT_DIR}/{cell}_vocab.pkl", "wb"))
    json.dump({"temperature": T, "train_time_s": round(time.time() - t0, 1)}, open(f"{OUT_DIR}/{cell}_meta.json", "w"))
    print(f"[{cell}] saved. temperature={T:.3f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    for c in (["rnn", "lstm"] if which == "both" else [which]):
        train(c)

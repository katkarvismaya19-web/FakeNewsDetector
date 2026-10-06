from collections import Counter
import torch
import torch.nn as nn


class Vocab:
    def __init__(self, texts, min_freq=2, max_size=30000):
        c = Counter(w for t in texts for w in t.lower().split())
        self.itos = ["<pad>", "<unk>"] + [w for w, f in c.most_common(max_size) if f >= min_freq]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, text, max_len):
        ids = [self.stoi.get(w, 1) for w in text.lower().split()][:max_len]
        return ids + [0] * (max_len - len(ids))

    def __len__(self):
        return len(self.itos)


class SeqClassifier(nn.Module):
    """Bidirectional RNN or LSTM -> masked mean-pool -> 2-way classifier."""

    def __init__(self, vocab_size, cell="lstm", emb=200, hid=128, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        rnn_cls = nn.LSTM if cell == "lstm" else nn.RNN
        self.rnn = rnn_cls(emb, hid, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hid * 2, 2)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1).float()
        out, _ = self.rnn(self.emb(x))
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(self.drop(pooled))

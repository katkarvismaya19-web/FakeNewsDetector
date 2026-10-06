"""Confidence-score validation: temperature scaling + Expected Calibration Error."""
import numpy as np
import torch


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Learn T on the validation set so softmax(logits / T) gives honest probabilities."""
    logits, labels = logits.detach().float(), labels.long()
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def ece(prob_fake: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Expected Calibration Error: gap between stated confidence and actual accuracy (lower = better)."""
    conf = np.maximum(prob_fake, 1 - prob_fake)
    correct = ((prob_fake >= 0.5).astype(int) == labels)
    edges = np.linspace(0.5, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(correct[idx == b].mean() - conf[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))

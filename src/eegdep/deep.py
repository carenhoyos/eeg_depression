"""EEGNet compacta (Lawhern et al., 2018) en PyTorch y entrenamiento con validación por SUJETO.

La predicción por sujeto es la media de las probabilidades de sus épocas; las métricas se calculan
a nivel sujeto, nunca a nivel época. La normalización por canal se ajusta solo con entrenamiento.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold


class EEGNet(nn.Module):
    def __init__(self, n_ch: int, n_t: int, F1: int, D: int, F2: int, kernel_length: int, dropout: float, n_classes: int = 1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
        )
        with torch.no_grad():
            n_feat = self.block2(self.block1(torch.zeros(1, 1, n_ch, n_t))).numel()
        self.head = nn.Linear(n_feat, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: (B, n_ch, n_t)
        z = self.block2(self.block1(x.unsqueeze(1)))
        return self.head(z.flatten(1)).squeeze(-1)


def stack_subjects(epochs_by_subject: dict[str, np.ndarray], subjects: list[str], y_by_subject: dict[str, int]):
    X = np.concatenate([epochs_by_subject[s] for s in subjects])
    y = np.concatenate([np.full(len(epochs_by_subject[s]), y_by_subject[s]) for s in subjects])
    g = np.concatenate([np.full(len(epochs_by_subject[s]), s) for s in subjects])
    return X, y, g


class ChannelScaler:
    """Media y DE por canal, ajustadas solo en entrenamiento."""

    def fit(self, X: np.ndarray) -> "ChannelScaler":
        self.mu = X.mean(axis=(0, 2), keepdims=True)
        self.sd = X.std(axis=(0, 2), keepdims=True) + 1e-6
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd


def subsample_epochs(epochs_by_subject: dict[str, np.ndarray], max_per_subject: int | None, seed: int) -> dict[str, np.ndarray]:
    """Submuestreo aleatorio de épocas por sujeto para acotar el coste en CPU (configurable)."""
    if not max_per_subject:
        return epochs_by_subject
    rng = np.random.default_rng(seed)
    out = {}
    for s, X in epochs_by_subject.items():
        idx = rng.choice(len(X), size=min(len(X), int(max_per_subject)), replace=False)
        out[s] = X[np.sort(idx)]
    return out


def train_eegnet(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, g_va: np.ndarray,
                 deep_cfg: dict[str, Any], seed: int, log_prefix: str = "") -> tuple[EEGNet, dict[str, Any]]:
    torch.manual_seed(seed)
    if deep_cfg.get("torch_threads"):
        torch.set_num_threads(int(deep_cfg["torch_threads"]))
    dev = torch.device(deep_cfg["device"])
    model = EEGNet(X_tr.shape[1], X_tr.shape[2], deep_cfg["F1"], deep_cfg["D"], deep_cfg["F2"], deep_cfg["kernel_length"], deep_cfg["dropout"]).to(dev)
    pos_w = torch.tensor([(y_tr == 0).sum() / max(1, (y_tr == 1).sum())], dtype=torch.float32, device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(model.parameters(), lr=float(deep_cfg["lr"]), weight_decay=float(deep_cfg["weight_decay"]))
    Xt, yt = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)
    Xv = torch.tensor(X_va, dtype=torch.float32, device=dev)
    best, best_state, patience, hist = -np.inf, None, 0, []
    bs = int(deep_cfg["batch_size"])
    rng = np.random.default_rng(seed)
    for ep in range(int(deep_cfg["epochs"])):
        model.train()
        idx = rng.permutation(len(Xt))
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xt[b].to(dev)), yt[b].to(dev))
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xv)).cpu().numpy()
        score = subject_auc(y_va, p, g_va)
        hist.append(float(score))
        if deep_cfg.get("log_every_epochs") and (ep + 1) % int(deep_cfg["log_every_epochs"]) == 0:
            print(f"{log_prefix}época {ep + 1}: AUC val (sujeto) = {score:.3f}", flush=True)
        if score > best:
            best, best_state, patience = score, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= int(deep_cfg["early_stopping_patience"]):
                break
    model.load_state_dict(best_state)
    return model, {"best_val_subject_auc": float(best), "epochs_run": len(hist)}


def predict_subjects(model: EEGNet, X: np.ndarray, g: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(X, dtype=torch.float32, device=torch.device(device)))).cpu().numpy()
    subs = np.unique(g)
    return subs, np.array([p[g == s].mean() for s in subs])


def subject_auc(y_ep: np.ndarray, p_ep: np.ndarray, g_ep: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    subs = np.unique(g_ep)
    ys = np.array([y_ep[g_ep == s][0] for s in subs]); ps = np.array([p_ep[g_ep == s].mean() for s in subs])
    return float(roc_auc_score(ys, ps)) if len(np.unique(ys)) == 2 else float("nan")


def split_train_val_subjects(subjects: np.ndarray, y: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n_splits = max(2, int(round(1 / fraction)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr, va = next(sgkf.split(np.zeros(len(y)), y, subjects))
    return subjects[tr], subjects[va]

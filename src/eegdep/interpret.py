"""Interpretabilidad: efectos univariados, importancia de características, topografías y calibración.
Convenciones gráficas: una sola escala por eje, colormap secuencial de un tono para magnitud,
divergente con punto medio neutro para polaridad, marcas finas, leyenda cuando hay >= 2 series."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score



def univariate_auc(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """AUC de cada característica como predictor univariado (solo con datos de desarrollo)."""
    return pd.Series({c: roc_auc_score(y, X[c].to_numpy()) for c in X.columns}, name="auc")


def cohen_d(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    a, b = X[y == 1], X[y == 0]
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return ((a.mean() - b.mean()) / sp).rename("cohen_d")


def model_feature_importance(calibrated_or_pipeline, feature_names: list[str]) -> pd.Series | None:
    """Importancia nativa: coeficientes (lineales) o impurity (RF), mapeada a través del selector."""
    pipe = calibrated_or_pipeline
    if hasattr(pipe, "calibrated_classifiers_"):
        pipe = pipe.calibrated_classifiers_[0].estimator
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
    elif hasattr(clf, "coef_") and getattr(clf, "kernel", "linear") == "linear":
        vals = np.ravel(clf.coef_)
    else:
        return None
    mask = pipe.named_steps["select"].get_support() if "select" in pipe.named_steps else np.ones(len(feature_names), bool)
    s = pd.Series(0.0, index=feature_names)
    s[np.array(feature_names)[mask]] = vals
    return s.rename("importance")


def permutation_importance_series(model, X: np.ndarray, y: np.ndarray, feature_names: list[str],
                                  n_repeats: int, seed: int, scoring: str, n_jobs: int) -> pd.DataFrame:
    r = permutation_importance(model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=seed, n_jobs=n_jobs)
    return pd.DataFrame({"mean": r.importances_mean, "sd": r.importances_std}, index=feature_names).sort_values("mean", ascending=False)


def _info(channels: list[str], montage: str):
    info = mne.create_info(channels, sfreq=1.0, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage(montage))
    return info


def topomap_grid(values: dict[str, np.ndarray], channels: list[str], montage: str, out: Path, title: str, plot_cfg: dict[str, Any],
                 diverging: bool = True, center: float = 0.0, cbar_label: str = "") -> Path:
    """Un topomapa por banda con escala de color COMPARTIDA. values[banda] -> (n_ch,)."""
    bands = list(values.keys())
    allv = np.concatenate([np.asarray(v) for v in values.values()])
    if diverging:
        lim = float(np.nanmax(np.abs(allv - center)))
        vmin, vmax, cmap = center - lim, center + lim, plot_cfg["diverging_cmap"]
    else:
        vmin, vmax, cmap = float(np.nanmin(allv)), float(np.nanmax(allv)), plot_cfg["sequential_cmap"]
    info = _info(channels, montage)
    fig, axes = plt.subplots(1, len(bands), figsize=(2.6 * len(bands), 3.2))
    axes = np.atleast_1d(axes)
    for ax, b in zip(axes, bands):
        im, _ = mne.viz.plot_topomap(np.asarray(values[b]), info, axes=ax, show=False, cmap=cmap,
                                     vlim=(vmin, vmax), contours=int(plot_cfg["topomap_contours"]), sensors=True)
        ax.set_title(b, fontsize=10)
    cb = fig.colorbar(im, ax=axes.tolist(), shrink=0.7, pad=0.02)
    cb.set_label(cbar_label)
    fig.suptitle(title, fontsize=11)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(plot_cfg["dpi"]), bbox_inches="tight")
    plt.close(fig)
    return out


def band_channel_matrix(series: pd.Series, feature_meta: pd.DataFrame, kind: str, bands: list[str], channels: list[str]) -> dict[str, np.ndarray]:
    """Reorganiza una Serie (por nombre de característica) en {banda: vector por canal} para un tipo dado."""
    out = {}
    for b in bands:
        vec = np.full(len(channels), np.nan)
        for i, c in enumerate(channels):
            name = f"{kind}__{b}__{c}"
            if name in series.index:
                vec[i] = series[name]
        if not np.all(np.isnan(vec)):
            out[b] = vec
    return out


def plot_calibration(y: np.ndarray, p: np.ndarray, n_bins: int, out: Path, title: str, plot_cfg: dict[str, Any]) -> Path:
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], color=plot_cfg["color_reference"], lw=1, ls="--", label="Calibración perfecta")
    ax.plot(mean_pred, frac_pos, marker="o", ms=5, lw=2, color=plot_cfg["color_primary"], label="Modelo")
    ax.set_xlabel("Probabilidad predicha (media por bin)")
    ax.set_ylabel("Fracción observada de positivos")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(plot_cfg["dpi"]), bbox_inches="tight")
    plt.close(fig)
    return out


def plot_top_features(imp: pd.Series, k: int, out: Path, title: str, xlabel: str, plot_cfg: dict[str, Any]) -> Path:
    top = imp.reindex(imp.abs().sort_values(ascending=False).index[:k]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 0.28 * k + 1))
    colors = [plot_cfg["color_primary"] if v >= 0 else plot_cfg["color_negative"] for v in top.values]
    ax.barh(top.index, top.values, color=colors, height=0.6)
    ax.axvline(0, color=plot_cfg["color_reference"], lw=1)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(plot_cfg["dpi"]), bbox_inches="tight")
    plt.close(fig)
    return out

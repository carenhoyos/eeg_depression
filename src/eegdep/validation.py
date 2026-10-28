"""Validación por sujeto: holdout, CV anidada agrupada, métricas con IC bootstrap,
calibración y prueba de permutación. Ninguna partición se hace por ventana/fila sin grupo."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (balanced_accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, StratifiedGroupKFold


# ---------------------------------------------------------------- particiones
def holdout_split(y: np.ndarray, groups: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Reserva ~fraction de los GRUPOS (sujetos), estratificado por clase."""
    n_splits = max(2, int(round(1.0 / fraction)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(sgkf.split(np.zeros(len(y)), y, groups))
    assert not set(groups[dev_idx]) & set(groups[test_idx]), "fuga de sujetos entre dev y holdout"
    return dev_idx, test_idx


def outer_splits(y: np.ndarray, groups: np.ndarray, outer_cfg: dict[str, Any], seed: int) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Lista de (repetición, fold, train_idx, test_idx), siempre agrupada por sujeto."""
    splits = []
    if outer_cfg["scheme"] == "loso":
        for k, (tr, te) in enumerate(LeaveOneGroupOut().split(np.zeros(len(y)), y, groups)):
            splits.append((0, k, tr, te))
        return splits
    for rep in range(int(outer_cfg["n_repeats"])):
        sgkf = StratifiedGroupKFold(n_splits=int(outer_cfg["n_splits"]), shuffle=True, random_state=seed + rep)
        for k, (tr, te) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
            assert not set(groups[tr]) & set(groups[te])
            splits.append((rep, k, tr, te))
    return splits


def inner_cv(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Particiones internas (para GridSearch / calibración) materializadas como lista, agrupadas."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(y)), y, groups))


def resolve_threshold(thr_cfg: dict[str, Any], y_train: np.ndarray) -> float:
    """Umbral de decisión según configuración, calculado SOLO con etiquetas de entrenamiento."""
    strategy = thr_cfg["strategy"]
    if strategy == "fixed":
        return float(thr_cfg["value"])
    if strategy == "train_prevalence":
        return float(np.mean(y_train))
    raise ValueError(f"validation.threshold.strategy desconocida: {strategy}")


# ---------------------------------------------------------------- entrenamiento
def fit_with_search(pipeline, grid: dict, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    val_cfg: dict[str, Any], seed: int, n_jobs: int):
    """GridSearch con CV interna agrupada; opcionalmente calibra sobre CV interna agrupada."""
    cv_in = inner_cv(y, groups, int(val_cfg["inner"]["n_splits"]), seed)
    if grid:
        gs = GridSearchCV(pipeline, grid, cv=cv_in, scoring=val_cfg["scoring"], n_jobs=n_jobs, refit=True)
        gs.fit(X, y)
        best, best_params = gs.best_estimator_, gs.best_params_
    else:
        best, best_params = clone(pipeline).fit(X, y), {}
    cal = val_cfg["calibration"]
    if cal["enabled"]:
        cv_cal = inner_cv(y, groups, int(cal["cv"]), seed + 1)
        model = CalibratedClassifierCV(clone(best), method=cal["method"], cv=cv_cal).fit(X, y)
    else:
        model = best
    return model, best, best_params


def run_nested_cv(pipeline, grid: dict, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  val_cfg: dict[str, Any], seed: int, n_jobs: int, verbose: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    """Predicciones fuera de muestra por sujeto. Devuelve (oof, best_params por fold)."""
    rows, params_log = [], []
    splits = outer_splits(y, groups, val_cfg["outer"], seed)
    for rep, k, tr, te in splits:
        model, _, bp = fit_with_search(pipeline, grid, X[tr], y[tr], groups[tr], val_cfg, seed + 100 * rep + k, n_jobs)
        p = model.predict_proba(X[te])[:, 1]
        thr = resolve_threshold(val_cfg["threshold"], y[tr])
        rows.append(pd.DataFrame({"rep": rep, "fold": k, "idx": te, "group": groups[te], "y": y[te], "p": p, "thr": thr}))
        params_log.append({"rep": rep, "fold": k, **bp})
        if verbose:
            print(f"  rep {rep} fold {k}: n_train={len(tr)} n_test={len(te)} AUC_fold={roc_auc_score(y[te], p):.3f} params={bp}")
    return pd.concat(rows, ignore_index=True), params_log


# ---------------------------------------------------------------- métricas
def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def compute_metrics(y: np.ndarray, p: np.ndarray, threshold: float | np.ndarray, ece_bins: int) -> dict[str, float]:
    """`threshold` puede ser escalar o un vector por observación (umbral de su partición)."""
    yhat = (p >= np.asarray(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p, ece_bins),
        "n": int(len(y)), "n_pos": int(y.sum()),
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray, groups: np.ndarray, threshold: float | np.ndarray, ece_bins: int,
                 n_boot: int, ci: float, seed: int) -> dict[str, tuple[float, float]]:
    """IC por bootstrap remuestreando SUJETOS (grupos) con reemplazo."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    by_group = {g: np.where(groups == g)[0] for g in ug}
    samples: dict[str, list[float]] = {}
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([by_group[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        thr_i = threshold if np.isscalar(threshold) else np.asarray(threshold)[idx]
        for k, v in compute_metrics(y[idx], p[idx], thr_i, ece_bins).items():
            samples.setdefault(k, []).append(v)
    a = (1 - ci) / 2
    return {k: (float(np.nanpercentile(v, 100 * a)), float(np.nanpercentile(v, 100 * (1 - a)))) for k, v in samples.items()}


def summarize_oof(oof: pd.DataFrame, val_cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Métricas por repetición (media ± DE) y, sobre la probabilidad media por sujeto, IC bootstrap."""
    nb = int(val_cfg["ece_bins"])
    per_rep = pd.DataFrame([compute_metrics(g.y.to_numpy(), g.p.to_numpy(), g.thr.to_numpy(), nb) for _, g in oof.groupby("rep")])
    agg = oof.groupby("idx").agg(y=("y", "first"), p=("p", "mean"), thr=("thr", "mean"), group=("group", "first"))
    point = compute_metrics(agg.y.to_numpy(), agg.p.to_numpy(), agg.thr.to_numpy(), nb)
    ci = bootstrap_ci(agg.y.to_numpy(), agg.p.to_numpy(), agg.group.to_numpy(), agg.thr.to_numpy(), nb,
                      int(val_cfg["bootstrap"]["n"]), float(val_cfg["bootstrap"]["ci"]), seed)
    return {"per_repeat_mean": per_rep.mean().to_dict(), "per_repeat_sd": per_rep.std().to_dict(),
            "pooled_point": point, "pooled_ci": ci, "subject_level_predictions": agg}


def permutation_test(pipeline, best_params: dict, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                     val_cfg: dict[str, Any], observed_auc: float, n_perm: int, seed: int, n_jobs: int) -> dict[str, Any]:
    """Permuta etiquetas a nivel de SUJETO y repite la CV externa (1 repetición, sin búsqueda) para estimar
    la distribución nula del AUC. p = (1 + #perm >= observado) / (1 + n_perm)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    label_of_group = {g: int(y[groups == g][0]) for g in ug}
    fixed = clone(pipeline).set_params(**best_params)
    cfg1 = {**val_cfg, "outer": {**val_cfg["outer"], "n_repeats": 1}, "calibration": {**val_cfg["calibration"], "enabled": False}}
    null = []
    for i in range(n_perm):
        perm_labels = dict(zip(ug, rng.permutation([label_of_group[g] for g in ug])))
        yp = np.array([perm_labels[g] for g in groups])
        oof, _ = run_nested_cv(fixed, {}, X, yp, groups, cfg1, seed + i, n_jobs, verbose=False)
        null.append(roc_auc_score(oof.y, oof.p))
    null = np.array(null)
    return {"null_auc_mean": float(null.mean()), "null_auc_p95": float(np.percentile(null, 95)),
            "p_value": float((1 + (null >= observed_auc).sum()) / (1 + n_perm)), "n_permutations": n_perm}

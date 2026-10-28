"""Construcción de pipelines scikit-learn a partir de la configuración."""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC


def make_estimator(spec: dict[str, Any], seed: int):
    params = dict(spec.get("params", {}))
    t = spec["type"]
    if t == "svm":
        return SVC(random_state=seed, **params)
    if t == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=1, **params)
    if t == "logistic":
        return LogisticRegression(random_state=seed, **params)
    raise ValueError(f"Tipo de modelo no soportado: {t}")


def make_pipeline(spec: dict[str, Any], val_cfg: dict[str, Any], seed: int) -> tuple[Pipeline, dict[str, list]]:
    """Pipeline = escalado -> selección -> clasificador. Todo se ajusta solo con entrenamiento."""
    steps = []
    scaler = val_cfg["scaler"]
    if scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler == "robust":
        steps.append(("scaler", RobustScaler()))
    elif scaler != "none":
        raise ValueError(f"scaler desconocido: {scaler}")
    grid: dict[str, list] = {}
    fsel = val_cfg["feature_selection"]
    if fsel["method"] == "kbest_f":
        steps.append(("select", SelectKBest(f_classif)))
        grid["select__k"] = list(fsel["k_grid"])
    elif fsel["method"] != "none":
        raise ValueError(f"feature_selection.method desconocido: {fsel['method']}")
    steps.append(("clf", make_estimator(spec, seed)))
    grid.update({k: list(v) for k, v in spec.get("grid", {}).items()})
    return Pipeline(steps), grid


def selected_feature_mask(pipeline: Pipeline, n_features: int):
    """Máscara booleana de las características que llegan al clasificador."""
    import numpy as np

    if "select" in pipeline.named_steps:
        return pipeline.named_steps["select"].get_support()
    return np.ones(n_features, dtype=bool)

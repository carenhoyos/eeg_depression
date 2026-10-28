"""Bundle de modelo versionado e inferencia por registro.

El bundle contiene todo lo necesario para servir el modelo sin acceder al CSV de entrenamiento:
pipeline calibrado, nombres/metadatos de características, configuración congelada, prevalencia de
entrenamiento, umbral, métricas de validación y distribuciones de referencia (percentiles y
media/DE por característica en el grupo de referencia).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import build_feature_table
from .io import EEGFeatureSet, build_feature_set


@dataclass
class ModelBundle:
    model_name: str
    version: str
    created_at: str
    model: Any                               # pipeline calibrado (predict_proba)
    uncalibrated: Any                        # pipeline sin calibrar (coeficientes / importancias)
    feature_names: list[str]
    feature_meta: pd.DataFrame
    pipeline_cfg: dict[str, Any]
    train_prevalence: float
    threshold: float
    validation: dict[str, Any]
    reference: dict[str, pd.DataFrame]       # {"controls": percentiles, "cases": ..., "all": ...}
    reference_stats: dict[str, pd.DataFrame] # {"controls": DataFrame[mean, sd], ...}
    n_train: dict[str, int] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "ModelBundle":
        with open(path, "rb") as f:
            return pickle.load(f)

    def card(self) -> dict[str, Any]:
        """Resumen tipo model card para el reporte y el endpoint /model."""
        return {"model_name": self.model_name, "version": self.version, "created_at": self.created_at,
                "n_features": len(self.feature_names), "train_prevalence": self.train_prevalence,
                "threshold": self.threshold, "n_train": self.n_train, "validation": self.validation}


def build_bundle(model_name: str, version: str, final_model: Any, uncalibrated: Any, feature_names: list[str],
                 feature_meta: pd.DataFrame, pipeline_cfg: dict[str, Any], X_dev: pd.DataFrame, y_dev: np.ndarray,
                 threshold: float, validation: dict[str, Any], percentiles: list[int]) -> ModelBundle:
    groups = {"controls": X_dev[y_dev == 0], "cases": X_dev[y_dev == 1], "all": X_dev}
    reference = {k: v.quantile([p / 100 for p in percentiles]).set_axis([f"p{p}" for p in percentiles]) for k, v in groups.items()}
    stats = {k: pd.DataFrame({"mean": v.mean(), "sd": v.std(ddof=1)}) for k, v in groups.items()}
    cfg_frozen = {k: v for k, v in pipeline_cfg.items() if not k.startswith("_")}
    return ModelBundle(model_name=model_name, version=version, created_at=datetime.now(timezone.utc).isoformat(),
                       model=final_model, uncalibrated=uncalibrated, feature_names=feature_names, feature_meta=feature_meta,
                       pipeline_cfg=cfg_frozen, train_prevalence=float(y_dev.mean()), threshold=float(threshold),
                       validation=validation, reference=reference, reference_stats=stats,
                       n_train={"controls": int((y_dev == 0).sum()), "cases": int((y_dev == 1).sum())})


def features_from_csv(df: pd.DataFrame, pipeline_cfg: dict[str, Any]) -> tuple[EEGFeatureSet, pd.DataFrame]:
    """CSV con la estructura de entrenamiento (con o sin columna de etiqueta) -> set y tabla de características."""
    d = pipeline_cfg["data"]
    if d["label_column"] not in df.columns:
        df = df.copy()
        df[d["label_column"]] = d["negative_class"]   # etiqueta ficticia; no se usa en inferencia
        unknown_label = True
    else:
        unknown_label = False
    df = df.drop(columns=[c for c in d["drop_columns"] if c in df.columns])
    fs = build_feature_set(df, pipeline_cfg)
    if unknown_label:
        fs.y = np.full(fs.n, -1)
        fs.labels_raw = np.array(["unknown"] * fs.n)
    X, _ = build_feature_table(fs, pipeline_cfg)
    return fs, X


def posterior_at_prevalence(p_cal: float, train_prevalence: float, pretest: float) -> tuple[float, float]:
    """Reajusta una probabilidad calibrada a otra prevalencia vía razón de verosimilitud.
    Devuelve (posterior, LR)."""
    eps = 1e-9
    odds = lambda q: (q + eps) / (1 - q + eps)
    lr = odds(p_cal) / odds(train_prevalence)
    post_odds = lr * odds(pretest)
    return float(post_odds / (1 + post_odds)), float(lr)


def z_scores(x: pd.Series, stats: pd.DataFrame) -> pd.Series:
    return (x - stats["mean"]) / stats["sd"].replace(0, np.nan)


def linear_contributions(bundle: ModelBundle, x_row: np.ndarray) -> pd.Series | None:
    """Contribución por característica = coef * valor escalado (solo modelos lineales)."""
    pipe = bundle.uncalibrated
    clf = pipe.named_steps["clf"]
    coef = getattr(clf, "coef_", None)
    if coef is None or getattr(clf, "kernel", "linear") != "linear":
        return None
    z = x_row.reshape(1, -1)
    names = np.array(bundle.feature_names)
    if "scaler" in pipe.named_steps:
        z = pipe.named_steps["scaler"].transform(z)
    if "select" in pipe.named_steps:
        mask = pipe.named_steps["select"].get_support()
        z, names = z[:, mask], names[mask]
    return pd.Series(np.ravel(coef) * z.ravel(), index=names)


def predict_record(bundle: ModelBundle, X: pd.DataFrame, pretest_prevalence: float, reference_group: str) -> list[dict[str, Any]]:
    """Predicción por fila: probabilidad calibrada, LR, posterior, z-scores y contribuciones."""
    X = X.reindex(columns=bundle.feature_names)
    if X.isna().any().any():
        missing = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Características faltantes o no calculables: {missing[:10]}")
    probs = bundle.model.predict_proba(X.to_numpy(float))[:, 1]
    stats = bundle.reference_stats[reference_group]
    out = []
    for i, (idx, row) in enumerate(X.iterrows()):
        post, lr = posterior_at_prevalence(float(probs[i]), bundle.train_prevalence, pretest_prevalence)
        out.append({"record_id": str(idx), "p_calibrated": float(probs[i]), "likelihood_ratio": lr,
                    "pretest_prevalence": pretest_prevalence, "p_posterior": post,
                    "above_threshold": bool(probs[i] >= bundle.threshold),
                    "z": z_scores(row, stats), "contributions": linear_contributions(bundle, row.to_numpy(float))})
    return out

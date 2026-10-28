"""Lectura del CSV y estandarización a una estructura EEG por sujeto, banda y canal."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import renamed_channels


@dataclass
class EEGFeatureSet:
    """Características por registro, en arreglos indexados por (sujeto, banda, canal)."""

    subjects: np.ndarray                 # id de sujeto por fila (n,)
    groups: np.ndarray                   # id de grupo para validación (n,)
    y: np.ndarray                        # 0 = control, 1 = depresión (n,)
    labels_raw: np.ndarray               # etiqueta textual original (n,)
    bands: list[str]
    channels: list[str]                  # nombres renombrados (10-20 moderno)
    power: np.ndarray                    # (n, n_bands, n_ch)
    coherence: np.ndarray | None         # (n, n_bands, n_ch, n_ch), simétrica, diag = nan
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)  # columnas no-EEG
    qc: pd.DataFrame = field(default_factory=pd.DataFrame)    # resultado de QC por fila

    @property
    def n(self) -> int:
        return len(self.y)

    def subset(self, idx: np.ndarray) -> "EEGFeatureSet":
        return EEGFeatureSet(
            subjects=self.subjects[idx], groups=self.groups[idx], y=self.y[idx],
            labels_raw=self.labels_raw[idx], bands=self.bands, channels=self.channels,
            power=self.power[idx], coherence=None if self.coherence is None else self.coherence[idx],
            meta=self.meta.iloc[idx].reset_index(drop=True),
            qc=self.qc.iloc[idx].reset_index(drop=True) if len(self.qc) else self.qc,
        )


def load_csv(path, drop_columns: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    drop = [c for c in (drop_columns or []) if c in df.columns]
    return df.drop(columns=drop)


def reserved_columns(data_cfg: dict[str, Any]) -> list[str]:
    """Columnas que nunca son características: etiqueta, sujeto, tiempo, época."""
    keys = ("label_column", "subject_column", "time_column", "epoch_column")
    return [data_cfg[k] for k in keys if data_cfg.get(k)]


def parse_columns(columns: list[str], patterns: dict[str, str], exclude: list[str] | None = None) -> pd.DataFrame:
    """Clasifica cada columna según los patrones del YAML -> (name, kind, band, ch1, ch2)."""
    compiled = {k: re.compile(v) for k, v in patterns.items()}
    exclude = set(exclude or [])
    rows = []
    for c in columns:
        rec = {"name": c, "kind": "other", "band": None, "ch1": None, "ch2": None}
        for kind, rx in ([] if c in exclude else compiled.items()):
            m = rx.match(c)
            if m:
                rec.update(kind=kind, **{k: v for k, v in m.groupdict().items() if k in ("band", "ch1", "ch2")})
                break
        rows.append(rec)
    return pd.DataFrame(rows)


def inspect_dataframe(df: pd.DataFrame, data_cfg: dict[str, Any]) -> dict[str, Any]:
    """Inspección genérica: dimensiones, tipos, faltantes, grupos de columnas, rangos."""
    parsed = parse_columns(list(df.columns), data_cfg["column_patterns"], reserved_columns(data_cfg))
    summary: dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "dtypes": df.dtypes.astype(str).value_counts().to_dict(),
        "missing_by_column": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "column_kinds": parsed["kind"].value_counts().to_dict(),
        "bands_found": sorted(parsed.loc[parsed.kind == "power", "band"].dropna().unique().tolist()),
        "channels_found": parsed.loc[parsed.kind == "power", "ch1"].dropna().unique().tolist(),
        "has_subject_column": data_cfg.get("subject_column") in df.columns,
        "has_time_column": data_cfg.get("time_column") in df.columns,
        "label_counts": df[data_cfg["label_column"]].value_counts().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    num = df.select_dtypes("number")
    for kind in ("power", "coherence"):
        cols = parsed.loc[parsed.kind == kind, "name"].tolist()
        if cols:
            v = num[cols].to_numpy().ravel()
            summary[f"{kind}_range"] = {
                "min": float(np.nanmin(v)), "p01": float(np.nanpercentile(v, 1)),
                "median": float(np.nanmedian(v)), "p99": float(np.nanpercentile(v, 99)),
                "max": float(np.nanmax(v)),
            }
    return summary


def build_feature_set(df: pd.DataFrame, cfg: dict[str, Any]) -> EEGFeatureSet:
    """Convierte el CSV ancho en arreglos (n, banda, canal) y (n, banda, canal, canal)."""
    d = cfg["data"]
    parsed = parse_columns(list(df.columns), d["column_patterns"], reserved_columns(d))
    bands = list(cfg["bands"].keys())
    ch_orig = list(cfg["channels"]["list"])
    ch_new = renamed_channels(cfg)
    rename = cfg["channels"]["rename"]
    ch_index = {rename.get(c, c): i for i, c in enumerate(ch_orig)}

    # --- etiquetas ---
    lab = df[d["label_column"]].astype(str).to_numpy()
    allowed = {d["positive_class"]: 1, d["negative_class"]: 0}
    unknown = sorted(set(lab) - set(allowed))
    if unknown:
        raise ValueError(f"Etiquetas no contempladas en config: {unknown}")
    y = np.array([allowed[v] for v in lab], dtype=int)

    # --- sujeto / grupo ---
    if d.get("subject_column") and d["subject_column"] in df.columns:
        subjects = df[d["subject_column"]].astype(str).to_numpy()
    else:
        subjects = np.array([f"row{i:04d}" for i in range(len(df))])
    groups = pd.factorize(subjects)[0]

    # --- potencia ---
    pw = parsed[parsed.kind == "power"]
    found_bands = set(pw.band)
    missing_bands = [b for b in bands if b not in found_bands]
    if missing_bands:
        raise ValueError(f"Bandas definidas en config ausentes en el CSV: {missing_bands}")
    extra = sorted(found_bands - set(bands))
    if extra:
        print(f"[io] Aviso: bandas en CSV no usadas por config: {extra}")
    power = np.full((len(df), len(bands), len(ch_new)), np.nan)
    for _, r in pw.iterrows():
        chn = rename.get(r.ch1, r.ch1)
        if r.band in bands and chn in ch_index:
            power[:, bands.index(r.band), ch_index[chn]] = df[r["name"]].to_numpy(float)
    if np.isnan(power).any():
        bad = [(bands[b], ch_new[c]) for b, c in zip(*np.where(np.isnan(power).all(axis=0)))]
        raise ValueError(f"Faltan columnas de potencia para: {bad[:10]}")

    # --- coherencia ---
    coh_cols = parsed[parsed.kind == "coherence"]
    coherence = None
    if len(coh_cols):
        coherence = np.full((len(df), len(bands), len(ch_new), len(ch_new)), np.nan)
        for _, r in coh_cols.iterrows():
            a, b = rename.get(r.ch1, r.ch1), rename.get(r.ch2, r.ch2)
            if r.band in bands and a in ch_index and b in ch_index:
                v = df[r["name"]].to_numpy(float)
                coherence[:, bands.index(r.band), ch_index[a], ch_index[b]] = v
                coherence[:, bands.index(r.band), ch_index[b], ch_index[a]] = v

    other = parsed.loc[parsed.kind == "other", "name"].tolist()
    meta = df[other].copy() if other else pd.DataFrame(index=range(len(df)))
    return EEGFeatureSet(subjects=subjects, groups=groups, y=y, labels_raw=lab, bands=bands,
                         channels=ch_new, power=power, coherence=coherence, meta=meta)


def to_long(fs: EEGFeatureSet) -> pd.DataFrame:
    """Tabla larga (sujeto, etiqueta, banda, canal, potencia) para auditoría/exportación."""
    n, nb, nc = fs.power.shape
    idx = pd.MultiIndex.from_product([fs.subjects, fs.bands, fs.channels], names=["subject", "band", "channel"])
    out = pd.DataFrame({"power": fs.power.reshape(-1)}, index=idx).reset_index()
    out["y"] = np.repeat(fs.y, nb * nc)
    return out

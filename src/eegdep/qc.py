"""Control de calidad por registro sobre características ya extraídas."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .io import EEGFeatureSet


def find_duplicate_power_rows(power: np.ndarray) -> list[tuple[int, int]]:
    """Pares (i, j), i<j, cuyo bloque completo de potencia es idéntico."""
    flat = power.reshape(power.shape[0], -1)
    seen: dict[bytes, int] = {}
    pairs = []
    for i, row in enumerate(flat):
        key = row.tobytes()
        if key in seen:
            pairs.append((seen[key], i))
        else:
            seen[key] = i
    return pairs


def robust_z(x: np.ndarray) -> np.ndarray:
    """z robusto por columna: (x - mediana) / (1.4826 * MAD)."""
    med = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - med), axis=0) * 1.4826
    mad = np.where(mad == 0, np.nan, mad)
    return (x - med) / mad


def qc_table(fs: EEGFeatureSet, qc_cfg: dict[str, Any]) -> pd.DataFrame:
    """Por fila: nº de features de potencia atípicas, fracción, coherencia fuera de rango, fallo QC."""
    pw = fs.power.reshape(fs.n, -1)
    if qc_cfg["log_transform_power"]:
        pw = np.log10(np.clip(pw, np.finfo(float).tiny, None))
    z = robust_z(pw)
    thr = float(qc_cfg["robust_z_threshold"])
    flagged = np.abs(z) > thr
    n_flag = flagged.sum(axis=1)
    frac = n_flag / flagged.shape[1]
    out = pd.DataFrame({
        "subject": fs.subjects,
        "y": fs.y,
        "n_power_outliers": n_flag,
        "frac_power_outliers": frac,
        "max_abs_robust_z": np.nanmax(np.abs(z), axis=1),
        "nonpositive_power": (fs.power <= 0).reshape(fs.n, -1).sum(axis=1),
    })
    if fs.coherence is not None:
        lo, hi = qc_cfg["coherence_range"]
        c = fs.coherence.reshape(fs.n, -1)
        out["coh_out_of_range"] = np.nansum((c < lo) | (c > hi), axis=1)
    else:
        out["coh_out_of_range"] = 0
    out["qc_fail"] = (frac > float(qc_cfg["max_flagged_fraction"])) | (out["nonpositive_power"] > 0) | (out["coh_out_of_range"] > 0)
    return out


def apply_qc(fs: EEGFeatureSet, qc_cfg: dict[str, Any]) -> tuple[EEGFeatureSet, dict[str, Any]]:
    """Aplica política de duplicados y de filas que fallan QC. Devuelve el set y un informe."""
    report: dict[str, Any] = {"n_input": fs.n}
    dups = find_duplicate_power_rows(fs.power)
    report["duplicate_power_pairs"] = [(fs.subjects[i], fs.subjects[j]) for i, j in dups]
    groups = fs.groups.copy()
    keep = np.ones(fs.n, dtype=bool)
    policy = qc_cfg["duplicate_policy"]
    for i, j in dups:
        if policy == "same_group":
            groups[groups == groups[j]] = groups[i]
        elif policy == "drop":
            keep[j] = False
    report["duplicate_policy"] = policy

    table = qc_table(fs, qc_cfg)
    report["n_qc_fail"] = int(table["qc_fail"].sum())
    report["qc_fail_by_class"] = table.groupby("y")["qc_fail"].sum().to_dict()
    if qc_cfg["drop_rows_failing_qc"]:
        keep &= ~table["qc_fail"].to_numpy()
    fs.groups = groups
    fs.qc = table
    out = fs.subset(np.where(keep)[0])
    report["n_output"] = out.n
    report["n_discarded"] = int(fs.n - out.n)
    report["n_unique_groups"] = int(len(np.unique(out.groups)))
    return out, report

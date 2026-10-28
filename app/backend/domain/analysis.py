"""Caso de uso: archivo de entrada -> informe. Sin FastAPI ni E/S HTTP; se puede ejecutar en un worker aparte."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eegdep.config import set_seeds
from eegdep.features import build_feature_table
from eegdep.inference import ModelBundle, features_from_csv, predict_record
from eegdep.raw_features import raw_to_feature_set, read_raw_file

from ..core.config import Settings
from ..services import report as rp


def detect_input_kind(filename: str, settings: Settings) -> str | None:
    ext = Path(filename).suffix.lower()
    for kind, exts in settings.input.formats.items():
        if ext in exts:
            return kind
    return None


def run_analysis(analysis_id: str, workdir: Path, input_path: Path, kind: str, recording: dict[str, Any],
                 pretest: float, sampling_rate_hz: float | None, settings: Settings, bundle: ModelBundle,
                 pipeline_cfg: dict[str, Any]) -> dict[str, Any]:
    """Ejecuta el pipeline completo y devuelve el informe (dict serializable). Escribe figuras y PDF en workdir."""
    rep = settings.report
    set_seeds(int(pipeline_cfg["project"]["seed"]))
    warnings: list[str] = []
    log: dict[str, Any] = {}
    if kind == "csv_features":
        fs, X = features_from_csv(pd.read_csv(input_path), pipeline_cfg)
    else:
        raw = read_raw_file(input_path, kind, pipeline_cfg, sampling_rate_hz or settings.input.raw.default_sampling_rate_hz)
        fs, log = raw_to_feature_set(raw, pipeline_cfg, settings.input.raw.model_dump(), int(pipeline_cfg["project"]["seed"]), subject_id=input_path.stem)
        X, _ = build_feature_table(fs, pipeline_cfg)
        warnings.append("Características calculadas desde señal cruda con el pipeline propio; el modelo se entrenó con un extractor distinto (riesgo de desplazamiento de dominio).")
    if fs.n > 1:
        warnings.append(f"El archivo contiene {fs.n} registros; se reporta cada uno por separado.")

    preds = predict_record(bundle, X, pretest, rep.reference_group)
    figdir = workdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    figs: dict[str, Path] = {}
    results: list[dict[str, Any]] = []
    band_idx = fs.bands.index(rep.coherence_matrix_band)
    for i, r in enumerate(preds):
        rid, x_row = r["record_id"], X.iloc[i]
        topo = rp.topography_z(r["z"], bundle, rep)
        patient_bands, ref_bands = rp.band_summary(x_row, bundle, rep)
        figs[f"probability_{rid}"] = rp.fig_probability(r, rep, figdir / f"probability_{rid}.png")
        figs[f"band_power_{rid}"] = rp.fig_band_power(patient_bands, ref_bands, rep, figdir / f"band_power_{rid}.png")
        if (tp := rp.fig_topomaps(topo, bundle, rep, figdir / f"topomaps_{rid}.png")):
            figs[f"topomaps_{rid}"] = tp
        coh_json: dict[str, Any] = {}
        if fs.coherence is not None:
            m = fs.coherence[i, band_idx]
            figs[f"coherence_{rid}"] = rp.fig_coherence(m, fs.channels, rep.coherence_matrix_band, rep, figdir / f"coherence_{rid}.png")
            coh_json = {"band": rep.coherence_matrix_band, "channels": fs.channels, "values": np.nan_to_num(m, nan=0.0).round(2).tolist()}
        contribs: list[dict[str, Any]] = []
        if r["contributions"] is not None:
            figs[f"contributions_{rid}"] = rp.fig_contributions(r["contributions"], rep, figdir / f"contributions_{rid}.png")
            top = r["contributions"].reindex(r["contributions"].abs().sort_values(ascending=False).index[: rep.top_contributions])
            contribs = [{"feature": k, "contribution": float(v)} for k, v in top.items()]
        results.append({"record_id": rid, "p_calibrated": r["p_calibrated"], "likelihood_ratio": r["likelihood_ratio"],
                        "pretest_prevalence": r["pretest_prevalence"], "p_posterior": r["p_posterior"], "above_threshold": r["above_threshold"],
                        "tier": rp.assign_tier(r["p_posterior"], rep), "findings": rp.findings_table(x_row, r["z"], bundle, rep),
                        "contributions": contribs, "band_relative_power": patient_bands, "band_reference": ref_bands,
                        "topography_z": topo, "coherence_matrix": coh_json})

    prefix = settings.app.api_prefix
    report = {
        "analysis_id": analysis_id, "created_at": datetime.now(timezone.utc).isoformat(), "app_version": settings.app.version,
        "status": "done", "recording": recording,
        "quality": {"input_kind": kind, "n_records": int(fs.n), "warnings": warnings, "processing_log": log},
        "model": bundle.card(), "results": results,
        "figures": {k: f"{prefix}/analyses/{analysis_id}/figures/{v.name}" for k, v in figs.items()},
        "pdf_url": f"{prefix}/analyses/{analysis_id}/report.pdf",
        "disclaimer": rep.disclaimer.strip(), "limitations": rep.limitations,
    }
    rp.build_pdf(report, figs, rep, workdir / "report.pdf")
    return report

"""Caso de uso: archivo(s) de entrada -> informe. Sin FastAPI ni E/S HTTP; ejecutable en un worker aparte."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from eegdep.config import set_seeds
from eegdep.features import build_feature_table
from eegdep.inference import ModelBundle, features_from_csv, predict_record
from eegdep.raw_features import raw_to_feature_set, read_raw_file

from ..core.config import Settings
from ..services import report as rp


def detect_input_kind(filename: str, settings: Settings) -> Optional[str]:
    ext = Path(filename).suffix.lower()
    for kind, exts in settings.input.formats.items():
        if ext in exts:
            return kind
    return None


def _signal_excerpt(raw, settings: Settings) -> dict[str, Any]:
    """Fragmento submuestreado de la señal limpia + PSD por canal, para el visor interactivo."""
    sig = settings.signal_viewer
    sf = float(raw.info["sfreq"])
    dur = float(sig.excerpt_duration_s)
    total = float(raw.times[-1])
    t0 = float(sig.excerpt_start_s) if sig.excerpt_start_s is not None else max(0.0, total / 2 - dur / 2)
    t0 = min(t0, max(0.0, total - dur))
    seg = raw.copy().crop(tmin=t0, tmax=min(total, t0 + dur))
    step = max(1, int(round(sf / float(sig.excerpt_sfreq))))
    data = (seg.get_data() * 1e6)[:, ::step]
    n_seg = int(float(sig.psd_nperseg_s) * sf)
    spec = raw.compute_psd(method="welch", fmin=float(sig.psd_fmin), fmax=float(sig.psd_fmax), n_fft=n_seg, n_per_seg=n_seg, n_overlap=n_seg // 2, verbose=False)
    psd, freqs = spec.get_data(return_freqs=True)
    return {"sfreq": sf / step, "channels": list(raw.ch_names), "t0": float(seg.times[0] + raw.first_time), "duration_s": float(seg.times[-1]),
            "data": np.round(data, 2).tolist(), "psd_freqs": np.round(freqs, 2).tolist(),
            "psd": np.round(np.log10(psd * 1e12 + 1e-12), 3).tolist()}


def _assessment(r: dict[str, Any], tier: dict[str, Any], findings: list[dict[str, Any]], rep) -> str:
    """Texto de valoración preliminar a partir de la plantilla configurada (sin frases fijas en el código)."""
    top = findings[: rep.assessment_max_findings]
    desc = "; ".join(f"{f['kind']} {f['band']} en {f['location']} {'por encima' if f['direction'] == 'above' else 'por debajo'} de la referencia (z={f['z']:+.1f})" for f in top)
    return rep.assessment_template.format(
        p_posterior=100 * r["p_posterior"], p_calibrated=100 * r["p_calibrated"], pretest=100 * r["pretest_prevalence"],
        lr=r["likelihood_ratio"], tier_name=tier["name"], tier_text=tier["text"],
        findings=desc if desc else rep.assessment_no_findings_text).strip()


def run_analysis(analysis_id: str, workdir: Path, input_path: Path, kind: str, recording: dict[str, Any],
                 pretest: float, sampling_rate_hz: Optional[float], settings: Settings, bundle: ModelBundle,
                 pipeline_cfg: dict[str, Any], csv_cfg: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Ejecuta el pipeline completo y devuelve el informe (dict serializable). Escribe figuras y PDF en workdir."""
    rep = settings.report
    set_seeds(int(pipeline_cfg["project"]["seed"]))
    warnings: list[str] = []
    log: dict[str, Any] = {}
    signal: Optional[dict[str, Any]] = None
    if kind == "csv_features":
        fs, X = features_from_csv(pd.read_csv(input_path), csv_cfg or pipeline_cfg)
        warnings.append("Entrada de características precalculadas: no hay señal para el visor ni control de artefactos propio.")
    else:
        raw = read_raw_file(input_path, kind, pipeline_cfg, sampling_rate_hz or settings.input.raw.default_sampling_rate_hz)
        fs, log, raw_clean = raw_to_feature_set(raw, pipeline_cfg, settings.input.raw.model_dump(), int(pipeline_cfg["project"]["seed"]),
                                                subject_id=input_path.stem, return_clean=True)
        X, _ = build_feature_table(fs, pipeline_cfg)
        signal = _signal_excerpt(raw_clean, settings)
        if log.get("n_epochs_rejected", 0) > 0.3 * max(1, log.get("n_epochs_in", 1)):
            warnings.append(f"Se rechazó el {100 * log['n_epochs_rejected'] / log['n_epochs_in']:.0f} % de las épocas por artefactos; interpretar con cautela.")
        warnings.append(settings.input.raw.domain_shift_warning)
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
        tier = rp.assign_tier(r["p_posterior"], rep)
        findings = rp.findings_table(x_row, r["z"], bundle, rep)
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
        if signal is not None and i == 0:
            figs[f"signal_{rid}"] = rp.fig_signal(signal, settings.signal_viewer, rep, figdir / f"signal_{rid}.png")
        results.append({"record_id": rid, "p_calibrated": r["p_calibrated"], "likelihood_ratio": r["likelihood_ratio"],
                        "pretest_prevalence": r["pretest_prevalence"], "p_posterior": r["p_posterior"], "above_threshold": r["above_threshold"],
                        "tier": tier, "preliminary_assessment": _assessment(r, tier, findings, rep),
                        "findings": findings, "contributions": contribs, "band_relative_power": patient_bands, "band_reference": ref_bands,
                        "topography_z": topo, "coherence_matrix": coh_json, "signal": signal if i == 0 else None})

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

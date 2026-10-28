"""Señal cruda (EDF/BDF/CSV) -> EEGFeatureSet con potencia por banda y coherencia por par.

Reproduce, con el pipeline de este repositorio, la misma estructura de características que el CSV
de entrenamiento. ADVERTENCIA: los parámetros exactos del extractor original (frecuencia de muestreo,
referencia, duración, ventana espectral) son desconocidos; existe riesgo de desplazamiento de dominio.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
from scipy.signal import coherence as scipy_coherence

from .config import renamed_channels
from .io import EEGFeatureSet
from .preprocess import band_power_from_epochs, filter_raw, make_epochs, raw_from_dataframe, rereference, run_ica


def read_raw_file(path: Path, kind: str, cfg: dict[str, Any], sampling_rate_hz: float | None) -> mne.io.BaseRaw:
    """Carga EDF/BDF o CSV crudo y lo deja con los 19 canales del config renombrados al 10-20 moderno."""
    ch_cfg = cfg["channels"]
    if kind == "edf":
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False) if path.suffix.lower() == ".edf" \
            else mne.io.read_raw_bdf(path, preload=True, verbose=False)
        # normaliza nombres: quita prefijos tipo "EEG Fp1-REF" y aplica rename del config
        mapping = {}
        wanted = {c.upper(): c for c in ch_cfg["list"]} | {ch_cfg["rename"].get(c, c).upper(): c for c in ch_cfg["list"]}
        for name in raw.ch_names:
            token = name.replace("EEG", "").replace("-REF", "").replace("-LE", "").strip().split("-")[0].strip().upper()
            if token in wanted:
                mapping[name] = ch_cfg["rename"].get(wanted[token], wanted[token])
        raw.rename_channels(mapping)
        keep = renamed_channels(cfg)
        missing = [c for c in keep if c not in raw.ch_names]
        if missing:
            raise ValueError(f"Canales ausentes en el archivo: {missing}")
        raw.pick(keep)
        raw.set_montage(mne.channels.make_standard_montage(ch_cfg["montage"]), on_missing="ignore")
        return raw
    if kind == "csv_raw":
        if sampling_rate_hz is None:
            raise ValueError("csv_raw requiere sampling_rate_hz")
        df = pd.read_csv(path)
        cols = [c for c in ch_cfg["list"] if c in df.columns]
        if len(cols) != len(ch_cfg["list"]):
            raise ValueError(f"Faltan columnas de canal: {set(ch_cfg['list']) - set(cols)}")
        return raw_from_dataframe(df, ch_cfg["list"], ch_cfg["rename"], float(sampling_rate_hz), ch_cfg["montage"])
    raise ValueError(f"Tipo de entrada cruda no soportado: {kind}")


def coherence_from_epochs(epochs: mne.Epochs, bands: dict[str, list[float]], nperseg_s: float) -> np.ndarray:
    """Coherencia (magnitud cuadrada, en %) promedio por banda y par -> (n_bands, n_ch, n_ch)."""
    data = epochs.get_data(copy=False)                      # (n_ep, n_ch, n_t)
    sf = float(epochs.info["sfreq"])
    x = np.concatenate(list(data), axis=-1)                 # concatena épocas limpias
    n_ch = x.shape[0]
    nperseg = int(nperseg_s * sf)
    out = np.full((len(bands), n_ch, n_ch), np.nan)
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            f, cxy = scipy_coherence(x[i], x[j], fs=sf, nperseg=nperseg)
            for bi, (lo, hi) in enumerate(bands.values()):
                m = (f >= lo) & (f < hi)
                out[bi, i, j] = out[bi, j, i] = 100.0 * float(cxy[m].mean())
    return out


def raw_to_feature_set(raw: mne.io.BaseRaw, cfg: dict[str, Any], app_raw_cfg: dict[str, Any], seed: int,
                       subject_id: str = "upload") -> tuple[EEGFeatureSet, dict[str, Any]]:
    """Pipeline MNE completo (filtros, notch, referencia, ICA, épocas, rechazo) y extracción de características."""
    pre = cfg["preprocessing"]
    log: dict[str, Any] = {"sfreq": float(raw.info["sfreq"]), "duration_s": float(raw.times[-1])}
    if log["duration_s"] < float(app_raw_cfg["min_duration_s"]):
        raise ValueError(f"Registro demasiado corto ({log['duration_s']:.1f}s < {app_raw_cfg['min_duration_s']}s)")
    raw = filter_raw(raw, pre["bandpass_hz"], pre["notch_hz"])
    raw = rereference(raw, pre["rereference"])
    if pre["ica"]["enabled"]:
        raw, ica_log = run_ica(raw, pre["ica"], seed)
        log["ica_excluded"] = ica_log
    epochs, ep_log = make_epochs(raw, pre["epoch"]["length_s"], pre["epoch"]["overlap_fraction"], pre["reject"])
    log.update(ep_log)
    if len(epochs) == 0:
        raise ValueError("Todas las épocas fueron rechazadas por artefactos")
    bands = cfg["bands"]
    power = band_power_from_epochs(epochs, bands) * 1e12          # V^2 -> uV^2
    coh = coherence_from_epochs(epochs, bands, float(app_raw_cfg["coherence_nperseg_s"]))
    fs = EEGFeatureSet(
        subjects=np.array([subject_id]), groups=np.array([0]), y=np.array([-1]), labels_raw=np.array(["unknown"]),
        bands=list(bands.keys()), channels=list(epochs.ch_names), power=power[None], coherence=coh[None],
    )
    return fs, log

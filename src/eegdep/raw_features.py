"""Señal cruda (EDF/BDF/EEGLAB/FIF/CSV) -> EEGFeatureSet con potencia por banda y coherencia por par.

Mismo preprocesamiento que el entrenamiento con señal cruda (`config/config_ds003478.yaml`): remuestreo,
notch, pasabanda, referencia (sobre TODOS los canales EEG disponibles, como en el entrenamiento), ICA,
selección de los 19 canales 10-20, épocas con rechazo, potencia (Welch) y coherencia.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
from scipy.signal import coherence as scipy_coherence

from .io import EEGFeatureSet
from .preprocess import band_power_from_epochs, make_epochs, raw_from_dataframe, run_ica

mne.set_log_level("ERROR")


def _canonical_map(montage_name: str) -> dict[str, str]:
    return {n.upper(): n for n in mne.channels.make_standard_montage(montage_name).ch_names}


def _normalize_token(name: str) -> str:
    """'EEG Fp1-REF' -> 'FP1'; 'T3' -> 'T3' (el rename del config lo lleva a T7)."""
    return name.replace("EEG", "").replace("-REF", "").replace("-LE", "").replace("-A1", "").replace("-A2", "").strip().split("-")[0].strip().upper()


def read_raw_file(path: Path, kind: str, cfg: dict[str, Any], sampling_rate_hz: float | None) -> mne.io.BaseRaw:
    """Carga el archivo, nombra los canales según el montaje 10-20 moderno y tipa EOG/mastoides/otros como no-EEG."""
    ch_cfg = cfg["channels"]
    ext = path.suffix.lower()
    if kind == "csv_raw":
        if sampling_rate_hz is None:
            raise ValueError("csv_raw requiere sampling_rate_hz")
        df = pd.read_csv(path)
        cols = [c for c in ch_cfg["list"] if c in df.columns]
        if len(cols) != len(ch_cfg["list"]):
            raise ValueError(f"Faltan columnas de canal: {set(ch_cfg['list']) - set(cols)}")
        return raw_from_dataframe(df, ch_cfg["list"], ch_cfg["rename"], float(sampling_rate_hz), ch_cfg["montage"])
    readers = {".edf": mne.io.read_raw_edf, ".bdf": mne.io.read_raw_bdf, ".set": mne.io.read_raw_eeglab, ".fif": mne.io.read_raw_fif}
    if ext not in readers:
        raise ValueError(f"Extensión no soportada: {ext}")
    raw = readers[ext](path, preload=True, verbose=False)
    canonical = _canonical_map(ch_cfg["montage"])
    alias = {k.upper(): v for k, v in ch_cfg["rename"].items()}          # p.ej. T3 -> T7
    non_eeg = {c.upper() for key in ("eog", "mastoids", "drop") for c in ch_cfg.get(key, [])}
    rename, types = {}, {}
    for name in raw.ch_names:
        tok = _normalize_token(name)
        tok = alias.get(tok, tok).upper()
        if tok in non_eeg:
            types[name] = "eog" if tok in {c.upper() for c in ch_cfg.get("eog", [])} else "misc"
        elif tok in canonical:
            if canonical[tok] != name:
                rename[name] = canonical[tok]
        else:
            types[name] = "misc"
    raw.rename_channels(rename)
    if types:
        raw.set_channel_types(types)
    raw.set_montage(mne.channels.make_standard_montage(ch_cfg["montage"]), on_missing="ignore")
    keep = required_channels(cfg)
    missing = [c for c in keep if c not in raw.ch_names]
    if missing:
        raise ValueError(f"Canales 10-20 ausentes en el archivo: {missing}")
    return raw


def required_channels(cfg: dict[str, Any]) -> list[str]:
    ch_cfg = cfg["channels"]
    canonical = _canonical_map(ch_cfg["montage"])
    return [canonical.get(ch_cfg["rename"].get(c, c).upper(), ch_cfg["rename"].get(c, c)) for c in ch_cfg["list"]]


def coherence_from_epochs(epochs: mne.Epochs, bands: dict[str, list[float]], nperseg_s: float) -> np.ndarray:
    """Coherencia (magnitud cuadrada, en %) promedio por banda y par -> (n_bands, n_ch, n_ch)."""
    data = epochs.get_data(copy=False)
    sf = float(epochs.info["sfreq"])
    x = np.concatenate(list(data), axis=-1)
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


def preprocess_raw(raw: mne.io.BaseRaw, cfg: dict[str, Any], seed: int) -> tuple[mne.io.BaseRaw, dict[str, Any]]:
    """Remuestreo, notch, pasabanda, referencia sobre todos los EEG, ICA y selección de los 19 canales."""
    pre, ch_cfg = cfg["preprocessing"], cfg["channels"]
    log: dict[str, Any] = {"sfreq_native": float(raw.info["sfreq"]), "duration_s": float(raw.times[-1]),
                           "n_eeg_channels_input": len(mne.pick_types(raw.info, eeg=True))}
    if pre.get("resample_hz"):
        raw.resample(float(pre["resample_hz"]), verbose=False)
    raw.notch_filter(freqs=pre["notch_hz"], verbose=False)
    raw.filter(l_freq=pre["bandpass_hz"][0], h_freq=pre["bandpass_hz"][1], verbose=False)
    mode = pre["rereference"]
    if mode == "average":
        raw.set_eeg_reference("average", projection=False, verbose=False)
    elif mode == "linked_mastoids":
        mast = [m for m in ch_cfg.get("mastoids", []) if m in raw.ch_names]
        if len(mast) != 2:
            raise ValueError("linked_mastoids requiere ambos mastoides en el archivo")
        raw.set_channel_types({m: "eeg" for m in mast}); raw.set_eeg_reference(mast, projection=False, verbose=False)
        raw.set_channel_types({m: "misc" for m in mast})
    elif mode != "none":
        raise ValueError(f"rereference desconocido: {mode}")
    if pre["ica"]["enabled"]:
        n_eeg = len(mne.pick_types(raw.info, eeg=True))
        ica_cfg = {**pre["ica"], "n_components": min(int(pre["ica"]["n_components"]), max(1, n_eeg - 1))}
        raw, ica_log = run_ica(raw, ica_cfg, seed)
        log["ica_excluded"] = ica_log
    raw.pick(required_channels(cfg))
    return raw, log


def raw_to_feature_set(raw: mne.io.BaseRaw, cfg: dict[str, Any], app_raw_cfg: dict[str, Any], seed: int,
                       subject_id: str = "upload", return_clean: bool = False):
    """Pipeline MNE completo y extracción de características por registro."""
    pre = cfg["preprocessing"]
    if float(raw.times[-1]) < float(app_raw_cfg["min_duration_s"]):
        raise ValueError(f"Registro demasiado corto ({raw.times[-1]:.1f}s < {app_raw_cfg['min_duration_s']}s)")
    raw, log = preprocess_raw(raw, cfg, seed)
    epochs, ep_log = make_epochs(raw, pre["epoch"]["length_s"], pre["epoch"]["overlap_fraction"], pre["reject"])
    log.update(ep_log)
    if len(epochs) == 0:
        raise ValueError("Todas las épocas fueron rechazadas por artefactos")
    bands = cfg["bands"]
    power = band_power_from_epochs(epochs, bands) * 1e12
    coh = coherence_from_epochs(epochs, bands, float(app_raw_cfg["coherence_nperseg_s"]))
    fs = EEGFeatureSet(
        subjects=np.array([subject_id]), groups=np.array([0]), y=np.array([-1]), labels_raw=np.array(["unknown"]),
        bands=list(bands.keys()), channels=list(epochs.ch_names), power=power[None], coherence=coh[None],
    )
    if return_clean:
        return fs, log, raw
    return fs, log

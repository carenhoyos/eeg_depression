"""Preprocesamiento de SEÑAL CRUDA con MNE. Solo se usa cuando data.kind == raw.

Con el CSV actual (características ya extraídas) este módulo no se ejecuta: filtrado,
notch, rereferenciación, ICA y segmentación ya ocurrieron aguas arriba, fuera de
nuestro control. Se incluye para que el pipeline sea el mismo cuando exista señal cruda.
"""
from __future__ import annotations

from typing import Any

import mne
import numpy as np
import pandas as pd


def raw_from_dataframe(df: pd.DataFrame, channels: list[str], rename: dict[str, str], sfreq: float, montage: str) -> mne.io.RawArray:
    """Construye un Raw de MNE a partir de columnas (muestras x canales) en microvoltios."""
    data = df[channels].to_numpy(float).T * 1e-6  # uV -> V
    info = mne.create_info([rename.get(c, c) for c in channels], sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage(mne.channels.make_standard_montage(montage))
    return raw


def filter_raw(raw: mne.io.BaseRaw, bandpass_hz: list[float], notch_hz: list[float]) -> mne.io.BaseRaw:
    raw = raw.copy().notch_filter(freqs=notch_hz, verbose=False)
    return raw.filter(l_freq=bandpass_hz[0], h_freq=bandpass_hz[1], verbose=False)


def rereference(raw: mne.io.BaseRaw, mode: str) -> mne.io.BaseRaw:
    if mode == "average":
        return raw.copy().set_eeg_reference("average", projection=False, verbose=False)
    if mode == "none":
        return raw
    raise ValueError(f"rereference no soportado con 19 canales sin mastoides: {mode}")


def run_ica(raw: mne.io.BaseRaw, ica_cfg: dict[str, Any], seed: int) -> tuple[mne.io.BaseRaw, dict[str, list[int]]]:
    """ICA; componentes oculares por correlación con canales frontopolares; musculares por find_bads_muscle."""
    ica = mne.preprocessing.ICA(n_components=ica_cfg["n_components"], method=ica_cfg["method"], random_state=seed, verbose=False)
    ica.fit(raw, verbose=False)
    eog_idx: list[int] = []
    for ch in ica_cfg["eog_proxy_channels"]:
        idx, _ = ica.find_bads_eog(raw, ch_name=ch, verbose=False)
        eog_idx += idx
    mus_idx, mus_scores = ica.find_bads_muscle(raw, threshold=ica_cfg["muscle_score_threshold"], verbose=False)
    # Prioridad: oculares primero; musculares ordenados por puntuación; tope configurable de componentes excluidos.
    mus_sorted = sorted(mus_idx, key=lambda i: -float(mus_scores[i]))
    excl = list(dict.fromkeys(sorted(set(eog_idx)) + mus_sorted))[: int(ica_cfg.get("max_excluded", len(ica_cfg) and 99))]
    ica.exclude = excl
    return ica.apply(raw.copy(), verbose=False), {"eog": sorted(set(eog_idx)), "muscle": mus_idx, "excluded": excl}


def make_epochs(raw: mne.io.BaseRaw, length_s: float, overlap_fraction: float, reject_cfg: dict[str, float]) -> tuple[mne.Epochs, dict[str, int]]:
    """Ventanas de duración fija con solapamiento y rechazo por amplitud pico a pico / canal plano."""
    epochs = mne.make_fixed_length_epochs(raw, duration=length_s, overlap=length_s * overlap_fraction, preload=True, verbose=False)
    n0 = len(epochs)
    epochs.drop_bad(reject=dict(eeg=reject_cfg["peak_to_peak_uv"] * 1e-6), flat=dict(eeg=reject_cfg["flat_uv"] * 1e-6), verbose=False)
    return epochs, {"n_epochs_in": n0, "n_epochs_kept": len(epochs), "n_epochs_rejected": n0 - len(epochs)}


def band_power_from_epochs(epochs: mne.Epochs, bands: dict[str, list[float]]) -> np.ndarray:
    """Potencia absoluta media por (banda, canal) vía Welch -> (n_bands, n_ch)."""
    spec = epochs.compute_psd(method="welch", fmin=min(b[0] for b in bands.values()), fmax=max(b[1] for b in bands.values()), verbose=False)
    psd, freqs = spec.get_data(return_freqs=True)  # (n_epochs, n_ch, n_freq)
    out = []
    for lo, hi in bands.values():
        m = (freqs >= lo) & (freqs < hi)
        out.append(np.trapz(psd[:, :, m], freqs[m], axis=-1).mean(axis=0))
    return np.stack(out)

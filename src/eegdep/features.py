"""Construcción de la tabla de características por sujeto a partir de EEGFeatureSet.

Cada característica lleva un nombre `tipo__banda__localización` y una fila en
`feature_meta` para poder mapearla a topografía e interpretarla después.
"""
from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Any

import numpy as np
import pandas as pd

from .io import EEGFeatureSet


def relative_power(power: np.ndarray) -> np.ndarray:
    """Potencia relativa: P(banda, canal) / sum_bandas P(., canal)."""
    total = power.sum(axis=1, keepdims=True)
    return power / total


def regional_mean(arr: np.ndarray, channels: list[str], regions: dict[str, list[str]]) -> tuple[np.ndarray, list[str]]:
    """Promedia la última dimensión (canal) dentro de cada región."""
    names = list(regions.keys())
    out = np.stack([arr[..., [channels.index(c) for c in regions[r]]].mean(axis=-1) for r in names], axis=-1)
    return out, names


def asymmetry(power: np.ndarray, channels: list[str], pairs: list[list[str]]) -> tuple[np.ndarray, list[str]]:
    """ln(P_derecho) - ln(P_izquierdo) sobre potencia absoluta, por banda y par."""
    cols, names = [], []
    for left, right in pairs:
        li, ri = channels.index(left), channels.index(right)
        cols.append(np.log(power[:, :, ri]) - np.log(power[:, :, li]))
        names.append(f"{right}-{left}")
    return np.stack(cols, axis=-1), names


def coherence_regional(coh: np.ndarray, channels: list[str], regions: dict[str, list[str]]) -> tuple[np.ndarray, list[str]]:
    """Coherencia media entre (y dentro de) regiones. Excluye la diagonal (nan)."""
    names, cols = [], []
    for ra, rb in combinations_with_replacement(regions.keys(), 2):
        ia = [channels.index(c) for c in regions[ra]]
        ib = [channels.index(c) for c in regions[rb]]
        block = coh[:, :, ia][:, :, :, ib]  # (n, nb, |ra|, |rb|)
        cols.append(np.nanmean(block.reshape(block.shape[0], block.shape[1], -1), axis=-1))
        names.append(f"{ra}-{rb}")
    return np.stack(cols, axis=-1), names


def build_feature_table(fs: EEGFeatureSet, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (X: DataFrame n x p, feature_meta: DataFrame p x [kind, band, location])."""
    f = cfg["features"]
    regions = cfg["channels"]["regions"]
    bands, ch = fs.bands, fs.channels
    blocks: list[tuple[str, str, str, np.ndarray]] = []  # (kind, band, location, values)

    def add(kind: str, arr: np.ndarray, locs: list[str], band_list: list[str]):
        for bi, b in enumerate(band_list):
            for li, loc in enumerate(locs):
                blocks.append((kind, b, loc, arr[:, bi, li]))

    if f["absolute_power"]["enabled"]:
        pw = np.log10(fs.power) if f["absolute_power"]["log10"] else fs.power
        add("abspow", pw, ch, bands)
    rel = relative_power(fs.power)
    if f["relative_power"]["enabled"]:
        add("relpow", rel, ch, bands)
    if f["regional_relative_power"]["enabled"]:
        reg, rnames = regional_mean(rel, ch, regions)
        add("regpow", reg, rnames, bands)
    if f["asymmetry"]["enabled"]:
        asy, pnames = asymmetry(fs.power, ch, cfg["channels"]["asymmetry_pairs"])
        sel = [bands.index(b) for b in f["asymmetry"]["bands"]]
        add("asym", asy[:, sel, :], pnames, [bands[i] for i in sel])
    if f["coherence"]["enabled"] and fs.coherence is not None and f["coherence"]["mode"] != "none":
        sel = [bands.index(b) for b in f["coherence"]["bands"]]
        if f["coherence"]["mode"] == "regional":
            cr, cnames = coherence_regional(fs.coherence[:, sel], ch, regions)
            add("coh", cr, cnames, [bands[i] for i in sel])
        elif f["coherence"]["mode"] == "all":
            iu = np.triu_indices(len(ch), k=1)
            arr = fs.coherence[:, sel][:, :, iu[0], iu[1]]
            add("coh", arr, [f"{ch[a]}-{ch[b]}" for a, b in zip(*iu)], [bands[i] for i in sel])
        else:
            raise ValueError(f"features.coherence.mode desconocido: {f['coherence']['mode']}")
    if f["nonlinear"]["enabled"] or f["wpli"]["enabled"]:
        raise NotImplementedError("Medidas no lineales y wPLI requieren señal cruda (data.kind == raw).")

    names = [f"{k}__{b}__{loc}" for k, b, loc, _ in blocks]
    X = pd.DataFrame(np.column_stack([v for *_, v in blocks]), columns=names, index=fs.subjects)
    meta = pd.DataFrame([(k, b, loc) for k, b, loc, _ in blocks], columns=["kind", "band", "location"], index=names)
    return X, meta

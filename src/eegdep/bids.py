"""Lector BIDS-EEG (EEGLAB .set) -> EEGFeatureSet por sujeto + épocas limpias para deep learning.

Pasos por sujeto (todos parametrizados en el YAML):
  participants.tsv -> etiqueta derivada (BDI alto vs bajo, zona intermedia y exclusiones)
  .set/.fdt + events.tsv -> remuestreo, notch, pasabanda, montaje, referencia, ICA (cacheado)
  bloques de la condición elegida (ojos cerrados/abiertos) -> épocas con rechazo -> potencia y coherencia
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .io import EEGFeatureSet
from .preprocess import band_power_from_epochs, make_epochs, run_ica
from .raw_features import coherence_from_epochs

mne.set_log_level("ERROR")


# ------------------------------------------------------------------ participantes y etiquetas
def load_participants(bids_root: Path, data_cfg: dict[str, Any]) -> pd.DataFrame:
    df = pd.read_csv(bids_root / data_cfg["participants_file"], sep="\t", na_values=["n/a", "NaN", ""])
    return df.set_index(data_cfg["subject_column"])


def derive_labels(df: pd.DataFrame, data_cfg: dict[str, Any]) -> pd.DataFrame:
    """Devuelve DataFrame indexado por sujeto con y (0/1/-1 excluido), label_raw y motivo."""
    lc = data_cfg["labels"]
    score = pd.to_numeric(df[data_cfg["label_column"]], errors="coerce")
    y = pd.Series(-1, index=df.index, dtype=int)
    reason = pd.Series("", index=df.index, dtype=object)
    pos = score >= float(lc["positive_min"])
    neg = score <= float(lc["negative_max"]) if lc["negative_max_inclusive"] else score < float(lc["negative_max"])
    y[pos] = 1
    y[neg] = 0
    reason[score.isna()] = "sin puntaje"
    reason[~pos & ~neg & score.notna()] = "zona intermedia"
    for s in lc.get("exclude_subjects", []):
        if s in y.index:
            y[s], reason[s] = -1, "excluido por config (subject)"
    if "Original_ID" in df.columns:
        for oid in lc.get("exclude_original_ids", []):
            m = df["Original_ID"] == oid
            y[m], reason[m] = -1, "excluido por config (original_id)"
    if "SCID" in df.columns and lc.get("exclude_scid"):
        target = lc.get("exclude_scid_applies_to", "negative")
        m = df["SCID"].isin(lc["exclude_scid"])
        if target in ("negative", "both"):
            mm = m & (y == 0); y[mm], reason[mm] = -1, "SCID excluyente"
        if target in ("positive", "both"):
            mm = m & (y == 1); y[mm], reason[mm] = -1, "SCID excluyente"
    label_raw = y.map({1: data_cfg["positive_class"], 0: data_cfg["negative_class"], -1: "excluded"})
    return pd.DataFrame({"y": y, "label_raw": label_raw, "score": score, "exclusion_reason": reason})


# ------------------------------------------------------------------ archivos y eventos
def find_runs(bids_root: Path, subject: str, task: str, runs: list[str]) -> list[Path]:
    out = []
    for r in runs:
        p = bids_root / subject / "eeg" / f"{subject}_task-{task}_run-{r}_eeg.set"
        if p.exists():
            out.append(p)
    return out


def condition_spans(set_path: Path, cond_cfg: dict[str, Any], which: str) -> list[tuple[float, float]]:
    """Bloques [t0, t1] de la condición a partir de events.tsv (marcadores periódicos agrupados por cercanía)."""
    ev = pd.read_csv(set_path.with_name(set_path.name.replace("_eeg.set", "_events.tsv")), sep="\t")
    pat = re.compile(cond_cfg["patterns"][which])
    onsets = np.sort(ev.loc[ev[cond_cfg["event_column"]].astype(str).str.match(pat), "onset"].to_numpy(float))
    if len(onsets) == 0:
        return []
    gaps = np.diff(onsets)
    cuts = np.where(gaps > float(cond_cfg["block_gap_s"]))[0]
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts, len(onsets) - 1]
    spacing = float(np.median(gaps[gaps <= cond_cfg["block_gap_s"]])) if (gaps <= cond_cfg["block_gap_s"]).any() else 0.0
    spans = [(float(onsets[s]), float(onsets[e]) + spacing) for s, e in zip(starts, ends)]
    return [sp for sp in spans if sp[1] - sp[0] >= float(cond_cfg["min_block_s"])]


# ------------------------------------------------------------------ preprocesamiento por sujeto
def _apply_montage(raw: mne.io.BaseRaw, ch_cfg: dict[str, Any]) -> mne.io.BaseRaw:
    montage = mne.channels.make_standard_montage(ch_cfg["montage"])
    canonical = {n.upper(): n for n in montage.ch_names}
    rename, types = {}, {}
    for name in raw.ch_names:
        u = name.upper()
        if u in {c.upper() for c in ch_cfg.get("eog", [])}:
            types[name] = "eog"
        elif u in {c.upper() for c in ch_cfg.get("mastoids", [])}:
            types[name] = "misc"
        elif u in {c.upper() for c in ch_cfg.get("drop", [])}:
            types[name] = "misc"
        elif u in canonical:
            rename[name] = canonical[u]
        else:
            types[name] = "misc"
    raw.rename_channels(rename)
    if types:
        raw.set_channel_types(types)
    raw.set_montage(montage, on_missing="ignore")
    return raw


def clean_raw(set_path: Path, cfg: dict[str, Any], seed: int) -> tuple[mne.io.BaseRaw, dict[str, Any]]:
    """Carga + remuestreo + notch + pasabanda + montaje + referencia + ICA. Cacheado en .fif."""
    pre, ch_cfg = cfg["preprocessing"], cfg["channels"]
    cache_dir = Path(cfg["_root"]) / pre["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / (set_path.stem + "_clean_raw.fif")
    log_path = cache.with_suffix(".json")
    if cache.exists() and log_path.exists():
        return mne.io.read_raw_fif(cache, preload=True, verbose=False), json.loads(log_path.read_text())
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    log: dict[str, Any] = {"sfreq_native": float(raw.info["sfreq"]), "duration_s": float(raw.times[-1]), "n_channels_native": len(raw.ch_names)}
    raw = _apply_montage(raw, ch_cfg)
    if pre.get("resample_hz"):
        raw.resample(float(pre["resample_hz"]), verbose=False)
    raw.notch_filter(freqs=pre["notch_hz"], verbose=False)
    raw.filter(l_freq=pre["bandpass_hz"][0], h_freq=pre["bandpass_hz"][1], verbose=False)
    mode = pre["rereference"]
    if mode == "average":
        raw.set_eeg_reference("average", projection=False, verbose=False)
    elif mode == "linked_mastoids":
        raw.set_channel_types({m: "eeg" for m in ch_cfg["mastoids"] if m in raw.ch_names})
        raw.set_eeg_reference(ch_cfg["mastoids"], projection=False, verbose=False)
        raw.set_channel_types({m: "misc" for m in ch_cfg["mastoids"] if m in raw.ch_names})
    elif mode != "none":
        raise ValueError(f"rereference desconocido: {mode}")
    if pre["ica"]["enabled"]:
        raw, ica_log = run_ica(raw, pre["ica"], seed)
        log["ica_excluded"] = ica_log
        log["ica_n_excluded"] = len(ica_log["excluded"])
    canonical = {n.upper(): n for n in mne.channels.make_standard_montage(ch_cfg["montage"]).ch_names}
    keep = [canonical.get(ch_cfg["rename"].get(c, c).upper(), ch_cfg["rename"].get(c, c)) for c in ch_cfg["list"]]
    missing = [c for c in keep if c not in raw.ch_names]
    if missing:
        raise ValueError(f"{set_path.name}: canales ausentes {missing}")
    raw.pick(keep)
    raw.save(cache, overwrite=True, verbose=False)
    log_path.write_text(json.dumps(log))
    return raw, log


def subject_record(subject: str, set_paths: list[Path], cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Procesa todas las corridas de un sujeto y devuelve potencia, coherencia, épocas y log."""
    pre, dc = cfg["preprocessing"], cfg["data"]
    bands = cfg["bands"]
    which = dc["condition"]["use"]
    pieces, log = [], {"subject": subject, "runs": [p.name for p in set_paths], "spans": []}
    for p in set_paths:
        raw, rlog = clean_raw(p, cfg, seed)
        log.setdefault("clean", []).append(rlog)
        conds = ["eyes_closed", "eyes_open"] if which == "both" else [which]
        for c in conds:
            for t0, t1 in condition_spans(p, dc["condition"], c):
                t1 = min(t1, float(raw.times[-1]))
                if t1 - t0 >= float(dc["condition"]["min_block_s"]):
                    pieces.append(raw.copy().crop(tmin=t0, tmax=t1))
                    log["spans"].append([c, round(t0, 2), round(t1, 2)])
    if not pieces:
        return {**log, "error": "sin bloques de la condición"}
    raw_c = mne.concatenate_raws(pieces, verbose=False)
    epochs, ep_log = make_epochs(raw_c, pre["epoch"]["length_s"], pre["epoch"]["overlap_fraction"], pre["reject"])
    log.update(ep_log)
    if len(epochs) < int(pre["min_epochs_per_subject"]):
        return {**log, "error": f"solo {len(epochs)} épocas limpias"}
    power = band_power_from_epochs(epochs, bands) * 1e12                      # V^2 -> uV^2
    coh = coherence_from_epochs(epochs, bands, float(cfg["features"]["coherence"]["nperseg_s"]))
    deep = cfg.get("deep", {})
    ep_deep = epochs.copy()
    if deep.get("resample_hz"):
        ep_deep.resample(float(deep["resample_hz"]), verbose=False)
    X_ep = (ep_deep.get_data(copy=False) * 1e6).astype(np.float32)             # (n_ep, n_ch, n_t) en uV
    return {**log, "power": power, "coherence": coh, "epochs": X_ep, "channels": list(epochs.ch_names),
            "sfreq_epochs": float(ep_deep.info["sfreq"])}


# ------------------------------------------------------------------ dataset completo
def build_raw_feature_set(cfg: dict[str, Any]) -> tuple[EEGFeatureSet, dict[str, np.ndarray], pd.DataFrame]:
    dc = cfg["data"]
    bids_root = (Path(cfg["_root"]) / dc["bids_root"]).resolve()
    parts = load_participants(bids_root, dc)
    labels = derive_labels(parts, dc)
    seed = int(cfg["project"]["seed"])
    jobs = []
    for subject in labels.index:
        if labels.loc[subject, "y"] < 0:
            continue
        runs = find_runs(bids_root, subject, dc["task"], dc["runs"])
        if runs:
            jobs.append((subject, runs))
        else:
            labels.loc[subject, "y"], labels.loc[subject, "exclusion_reason"] = -1, "sin archivos EEG"
    n_jobs = int(cfg["project"]["n_jobs"])
    results = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(subject_record)(s, r, cfg, seed) for s, r in jobs)
    rows, epochs_by_subject, power, coh, subjects = [], {}, [], [], []
    for res in results:
        s = res["subject"]
        if "error" in res:
            labels.loc[s, "y"], labels.loc[s, "exclusion_reason"] = -1, res["error"]
            rows.append({"subject": s, "ok": False, "reason": res["error"], **{k: res.get(k) for k in ("n_epochs_in", "n_epochs_kept")}})
            continue
        subjects.append(s); power.append(res["power"]); coh.append(res["coherence"]); epochs_by_subject[s] = res["epochs"]
        rows.append({"subject": s, "ok": True, "n_epochs_in": res["n_epochs_in"], "n_epochs_kept": res["n_epochs_kept"],
                     "n_spans": len(res["spans"]), "ica_n_excluded": res["clean"][0].get("ica_n_excluded"), "sfreq_epochs": res["sfreq_epochs"]})
    channels = results[[i for i, r in enumerate(results) if "error" not in r][0]]["channels"]
    y = labels.loc[subjects, "y"].to_numpy(int)
    meta = parts.loc[subjects, [c for c in dc.get("covariates", []) if c in parts.columns]].reset_index(drop=True)
    meta["score"] = labels.loc[subjects, "score"].to_numpy()
    fs = EEGFeatureSet(subjects=np.array(subjects), groups=pd.factorize(subjects)[0], y=y,
                       labels_raw=labels.loc[subjects, "label_raw"].to_numpy(), bands=list(cfg["bands"].keys()), channels=channels,
                       power=np.stack(power), coherence=np.stack(coh), meta=meta)
    processing = pd.DataFrame(rows).merge(labels.reset_index().rename(columns={dc["subject_column"]: "subject"}), on="subject", how="right")
    return fs, epochs_by_subject, processing

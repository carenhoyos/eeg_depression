"""Estandarización a (sujeto, banda, canal) + control de calidad + registro de descartes."""
import json
import pickle

from _bootstrap import get_config
from eegdep.config import output_dir, resolve_path
from eegdep.io import build_feature_set, load_csv, to_long
from eegdep.qc import apply_qc

cfg = get_config()
out = output_dir(cfg)
if cfg["data"]["kind"] == "raw":
    # Señal cruda BIDS -> preprocesamiento MNE -> características por sujeto + épocas para la CNN
    import numpy as np
    from eegdep.bids import build_raw_feature_set
    fs, epochs_by_subject, processing = build_raw_feature_set(cfg)
    processing.to_csv(out / "02_processing_log.csv", index=False)
    np.savez_compressed(out / "02_epochs.npz", **{s: e for s, e in epochs_by_subject.items()})
    ok = processing[processing.ok == True]
    print(f"sujetos procesados: {len(ok)}  excluidos: {(processing.y < 0).sum()}  motivos: {processing.loc[processing.y < 0, 'exclusion_reason'].value_counts().to_dict()}")
    print(f"épocas por sujeto: media {ok.n_epochs_kept.mean():.1f}, min {ok.n_epochs_kept.min()}, rechazadas en total {(ok.n_epochs_in - ok.n_epochs_kept).sum()}")
else:
    df = load_csv(resolve_path(cfg, cfg["data"]["csv_path"]), cfg["data"]["drop_columns"])
    fs = build_feature_set(df, cfg)
print(f"power {fs.power.shape}  coherence {None if fs.coherence is None else fs.coherence.shape}")
print(f"bandas {fs.bands}\ncanales {fs.channels}")
fs, report = apply_qc(fs, cfg["qc"])
print(json.dumps(report, indent=2, default=str))
fs.qc.to_csv(out / "02_qc_table.csv", index=False)
to_long(fs).to_csv(out / "02_power_long.csv", index=False)
(out / "02_qc_report.json").write_text(json.dumps(report, indent=2, default=str))
with open(out / "02_feature_set.pkl", "wb") as f:
    pickle.dump(fs, f)

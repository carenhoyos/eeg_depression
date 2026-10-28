"""Estandarización a (sujeto, banda, canal) + control de calidad + registro de descartes."""
import json
import pickle

from _bootstrap import get_config
from eegdep.config import output_dir, resolve_path
from eegdep.io import build_feature_set, load_csv, to_long
from eegdep.qc import apply_qc

cfg = get_config()
df = load_csv(resolve_path(cfg, cfg["data"]["csv_path"]), cfg["data"]["drop_columns"])
fs = build_feature_set(df, cfg)
print(f"power {fs.power.shape}  coherence {None if fs.coherence is None else fs.coherence.shape}")
print(f"bandas {fs.bands}\ncanales {fs.channels}")
fs, report = apply_qc(fs, cfg["qc"])
print(json.dumps(report, indent=2, default=str))
out = output_dir(cfg)
fs.qc.to_csv(out / "02_qc_table.csv", index=False)
to_long(fs).to_csv(out / "02_power_long.csv", index=False)
(out / "02_qc_report.json").write_text(json.dumps(report, indent=2, default=str))
with open(out / "02_feature_set.pkl", "wb") as f:
    pickle.dump(fs, f)

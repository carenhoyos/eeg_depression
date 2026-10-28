"""Inspección del CSV guiada por la configuración (no asume estructura)."""
import json

from _bootstrap import get_config
from eegdep.config import output_dir, resolve_path
from eegdep.io import inspect_dataframe, load_csv

cfg = get_config()
df = load_csv(resolve_path(cfg, cfg["data"]["csv_path"]), drop_columns=[])
summary = inspect_dataframe(df, cfg["data"])
print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
(output_dir(cfg) / "01_inspection.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

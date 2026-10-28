"""Exporta el modelo elegido como bundle versionado para la aplicación (models/release/<version>/bundle.pkl).

Selección del modelo: por defecto el de mayor AUC medio en CV (criterio a priori, NO el holdout).
Uso: python scripts/06_export_bundle.py --config config/config.yaml --app-config app/config/app.yaml [--model logistic_l2] [--version 0.1.0]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eegdep.config import load_config, output_dir, resolve_path  # noqa: E402
from eegdep.inference import build_bundle  # noqa: E402

ap = argparse.ArgumentParser()
root = Path(__file__).resolve().parents[1]
ap.add_argument("--config", default=str(root / "config/config.yaml"))
ap.add_argument("--app-config", default=str(root / "app/config/app.yaml"))
ap.add_argument("--model", default=None, help="nombre del modelo en config.models; por defecto el mejor AUC en CV")
ap.add_argument("--version", default=None, help="por defecto app.version del app.yaml")
args = ap.parse_args()

cfg = load_config(args.config)
app_cfg = yaml.safe_load(Path(args.app_config).read_text(encoding="utf-8"))
out = output_dir(cfg)
results = json.loads((out / "04_results.json").read_text())
split = json.loads((out / "04_split.json").read_text())
name = args.model or max(results, key=lambda m: results[m]["cv_per_repeat_mean"]["auc"])
version = args.version or app_cfg["app"]["version"]
print(f"Modelo seleccionado: {name} (AUC CV={results[name]['cv_per_repeat_mean']['auc']:.3f})  versión {version}")

with open(out / f"04_{name}_final_model.pkl", "rb") as f:
    saved = pickle.load(f)
with open(out / "02_feature_set.pkl", "rb") as f:
    fs = pickle.load(f)
X = pd.read_csv(out / "03_features.csv", index_col=0)
meta = pd.read_csv(out / "03_feature_meta.csv", index_col=0)
dev = np.isin(fs.subjects, split["dev_subjects"])

validation = {k: results[name][k] for k in ("cv_per_repeat_mean", "cv_per_repeat_sd", "cv_pooled_ci95", "holdout", "holdout_ci95", "permutation") if k in results[name]}
validation["scheme"] = {"holdout_fraction": cfg["validation"]["holdout"]["fraction"], "outer": cfg["validation"]["outer"],
                        "inner": cfg["validation"]["inner"], "calibration": cfg["validation"]["calibration"]}
bundle = build_bundle(name, version, saved["model"], saved["uncalibrated"], saved["features"], meta, cfg,
                      X[dev], fs.y[dev], results[name].get("holdout_threshold", cfg["validation"]["threshold"]["value"]),
                      validation, list(app_cfg["report"]["reference_percentiles"]))
dest = root / "models" / "release" / version / "bundle.pkl"
bundle.save(dest)
latest = root / "models" / "release" / "latest"
if latest.is_symlink() or latest.exists():
    latest.unlink()
latest.symlink_to(version, target_is_directory=True)
(dest.parent / "model_card.json").write_text(json.dumps(bundle.card(), indent=2, default=str))
print(f"Bundle guardado en {dest} (latest -> {version})")

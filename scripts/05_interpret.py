"""Interpretabilidad: efectos univariados topográficos (solo dev), importancia de modelos, calibración."""
import json
import pickle

import numpy as np
import pandas as pd

from _bootstrap import get_config
from eegdep.config import output_dir
from eegdep.interpret import (band_channel_matrix, cohen_d, model_feature_importance, permutation_importance_series,
                              plot_calibration, plot_top_features, topomap_grid, univariate_auc)

cfg = get_config()
out = output_dir(cfg)
fig_dir = output_dir(cfg, "figures")
icfg, seed, n_jobs = cfg["interpretability"], int(cfg["project"]["seed"]), int(cfg["project"]["n_jobs"])
with open(out / "02_feature_set.pkl", "rb") as f:
    fs = pickle.load(f)
X = pd.read_csv(out / "03_features.csv", index_col=0)
meta = pd.read_csv(out / "03_feature_meta.csv", index_col=0)
split = json.loads((out / "04_split.json").read_text())
dev = np.isin(fs.subjects, split["dev_subjects"])
test = np.isin(fs.subjects, split["test_subjects"])

# 1) Efecto univariado por banda y canal, SOLO en desarrollo (el holdout no se mira)
uni = univariate_auc(X[dev], fs.y[dev]) if icfg["univariate_metric"] == "auc" else cohen_d(X[dev], fs.y[dev])
uni.to_csv(out / "05_univariate_effects_dev.csv")
center = 0.5 if icfg["univariate_metric"] == "auc" else 0.0
kind = "relpow" if (meta.kind == "relpow").any() else "abspow"
vals = band_channel_matrix(uni, meta, kind, icfg["topomap_bands"], fs.channels)
if vals:
    topomap_grid(vals, fs.channels, cfg["channels"]["montage"], fig_dir / f"topomap_univariate_{icfg['univariate_metric']}.png",
                 f"Efecto univariado ({icfg['univariate_metric']}) de potencia relativa, depresión vs control (solo desarrollo)",
                 diverging=True, center=center, cbar_label=icfg["univariate_metric"])

# 2) Importancia por modelo + importancia por permutación en holdout + calibración
summary = {}
for name in cfg["models"]:
    with open(out / f"04_{name}_final_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model, feats = bundle["model"], bundle["features"]
    imp = model_feature_importance(model, feats)
    if imp is not None:
        imp.to_csv(out / f"05_{name}_native_importance.csv")
        plot_top_features(imp, int(icfg["top_k_features"]), fig_dir / f"importance_{name}.png", f"{name}: importancia nativa", "coeficiente / importancia")
        v = band_channel_matrix(imp.abs(), meta, kind, icfg["topomap_bands"], fs.channels)
        if v:
            topomap_grid(v, fs.channels, cfg["channels"]["montage"], fig_dir / f"topomap_importance_{name}.png",
                         f"{name}: |importancia| por canal (potencia relativa)", diverging=False, cbar_label="|importancia|")
    if test.any():
        pi = permutation_importance_series(model, X[test].to_numpy(float), fs.y[test], feats,
                                           int(icfg["permutation_importance_repeats"]), seed, cfg["validation"]["scoring"], n_jobs)
        pi.to_csv(out / f"05_{name}_permutation_importance_holdout.csv")
        p = model.predict_proba(X[test].to_numpy(float))[:, 1]
        plot_calibration(fs.y[test], p, int(cfg["validation"]["ece_bins"]), fig_dir / f"calibration_{name}_holdout.png", f"{name}: calibración en holdout")
        summary[name] = {"top_permutation_importance_holdout": pi.head(10)["mean"].round(4).to_dict()}
    cvp = pd.read_csv(out / f"04_{name}_cv_predictions.csv")
    plot_calibration(cvp.y.to_numpy(), cvp.p.to_numpy(), int(cfg["validation"]["ece_bins"]), fig_dir / f"calibration_{name}_cv.png", f"{name}: calibración CV (fuera de muestra)")
(out / "05_interpret_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print("Top 15 efectos univariados (dev):")
print(uni.reindex((uni - center).abs().sort_values(ascending=False).index[:15]).round(3).to_string())

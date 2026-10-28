"""Validación cruzada entre sitios: el modelo final de un pipeline se aplica a las características del otro.

Uso: python scripts/08_cross_site.py --config config/config_ds003478.yaml
Lee external_validation.* del YAML: cada entrada apunta a otra configuración cuyo outputs/03_features.csv
y 02_feature_set.pkl ya existen. Las características deben tener los mismos nombres (mismos canales, bandas, regiones).
Prueba de TRANSFERENCIA entre poblaciones/extractores, no de rendimiento: se reporta junto a sus advertencias.
"""
import json
import pickle

import numpy as np
import pandas as pd

from _bootstrap import get_config
from eegdep.config import load_config, output_dir
from eegdep.validation import bootstrap_ci, compute_metrics

cfg = get_config()
out = output_dir(cfg)
val = cfg["validation"]
nb, seed = int(val["ece_bins"]), int(cfg["project"]["seed"])
results_here = json.loads((out / "04_results.json").read_text())
X_here = pd.read_csv(out / "03_features.csv", index_col=0)
with open(out / "02_feature_set.pkl", "rb") as f:
    fs_here = pickle.load(f)


def apply(model_bundle_path, X_target: pd.DataFrame, y_target: np.ndarray, groups: np.ndarray, thr: float, label: str) -> dict:
    with open(model_bundle_path, "rb") as f:
        saved = pickle.load(f)
    Xa = X_target.reindex(columns=saved["features"])
    missing = Xa.columns[Xa.isna().any()].tolist()
    if missing:
        raise ValueError(f"{label}: características ausentes en el destino: {missing[:8]}")
    p = saved["model"].predict_proba(Xa.to_numpy(float))[:, 1]
    m = compute_metrics(y_target, p, thr, nb)
    ci = bootstrap_ci(y_target, p, groups, thr, nb, int(val["bootstrap"]["n"]), float(val["bootstrap"]["ci"]), seed)
    return {"metrics": m, "ci95": ci, "n": int(len(y_target)), "n_pos": int(y_target.sum())}


summary = {}
for name, ext in cfg.get("external_validation", {}).items():
    if not ext.get("enabled"):
        continue
    other_cfg = load_config((out.parent / ext["config"]) if not str(ext["config"]).startswith("/") else ext["config"])
    other_out = output_dir(other_cfg)
    X_other = pd.read_csv(other_out / "03_features.csv", index_col=0)
    with open(other_out / "02_feature_set.pkl", "rb") as f:
        fs_other = pickle.load(f)
    results_other = json.loads((other_out / "04_results.json").read_text())
    block = {"note": ext.get("note", "")}
    for model_name in cfg["models"]:
        thr_here = results_here[model_name].get("holdout_threshold", float(fs_here.y.mean()))
        block[f"{cfg['project']['name']}__to__{other_cfg['project']['name']}__{model_name}"] = apply(
            out / f"04_{model_name}_final_model.pkl", X_other, fs_other.y, fs_other.groups, thr_here, model_name)
        if model_name in results_other:
            thr_other = results_other[model_name].get("holdout_threshold", float(fs_other.y.mean()))
            block[f"{other_cfg['project']['name']}__to__{cfg['project']['name']}__{model_name}"] = apply(
                other_out / f"04_{model_name}_final_model.pkl", X_here, fs_here.y, fs_here.groups, thr_other, model_name)
    summary[name] = block

(out / "08_cross_site.json").write_text(json.dumps(summary, indent=2, default=str))
D = int(cfg["report"]["decimals"])
print("Transferencia entre sitios (AUC [IC95] · sens · espec · exact. balanceada):")
for name, block in summary.items():
    print(f"== {name}: {block['note']}")
    for k, v in block.items():
        if k == "note":
            continue
        m, ci = v["metrics"], v["ci95"]
        print(f"  {k:70s} {m['auc']:.{D}f} [{ci['auc'][0]:.{D}f}, {ci['auc'][1]:.{D}f}] · {m['sensitivity']:.2f} · {m['specificity']:.2f} · {m['balanced_accuracy']:.2f}  (n={v['n']}, pos={v['n_pos']})")

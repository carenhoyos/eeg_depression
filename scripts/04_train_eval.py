"""Holdout por sujeto + CV anidada agrupada para cada modelo + métricas con IC + permutación."""
import json
import pickle

import numpy as np
import pandas as pd

from _bootstrap import get_config
from eegdep.config import output_dir
from eegdep.models import make_pipeline
from eegdep.validation import (compute_metrics, bootstrap_ci, fit_with_search, holdout_split,
                               permutation_test, run_nested_cv, summarize_oof)

cfg = get_config()
out = output_dir(cfg)
seed, n_jobs, val = int(cfg["project"]["seed"]), int(cfg["project"]["n_jobs"]), cfg["validation"]
with open(out / "02_feature_set.pkl", "rb") as f:
    fs = pickle.load(f)
X = pd.read_csv(out / "03_features.csv", index_col=0)
assert list(X.index) == list(fs.subjects)
Xv, y, groups = X.to_numpy(float), fs.y, fs.groups

# 1) holdout de sujetos que ningún modelo ve durante desarrollo
if val["holdout"]["enabled"]:
    dev, test = holdout_split(y, groups, float(val["holdout"]["fraction"]), seed)
else:
    dev, test = np.arange(len(y)), np.array([], int)
split_info = {"n_dev": int(len(dev)), "n_test": int(len(test)), "dev_pos": int(y[dev].sum()), "test_pos": int(y[test].sum()),
              "dev_subjects": fs.subjects[dev].tolist(), "test_subjects": fs.subjects[test].tolist()}
print(f"dev n={len(dev)} (pos={y[dev].sum()})  holdout n={len(test)} (pos={y[test].sum() if len(test) else 0})")
(out / "04_split.json").write_text(json.dumps(split_info, indent=2))

results = {}
for name, spec in cfg["models"].items():
    print(f"\n=== {name} ===")
    pipe, grid = make_pipeline(spec, val, seed)
    oof, params_log = run_nested_cv(pipe, grid, Xv[dev], y[dev], groups[dev], val, seed, n_jobs)
    summ = summarize_oof(oof, val, seed)
    res = {"cv_per_repeat_mean": summ["per_repeat_mean"], "cv_per_repeat_sd": summ["per_repeat_sd"],
           "cv_pooled": summ["pooled_point"], "cv_pooled_ci95": summ["pooled_ci"], "best_params_per_fold": params_log}
    print("CV (media por repetición):", {k: round(v, 3) for k, v in summ["per_repeat_mean"].items() if k in ("auc", "sensitivity", "specificity", "f1", "balanced_accuracy", "brier", "ece")})
    print("CV IC95 (bootstrap por sujeto):", {k: tuple(round(x, 3) for x in v) for k, v in summ["pooled_ci"].items() if k in ("auc", "balanced_accuracy")})
    summ["subject_level_predictions"].assign(subject=fs.subjects[dev][summ["subject_level_predictions"].index]).to_csv(out / f"04_{name}_cv_predictions.csv", index=False)

    # modelo final: ajustado en TODO el dev (búsqueda interna agrupada) y evaluado UNA vez en holdout
    final, best_uncal, best_params = fit_with_search(pipe, grid, Xv[dev], y[dev], groups[dev], val, seed, n_jobs)
    res["final_best_params"] = best_params
    if len(test):
        p = final.predict_proba(Xv[test])[:, 1]
        res["holdout"] = compute_metrics(y[test], p, float(val["threshold"]), int(val["ece_bins"]))
        res["holdout_ci95"] = bootstrap_ci(y[test], p, groups[test], float(val["threshold"]), int(val["ece_bins"]),
                                           int(val["bootstrap"]["n"]), float(val["bootstrap"]["ci"]), seed)
        pd.DataFrame({"subject": fs.subjects[test], "y": y[test], "p": p}).to_csv(out / f"04_{name}_holdout_predictions.csv", index=False)
        print("HOLDOUT:", {k: round(v, 3) for k, v in res["holdout"].items() if k in ("auc", "sensitivity", "specificity", "f1", "balanced_accuracy", "brier", "ece")},
              " AUC IC95:", tuple(round(x, 3) for x in res["holdout_ci95"]["auc"]))
    if val["permutation_test"]["enabled"]:
        res["permutation"] = permutation_test(pipe, best_params, Xv[dev], y[dev], groups[dev], val,
                                              summ["pooled_point"]["auc"], int(val["permutation_test"]["n_permutations"]), seed, n_jobs)
        print("Permutación:", res["permutation"])
    with open(out / f"04_{name}_final_model.pkl", "wb") as f:
        pickle.dump({"model": final, "uncalibrated": best_uncal, "features": list(X.columns)}, f)
    results[name] = res

(out / "04_results.json").write_text(json.dumps(results, indent=2, default=str))
print("\nResumen (AUC): CV media ± DE | holdout [IC95]")
for name, r in results.items():
    h = r.get("holdout", {}).get("auc", float("nan"))
    hc = r.get("holdout_ci95", {}).get("auc", (float("nan"),) * 2)
    print(f"  {name:15s} {r['cv_per_repeat_mean']['auc']:.3f} ± {r['cv_per_repeat_sd']['auc']:.3f} | {h:.3f} [{hc[0]:.3f}, {hc[1]:.3f}]")

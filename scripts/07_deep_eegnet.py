"""EEGNet sobre épocas crudas con holdout y CV externa por SUJETO (mismo holdout que 04_train_eval.py).
Requiere 02_epochs.npz (data.kind == raw) y 04_split.json. Métricas a nivel sujeto con IC bootstrap."""
import json

import numpy as np
import pandas as pd

from _bootstrap import get_config
from eegdep.config import output_dir
from eegdep.deep import ChannelScaler, predict_subjects, split_train_val_subjects, stack_subjects, subsample_epochs, train_eegnet
from eegdep.validation import bootstrap_ci, compute_metrics, outer_splits, resolve_threshold

cfg = get_config()
deep, val = cfg["deep"], cfg["validation"]
if not deep["enabled"] or cfg["data"]["kind"] != "raw":
    raise SystemExit("deep.enabled es false o no hay señal cruda: nada que hacer")
out = output_dir(cfg)
seed = int(cfg["project"]["seed"])
npz = np.load(out / "02_epochs.npz")
epochs_by_subject = subsample_epochs({k: npz[k] for k in npz.files}, deep.get("max_epochs_per_subject"), seed)
split = json.loads((out / "04_split.json").read_text())
proc = pd.read_csv(out / "02_processing_log.csv").set_index("subject")
y_by_subject = {s: int(proc.loc[s, "y"]) for s in epochs_by_subject}
dev_subjects = np.array([s for s in split["dev_subjects"] if s in epochs_by_subject])
test_subjects = np.array([s for s in split["test_subjects"] if s in epochs_by_subject])
y_dev = np.array([y_by_subject[s] for s in dev_subjects])
print(f"EEGNet: dev {len(dev_subjects)} sujetos, holdout {len(test_subjects)}, épocas/sujeto≈{np.mean([len(v) for v in epochs_by_subject.values()]):.0f}, "
      f"forma {next(iter(epochs_by_subject.values())).shape[1:]}")


def fit_on(subjects_tr: np.ndarray, seed_: int, label: str = ""):
    """Separa sujetos de validación interna para parada temprana, escala por canal y entrena."""
    s_tr, s_va = split_train_val_subjects(subjects_tr, np.array([y_by_subject[s] for s in subjects_tr]), float(deep["validation_fraction"]), seed_)
    X_tr, y_tr, _ = stack_subjects(epochs_by_subject, list(s_tr), y_by_subject)
    X_va, y_va, g_va = stack_subjects(epochs_by_subject, list(s_va), y_by_subject)
    sc = ChannelScaler().fit(X_tr)
    model, info = train_eegnet(sc.transform(X_tr), y_tr, sc.transform(X_va), y_va, g_va, deep, seed_, log_prefix=f"  [{label}] ")
    return model, sc, info


# ---- CV externa por sujeto dentro de dev
thr_cfg, nb = val["threshold"], int(val["ece_bins"])
rows = []
for rep, k, tr, te in outer_splits(y_dev, dev_subjects, {"scheme": "group_kfold", **deep["outer"]}, seed):
    model, sc, info = fit_on(dev_subjects[tr], seed + 100 * rep + k, label=f"rep {rep} fold {k}")
    X_te, _, g_te = stack_subjects(epochs_by_subject, list(dev_subjects[te]), y_by_subject)
    subs, p = predict_subjects(model, sc.transform(X_te), g_te, deep["device"])
    thr = resolve_threshold(thr_cfg, y_dev[tr])
    rows += [{"rep": rep, "fold": k, "subject": s, "y": y_by_subject[s], "p": float(pp), "thr": thr} for s, pp in zip(subs, p)]
    print(f"  rep {rep} fold {k}: val_auc={info['best_val_subject_auc']:.3f} epochs={info['epochs_run']} test_auc={compute_metrics(np.array([y_by_subject[s] for s in subs]), p, thr, nb)['auc']:.3f}")
oof = pd.DataFrame(rows)
oof.to_csv(out / "07_eegnet_cv_predictions.csv", index=False)
agg = oof.groupby("subject").agg(y=("y", "first"), p=("p", "mean"), thr=("thr", "mean"))
cv_point = compute_metrics(agg.y.to_numpy(), agg.p.to_numpy(), agg.thr.to_numpy(), nb)
cv_ci = bootstrap_ci(agg.y.to_numpy(), agg.p.to_numpy(), agg.index.to_numpy(), agg.thr.to_numpy(), nb, int(val["bootstrap"]["n"]), float(val["bootstrap"]["ci"]), seed)

# ---- modelo final en todo dev, evaluado una vez en holdout
res = {"cv_pooled": cv_point, "cv_pooled_ci95": cv_ci}
if len(test_subjects):
    model, sc, info = fit_on(dev_subjects, seed, label="final")
    X_te, _, g_te = stack_subjects(epochs_by_subject, list(test_subjects), y_by_subject)
    subs, p = predict_subjects(model, sc.transform(X_te), g_te, deep["device"])
    y_te = np.array([y_by_subject[s] for s in subs])
    thr = resolve_threshold(thr_cfg, y_dev)
    res["holdout"] = compute_metrics(y_te, p, thr, nb)
    res["holdout_ci95"] = bootstrap_ci(y_te, p, subs, thr, nb, int(val["bootstrap"]["n"]), float(val["bootstrap"]["ci"]), seed)
    res["final_training"] = info
    pd.DataFrame({"subject": subs, "y": y_te, "p": p, "thr": thr}).to_csv(out / "07_eegnet_holdout_predictions.csv", index=False)
    import torch
    torch.save({"state_dict": model.state_dict(), "scaler_mu": sc.mu, "scaler_sd": sc.sd, "cfg": deep}, out / "07_eegnet_final.pt")
(out / "07_eegnet_results.json").write_text(json.dumps(res, indent=2, default=str))
M, D = cfg["report"]["metrics"], int(cfg["report"]["decimals"])
print("CV por sujeto:", {m: round(cv_point[m], D) for m in M}, " IC AUC:", tuple(round(x, D) for x in cv_ci["auc"]))
if "holdout" in res:
    print("HOLDOUT:", {m: round(res["holdout"][m], D) for m in M}, " IC AUC:", tuple(round(x, D) for x in res["holdout_ci95"]["auc"]))

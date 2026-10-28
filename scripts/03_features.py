"""Tabla de características por sujeto según features.* de la configuración."""
import pickle

from _bootstrap import get_config
from eegdep.config import output_dir
from eegdep.features import build_feature_table

cfg = get_config()
out = output_dir(cfg)
with open(out / "02_feature_set.pkl", "rb") as f:
    fs = pickle.load(f)
X, meta = build_feature_table(fs, cfg)
print(f"X: {X.shape}  por tipo: {meta.kind.value_counts().to_dict()}")
X.to_csv(out / "03_features.csv")
meta.to_csv(out / "03_feature_meta.csv")

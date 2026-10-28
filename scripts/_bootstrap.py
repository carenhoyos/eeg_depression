"""Inserta src/ en sys.path y parsea --config. Importado por todos los scripts."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eegdep.config import load_config, set_seeds  # noqa: E402


def get_config(default="config/config.yaml"):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / default))
    cfg = load_config(ap.parse_args().config)
    set_seeds(int(cfg["project"]["seed"]))
    return cfg

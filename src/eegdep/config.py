"""Carga y validación de la configuración YAML. Única fuente de parámetros."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REQUIRED_SECTIONS = ("project", "data", "channels", "bands", "qc", "features", "validation", "models")


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise KeyError(f"Secciones faltantes en {path}: {missing}")
    cfg["_config_path"] = str(path)
    cfg["_root"] = str(path.parent.parent)  # <repo>/config/config.yaml -> <repo>
    _validate(cfg)
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    d = cfg["data"]
    if d["kind"] not in ("features", "raw"):
        raise ValueError("data.kind debe ser 'features' o 'raw'")
    ch = cfg["channels"]
    renamed = [ch["rename"].get(c, c) for c in ch["list"]]
    for region, chans in ch["regions"].items():
        unknown = set(chans) - set(renamed)
        if unknown:
            raise ValueError(f"Región {region} usa canales no definidos: {unknown}")
    for left, right in ch["asymmetry_pairs"]:
        if left not in renamed or right not in renamed:
            raise ValueError(f"Par de asimetría inválido: {left}, {right}")
    v = cfg["validation"]
    if v["outer"]["scheme"] not in ("group_kfold", "loso"):
        raise ValueError("validation.outer.scheme debe ser group_kfold o loso")
    if not 0 < v["holdout"]["fraction"] < 1:
        raise ValueError("validation.holdout.fraction debe estar en (0,1)")


def resolve_path(cfg: dict[str, Any], p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else Path(cfg["_root"]) / p


def output_dir(cfg: dict[str, Any], *sub: str) -> Path:
    out = resolve_path(cfg, cfg["project"]["output_dir"]).joinpath(*sub)
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # opcional

        torch.manual_seed(seed)
    except ImportError:
        pass


def renamed_channels(cfg: dict[str, Any]) -> list[str]:
    ch = cfg["channels"]
    return [ch["rename"].get(c, c) for c in ch["list"]]

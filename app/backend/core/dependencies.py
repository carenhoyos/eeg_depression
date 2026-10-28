"""Contenedor de dependencias: se construye una vez en el arranque (lifespan) y se inyecta en los routers."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from fastapi import Request

from .config import ROOT, Settings, get_settings

sys.path.insert(0, str(ROOT / "src"))
from eegdep.config import load_config  # noqa: E402
from eegdep.inference import ModelBundle  # noqa: E402

from ..infra.queue import InProcessQueue, JobQueue  # noqa: E402
from ..infra.storage import LocalStorage, Storage  # noqa: E402


@dataclass
class Container:
    settings: Settings
    pipeline_cfg: dict[str, Any]
    bundle: ModelBundle
    storage: Storage
    queue: JobQueue


def build_container(settings: Settings | None = None) -> Container:
    settings = settings or get_settings()
    pipeline_cfg = load_config(settings.resolve(settings.model.pipeline_config))
    bundle = ModelBundle.load(settings.resolve(settings.model.bundle_path))
    storage = LocalStorage(settings.resolve(settings.app.storage_dir))
    return Container(settings=settings, pipeline_cfg=pipeline_cfg, bundle=bundle, storage=storage, queue=InProcessQueue(storage))


def get_container(request: Request) -> Container:
    return request.app.state.container

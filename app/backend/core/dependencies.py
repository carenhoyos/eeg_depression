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

from ..infra.db import Database  # noqa: E402
from ..infra.queue import InProcessQueue, JobQueue  # noqa: E402
from ..infra.storage import LocalStorage, Storage  # noqa: E402


@dataclass
class Container:
    settings: Settings
    pipeline_cfg: dict[str, Any]
    csv_cfg: dict[str, Any]
    bundle: ModelBundle
    storage: Storage
    queue: JobQueue
    db: Database


def build_container(settings: Settings | None = None) -> Container:
    from .security import hash_password  # import tardío: security depende de este módulo

    settings = settings or get_settings()
    pipeline_cfg = load_config(settings.resolve(settings.model.pipeline_config))
    csv_cfg = load_config(settings.resolve(settings.model.csv_features_config)) if settings.model.csv_features_config else pipeline_cfg
    bundle = ModelBundle.load(settings.resolve(settings.model.bundle_path))
    storage = LocalStorage(settings.resolve(settings.app.storage_dir))
    url = settings.database.url
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        url = "sqlite:///" + str(settings.resolve(url[len("sqlite:///"):]))
    db = Database(url)
    auth = settings.auth
    if auth.bootstrap_username and auth.bootstrap_password:
        db.ensure_user(auth.bootstrap_username, hash_password(auth.bootstrap_password, auth.pbkdf2_iterations), auth.bootstrap_role, auth.bootstrap_full_name)
    return Container(settings=settings, pipeline_cfg=pipeline_cfg, csv_cfg=csv_cfg, bundle=bundle, storage=storage, queue=InProcessQueue(storage), db=db)


def get_container(request: Request) -> Container:
    return request.app.state.container

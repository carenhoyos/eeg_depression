"""Fábrica de la aplicación FastAPI.

Ejecución local (desde la raíz del repo):
    python -m uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
Configuración: app/config/app.yaml, sobreescribible con EEGDEP_APP_CONFIG y EEGDEP_<SECCION>__<CLAVE>.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.v1.routers import analyses, auth, meta, patients
from .core.config import Settings, get_settings
from .core.dependencies import build_container


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.container = build_container(settings)   # carga el bundle una sola vez
        yield

    app = FastAPI(title=settings.app.name, version=settings.app.version, lifespan=lifespan,
                  docs_url=f"{settings.app.api_prefix}/docs", openapi_url=f"{settings.app.api_prefix}/openapi.json")
    app.add_middleware(CORSMiddleware, allow_origins=settings.app.cors_origins, allow_methods=["*"], allow_headers=["*"])
    for r in (meta.router, auth.router, patients.router, analyses.router):
        app.include_router(r, prefix=settings.app.api_prefix)
    frontend = settings.resolve(settings.app.frontend_dir)
    if frontend.exists():
        app.mount("/assets", StaticFiles(directory=frontend / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str, request: Request):
            # Archivos estáticos reales (favicon, iconos) o, para rutas del cliente, index.html (SPA fallback).
            if path.startswith(settings.app.api_prefix.lstrip("/")):
                raise HTTPException(404)
            candidate = frontend / path
            if path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(frontend / "index.html")
    return app


app = create_app()

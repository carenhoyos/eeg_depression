"""Endpoints de metadatos: salud, ficha del modelo y configuración del informe para los clientes."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ....core.dependencies import Container, get_container
from ....domain.schemas import ModelCard, ReportConfig

router = APIRouter(tags=["meta"])


@router.get("/health")
def health(c: Container = Depends(get_container)):
    return {"status": "ok", "app": c.settings.app.name, "version": c.settings.app.version, "model_version": c.bundle.version}


@router.get("/model", response_model=ModelCard)
def model_card(c: Container = Depends(get_container)):
    return c.bundle.card()


@router.get("/config/report", response_model=ReportConfig)
def report_config(c: Container = Depends(get_container)):
    """Niveles, prevalencia por defecto y formatos: el cliente los lee de aquí, nunca los fija."""
    r, i = c.settings.report, c.settings.input
    return {"tiers": [t.model_dump() for t in r.tiers], "pretest_prevalence_default": r.pretest_prevalence_default,
            "formats": i.formats, "max_upload_mb": c.settings.app.max_upload_mb, "conditions": i.conditions}

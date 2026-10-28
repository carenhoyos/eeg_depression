"""Endpoints de análisis: creación (upload), estado, informe, figuras y PDF."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ....core.dependencies import Container, get_container
from ....domain.analysis import detect_input_kind, run_analysis
from ....domain.schemas import AnalysisStatus, Report

router = APIRouter(prefix="/analyses", tags=["analyses"])


@router.post("", response_model=AnalysisStatus, status_code=202)
async def create_analysis(c: Container = Depends(get_container), file: UploadFile = File(...),
                          pretest_prevalence: Optional[float] = Form(None), sampling_rate_hz: Optional[float] = Form(None),
                          patient_ref: Optional[str] = Form(None), age: Optional[int] = Form(None), sex: Optional[str] = Form(None),
                          condition: Optional[str] = Form(None), medication: Optional[str] = Form(None), notes: Optional[str] = Form(None)):
    kind = detect_input_kind(file.filename or "", c.settings)
    if kind is None:
        raise HTTPException(400, f"Extensión no soportada: {Path(file.filename or '').suffix}")
    pretest = float(pretest_prevalence if pretest_prevalence is not None else c.settings.report.pretest_prevalence_default)
    if not 0 < pretest < 1:
        raise HTTPException(400, "pretest_prevalence debe estar en (0, 1)")
    analysis_id, d = c.storage.new_analysis()
    dest = d / f"input{Path(file.filename or '').suffix.lower()}"
    max_bytes = c.settings.app.max_upload_mb * 1024 * 1024
    size = 0
    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                shutil.rmtree(d, ignore_errors=True)
                raise HTTPException(413, "Archivo demasiado grande")
            f.write(chunk)
    recording = {"patient_ref": patient_ref, "age": age, "sex": sex, "condition": condition, "medication": medication, "notes": notes}
    c.queue.enqueue(analysis_id, lambda: run_analysis(analysis_id, d, dest, kind, recording, pretest, sampling_rate_hz,
                                                     c.settings, c.bundle, c.pipeline_cfg))
    return {"analysis_id": analysis_id, "status": "queued", "report_url": f"{c.settings.app.api_prefix}/analyses/{analysis_id}"}


@router.get("", summary="Lista de análisis recientes")
def list_analyses(limit: int = 50, c: Container = Depends(get_container)):
    return c.storage.list_analyses(limit)


@router.get("/{analysis_id}/status", response_model=AnalysisStatus)
def get_status(analysis_id: str, c: Container = Depends(get_container)):
    try:
        s = c.storage.read_status(analysis_id)
    except FileNotFoundError:
        raise HTTPException(404, "análisis no encontrado")
    return {"analysis_id": analysis_id, "status": s["status"], "error": s.get("error"),
            "report_url": f"{c.settings.app.api_prefix}/analyses/{analysis_id}" if s["status"] == "done" else None}


@router.get("/{analysis_id}", response_model=Report)
def get_report(analysis_id: str, c: Container = Depends(get_container)):
    try:
        if c.storage.read_status(analysis_id)["status"] != "done":
            raise HTTPException(409, "el análisis no ha terminado")
        return c.storage.read_report(analysis_id)
    except FileNotFoundError:
        raise HTTPException(404, "análisis no encontrado")


@router.get("/{analysis_id}/figures/{name}")
def get_figure(analysis_id: str, name: str, c: Container = Depends(get_container)):
    try:
        p = c.storage.dir(analysis_id) / "figures" / Path(name).name
    except FileNotFoundError:
        raise HTTPException(404, "análisis no encontrado")
    if not p.exists():
        raise HTTPException(404, "figura no encontrada")
    return FileResponse(p, media_type="image/png")


@router.get("/{analysis_id}/report.pdf")
def get_pdf(analysis_id: str, c: Container = Depends(get_container)):
    try:
        p = c.storage.dir(analysis_id) / "report.pdf"
    except FileNotFoundError:
        raise HTTPException(404, "análisis no encontrado")
    if not p.exists():
        raise HTTPException(404, "PDF no disponible")
    return FileResponse(p, media_type="application/pdf", filename=f"informe_{analysis_id}.pdf")

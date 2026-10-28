"""Endpoints de análisis: creación (carga de uno o varios archivos), estado, informe, figuras y PDF.
Todos requieren usuario autenticado; cada análisis pertenece al usuario y, opcionalmente, a un paciente."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ....core.dependencies import Container, get_container
from ....core.security import CurrentUser, current_user
from ....domain.analysis import detect_input_kind, run_analysis
from ....domain.schemas import AnalysisStatus, AnalysisSummary, Report

router = APIRouter(prefix="/analyses", tags=["analyses"])


async def _save_upload(file: UploadFile, dest: Path, max_bytes: int) -> int:
    size = 0
    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(413, "Archivo demasiado grande")
            f.write(chunk)
    return size


@router.post("", response_model=AnalysisStatus, status_code=202)
async def create_analysis(c: Container = Depends(get_container), user: CurrentUser = Depends(current_user),
                          files: list[UploadFile] = File(..., description="Archivo EEG; para EEGLAB subir .set y .fdt juntos"),
                          patient_id: Optional[int] = Form(None), pretest_prevalence: Optional[float] = Form(None),
                          sampling_rate_hz: Optional[float] = Form(None), condition: Optional[str] = Form(None),
                          medication: Optional[str] = Form(None), notes: Optional[str] = Form(None)):
    primary = next((f for f in files if detect_input_kind(f.filename or "", c.settings)), None)
    if primary is None:
        raise HTTPException(400, f"Ningún archivo con extensión soportada: {[f.filename for f in files]}")
    kind = detect_input_kind(primary.filename or "", c.settings)
    pretest = float(pretest_prevalence if pretest_prevalence is not None else c.settings.report.pretest_prevalence_default)
    if not 0 < pretest < 1:
        raise HTTPException(400, "pretest_prevalence debe estar en (0, 1)")
    if condition and condition not in c.settings.input.conditions:
        raise HTTPException(400, f"condition debe ser una de {c.settings.input.conditions}")
    patient = None
    if patient_id is not None:
        patient = c.db.get_patient(user.id, patient_id)
        if patient is None:
            raise HTTPException(404, "paciente no encontrado")
    analysis_id, d = c.storage.new_analysis()
    max_bytes = c.settings.app.max_upload_mb * 1024 * 1024
    try:
        stem = Path(primary.filename or "input").stem
        for f in files:
            await _save_upload(f, d / f"{stem}{Path(f.filename or '').suffix.lower()}", max_bytes)
    except HTTPException:
        shutil.rmtree(d, ignore_errors=True)
        raise
    dest = d / f"{stem}{Path(primary.filename or '').suffix.lower()}"
    recording = {"patient_ref": patient.pseudonym if patient else None, "patient_id": patient_id,
                 "age": patient.age if patient else None, "sex": patient.sex if patient else None,
                 "condition": condition, "medication": medication or (patient.medication if patient else None), "notes": notes}
    c.db.create_analysis(id=analysis_id, patient_id=patient_id, owner_id=user.id, status="queued", input_kind=kind,
                         filename=primary.filename or "", pretest_prevalence=pretest, condition=condition)

    def _on_finish(status: str, report: Optional[dict], error: Optional[str]):
        fields = {"status": status, "error": error}
        if report and report["results"]:
            r0 = report["results"][0]
            fields.update(p_calibrated=r0["p_calibrated"], p_posterior=r0["p_posterior"], tier_code=r0["tier"]["code"], model_version=report["model"]["version"])
        c.db.update_analysis(analysis_id, **fields)

    c.queue.enqueue(analysis_id, lambda: run_analysis(analysis_id, d, dest, kind, recording, pretest, sampling_rate_hz,
                                                     c.settings, c.bundle, c.pipeline_cfg, c.csv_cfg), on_finish=_on_finish)
    return {"analysis_id": analysis_id, "status": "queued", "report_url": f"{c.settings.app.api_prefix}/analyses/{analysis_id}"}


@router.get("", response_model=list[AnalysisSummary], summary="Análisis del usuario (opcionalmente de un paciente)")
def list_analyses(patient_id: Optional[int] = None, limit: int = 100, user: CurrentUser = Depends(current_user), c: Container = Depends(get_container)):
    return [a.to_dict() for a in c.db.list_analyses(user.id, patient_id=patient_id, limit=limit)]


def _owned(analysis_id: str, user: CurrentUser, c: Container):
    if c.db.get_analysis(user.id, analysis_id) is None:
        raise HTTPException(404, "análisis no encontrado")


@router.get("/{analysis_id}/status", response_model=AnalysisStatus)
def get_status(analysis_id: str, user: CurrentUser = Depends(current_user), c: Container = Depends(get_container)):
    _owned(analysis_id, user, c)
    s = c.storage.read_status(analysis_id)
    return {"analysis_id": analysis_id, "status": s["status"], "error": s.get("error"),
            "report_url": f"{c.settings.app.api_prefix}/analyses/{analysis_id}" if s["status"] == "done" else None}


@router.get("/{analysis_id}", response_model=Report)
def get_report(analysis_id: str, user: CurrentUser = Depends(current_user), c: Container = Depends(get_container)):
    _owned(analysis_id, user, c)
    if c.storage.read_status(analysis_id)["status"] != "done":
        raise HTTPException(409, "el análisis no ha terminado")
    return c.storage.read_report(analysis_id)


@router.get("/{analysis_id}/figures/{name}")
def get_figure(analysis_id: str, name: str, user: CurrentUser = Depends(current_user), c: Container = Depends(get_container)):
    _owned(analysis_id, user, c)
    p = c.storage.dir(analysis_id) / "figures" / Path(name).name
    if not p.exists():
        raise HTTPException(404, "figura no encontrada")
    return FileResponse(p, media_type="image/png")


@router.get("/{analysis_id}/report.pdf")
def get_pdf(analysis_id: str, user: CurrentUser = Depends(current_user), c: Container = Depends(get_container)):
    _owned(analysis_id, user, c)
    p = c.storage.dir(analysis_id) / "report.pdf"
    if not p.exists():
        raise HTTPException(404, "PDF no disponible")
    return FileResponse(p, media_type="application/pdf", filename=f"informe_{analysis_id}.pdf")

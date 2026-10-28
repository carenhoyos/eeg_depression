"""Pacientes del psiquiatra autenticado (pseudónimos, nunca datos identificables) y su historial de análisis."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ....core.dependencies import Container, get_container
from ....core.security import CurrentUser, current_user
from ....domain.schemas import AnalysisSummary, PatientIn, PatientOut, PatientUpdate

router = APIRouter(prefix="/patients", tags=["patients"])


def _out(p) -> dict:
    return {"id": p.id, "pseudonym": p.pseudonym, "age": p.age, "sex": p.sex, "medication": p.medication, "notes": p.notes,
            "created_at": p.created_at.isoformat()}


@router.get("", response_model=list[PatientOut])
def list_patients(user: CurrentUser = Depends(current_user), c: Container = Depends(get_container)):
    return [_out(p) for p in c.db.list_patients(user.id)]


@router.post("", response_model=PatientOut, status_code=201)
def create_patient(body: PatientIn, user: CurrentUser = Depends(current_user), c: Container = Depends(get_container)):
    return _out(c.db.create_patient(user.id, **body.model_dump()))


@router.get("/{patient_id}", response_model=PatientOut)
def get_patient(patient_id: int, user: CurrentUser = Depends(current_user), c: Container = Depends(get_container)):
    p = c.db.get_patient(user.id, patient_id)
    if p is None:
        raise HTTPException(404, "paciente no encontrado")
    return _out(p)


@router.patch("/{patient_id}", response_model=PatientOut)
def update_patient(patient_id: int, body: PatientUpdate, user: CurrentUser = Depends(current_user), c: Container = Depends(get_container)):
    p = c.db.update_patient(user.id, patient_id, **body.model_dump(exclude_unset=True))
    if p is None:
        raise HTTPException(404, "paciente no encontrado")
    return _out(p)


@router.delete("/{patient_id}", status_code=204)
def delete_patient(patient_id: int, user: CurrentUser = Depends(current_user), c: Container = Depends(get_container)):
    if not c.db.delete_patient(user.id, patient_id):
        raise HTTPException(404, "paciente no encontrado")


@router.get("/{patient_id}/analyses", response_model=list[AnalysisSummary])
def patient_analyses(patient_id: int, user: CurrentUser = Depends(current_user), c: Container = Depends(get_container)):
    if c.db.get_patient(user.id, patient_id) is None:
        raise HTTPException(404, "paciente no encontrado")
    return [a.to_dict() for a in c.db.list_analyses(user.id, patient_id=patient_id)]

"""Contrato de la API (Pydantic). Lo consumen el cliente React y, después, el cliente Flutter."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ----------------------------------------------------------------- auth / usuarios
class LoginRequest(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    role: str
    full_name: str = ""


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserOut


# ----------------------------------------------------------------- pacientes
class PatientIn(BaseModel):
    pseudonym: str = Field(..., max_length=64, description="Pseudónimo; nunca nombre ni documento")
    age: Optional[int] = Field(None, ge=0, le=120)
    sex: Optional[str] = Field(None, max_length=16)
    medication: Optional[str] = Field(None, max_length=200)
    notes: Optional[str] = Field(None, max_length=2000)


class PatientUpdate(BaseModel):
    pseudonym: Optional[str] = Field(None, max_length=64)
    age: Optional[int] = Field(None, ge=0, le=120)
    sex: Optional[str] = Field(None, max_length=16)
    medication: Optional[str] = Field(None, max_length=200)
    notes: Optional[str] = Field(None, max_length=2000)


class PatientOut(PatientIn):
    id: int
    created_at: str


class AnalysisSummary(BaseModel):
    analysis_id: str
    patient_id: Optional[int]
    created_at: str
    status: str
    input_kind: str
    filename: str
    pretest_prevalence: float
    condition: Optional[str] = None
    p_calibrated: Optional[float] = None
    p_posterior: Optional[float] = None
    tier_code: Optional[str] = None
    model_version: Optional[str] = None
    error: Optional[str] = None


# ----------------------------------------------------------------- informe
class RecordingMeta(BaseModel):
    patient_ref: Optional[str] = Field(None, description="Pseudónimo del paciente")
    patient_id: Optional[int] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    condition: Optional[str] = Field(None, description="eyes_closed | eyes_open")
    medication: Optional[str] = None
    notes: Optional[str] = None


class Tier(BaseModel):
    code: str
    name: str
    text: str
    min: float
    max: float


class Finding(BaseModel):
    feature: str
    kind: str
    band: str
    location: str
    value: float
    z: float
    reference_p5: float
    reference_p50: float
    reference_p95: float
    direction: str


class Contribution(BaseModel):
    feature: str
    contribution: float


class ChannelValue(BaseModel):
    channel: str
    value: float


class SignalExcerpt(BaseModel):
    """Fragmento de señal limpia submuestreado para el visor (uV) y espectro por canal (log10 uV^2/Hz)."""
    sfreq: float
    channels: list[str]
    t0: float
    duration_s: float
    data: list[list[float]]           # (n_ch, n_t)
    psd_freqs: list[float]
    psd: list[list[float]]            # (n_ch, n_freq)


class RecordResult(BaseModel):
    record_id: str
    p_calibrated: float
    likelihood_ratio: float
    pretest_prevalence: float
    p_posterior: float
    above_threshold: bool
    tier: Tier
    preliminary_assessment: str
    findings: list[Finding]
    contributions: list[Contribution]
    band_relative_power: dict[str, float]
    band_reference: dict[str, dict[str, float]]
    topography_z: dict[str, list[ChannelValue]]
    coherence_matrix: dict[str, Any]
    signal: Optional[SignalExcerpt] = None


class ModelCard(BaseModel):
    model_name: str
    version: str
    created_at: str
    n_features: int
    train_prevalence: float
    threshold: float
    n_train: dict[str, int]
    validation: dict[str, Any]


class QualitySummary(BaseModel):
    input_kind: str
    n_records: int
    warnings: list[str]
    processing_log: dict[str, Any]


class Report(BaseModel):
    analysis_id: str
    created_at: str
    app_version: str
    status: str
    recording: RecordingMeta
    quality: QualitySummary
    model: ModelCard
    results: list[RecordResult]
    figures: dict[str, str]
    pdf_url: Optional[str] = None
    disclaimer: str
    limitations: list[str]


class AnalysisStatus(BaseModel):
    analysis_id: str
    status: str
    error: Optional[str] = None
    report_url: Optional[str] = None


class ReportConfig(BaseModel):
    tiers: list[Tier]
    pretest_prevalence_default: float
    formats: dict[str, list[str]]
    max_upload_mb: int
    conditions: list[str]

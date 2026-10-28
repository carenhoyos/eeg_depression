"""Contrato de la API (Pydantic). Lo consumen el cliente React y, después, el cliente Flutter."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class RecordingMeta(BaseModel):
    patient_ref: Optional[str] = Field(None, description="Pseudónimo; nunca datos identificables")
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


class RecordResult(BaseModel):
    record_id: str
    p_calibrated: float
    likelihood_ratio: float
    pretest_prevalence: float
    p_posterior: float
    above_threshold: bool
    tier: Tier
    findings: list[Finding]
    contributions: list[Contribution]
    band_relative_power: dict[str, float]
    band_reference: dict[str, dict[str, float]]
    topography_z: dict[str, list[ChannelValue]]     # banda -> z por canal (para gráficos interactivos)
    coherence_matrix: dict[str, Any]                # {band, channels, values[][]}


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

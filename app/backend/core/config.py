"""Configuración tipada de la aplicación: app.yaml + variables de entorno (EEGDEP_*) que la sobreescriben."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from typing import Optional

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[3]
ENV_PREFIX = "EEGDEP_"


class AppSection(BaseModel):
    name: str
    version: str
    host: str
    port: int
    cors_origins: list[str]
    storage_dir: str
    max_upload_mb: int
    api_prefix: str
    frontend_dir: str


class ModelSection(BaseModel):
    bundle_path: str
    pipeline_config: str
    csv_features_config: Optional[str] = None


class RawInput(BaseModel):
    default_sampling_rate_hz: Optional[float] = None
    min_duration_s: float
    coherence_nperseg_s: float
    domain_shift_warning: str


class InputSection(BaseModel):
    formats: dict[str, list[str]]
    conditions: list[str]
    raw: RawInput


class AuthSection(BaseModel):
    enabled: bool
    secret_key: str
    algorithm: str
    token_expire_minutes: int
    pbkdf2_iterations: int
    allowed_roles: list[str]
    bootstrap_username: str
    bootstrap_password: str
    bootstrap_role: str
    bootstrap_full_name: str = ""


class DatabaseSection(BaseModel):
    url: str


class SignalViewerSection(BaseModel):
    excerpt_start_s: Optional[float] = None
    excerpt_duration_s: float
    excerpt_sfreq: float
    psd_fmin: float
    psd_fmax: float
    psd_nperseg_s: float
    trace_spacing_uv: float


class TierSpec(BaseModel):
    name: str
    min: float
    max: float
    code: str
    text: str


class PdfSection(BaseModel):
    page_size: str
    title: str


class PlottingSection(BaseModel):
    sequential_cmap: str
    diverging_cmap: str
    color_primary: str
    color_negative: str
    color_reference: str
    tier_colors: dict[str, str]


class ReportSection(BaseModel):
    language: str
    pretest_prevalence_default: float
    tiers: list[TierSpec]
    reference_group: str
    reference_percentiles: list[int]
    deviation_z_threshold: float
    max_findings: int
    top_contributions: int
    topomap_bands: list[str]
    coherence_matrix_band: str
    figures_dpi: int
    pdf: PdfSection
    disclaimer: str
    plotting: PlottingSection
    limitations: list[str] = Field(default_factory=list)
    assessment_template: str
    assessment_max_findings: int
    assessment_no_findings_text: str


class Settings(BaseModel):
    app: AppSection
    auth: AuthSection
    database: DatabaseSection
    signal_viewer: SignalViewerSection
    model: ModelSection
    input: InputSection
    report: ReportSection
    config_path: Path

    def resolve(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else ROOT / p


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """EEGDEP_APP__PORT=9000 -> data['app']['port'] = 9000 (doble guion bajo separa niveles)."""
    for key, val in os.environ.items():
        if not key.startswith(ENV_PREFIX) or key == f"{ENV_PREFIX}APP_CONFIG":
            continue
        path = key[len(ENV_PREFIX):].lower().split("__")
        node = data
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = yaml.safe_load(val)
    return data


@lru_cache
def get_settings(config_path: str | None = None) -> Settings:
    path = Path(config_path or os.environ.get(f"{ENV_PREFIX}APP_CONFIG", ROOT / "app/config/app.yaml"))
    data = _apply_env_overrides(yaml.safe_load(path.read_text(encoding="utf-8")))
    return Settings(**data, config_path=path)

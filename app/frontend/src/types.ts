// Tipos espejo del contrato Pydantic (app/backend/domain/schemas.py).
export interface Tier { code: string; name: string; text: string; min: number; max: number }

export interface Finding {
  feature: string; kind: string; band: string; location: string; value: number; z: number
  reference_p5: number; reference_p50: number; reference_p95: number; direction: 'above' | 'below'
}

export interface Contribution { feature: string; contribution: number }
export interface ChannelValue { channel: string; value: number }

export interface SignalExcerpt {
  sfreq: number; channels: string[]; t0: number; duration_s: number; data: number[][]; psd_freqs: number[]; psd: number[][]
}

export interface RecordResult {
  record_id: string; p_calibrated: number; likelihood_ratio: number; pretest_prevalence: number
  p_posterior: number; above_threshold: boolean; tier: Tier; preliminary_assessment: string
  findings: Finding[]; contributions: Contribution[]
  band_relative_power: Record<string, number>
  band_reference: Record<string, Record<string, number>>
  topography_z: Record<string, ChannelValue[]>
  coherence_matrix: { band?: string; channels?: string[]; values?: number[][] }
  signal?: SignalExcerpt | null
}

export interface ModelCard {
  model_name: string; version: string; created_at: string; n_features: number; train_prevalence: number
  threshold: number; n_train: Record<string, number>; validation: Record<string, any>
}

export interface Report {
  analysis_id: string; created_at: string; app_version: string; status: string
  recording: { patient_ref?: string | null; patient_id?: number | null; age?: number | null; sex?: string | null; condition?: string | null; medication?: string | null; notes?: string | null }
  quality: { input_kind: string; n_records: number; warnings: string[]; processing_log: Record<string, unknown> }
  model: ModelCard; results: RecordResult[]; figures: Record<string, string>; pdf_url: string | null
  disclaimer: string; limitations: string[]
}

export interface AnalysisStatus { analysis_id: string; status: 'queued' | 'running' | 'done' | 'error'; error?: string | null; report_url?: string | null }
export interface ReportConfig { tiers: Tier[]; pretest_prevalence_default: number; formats: Record<string, string[]>; max_upload_mb: number; conditions: string[] }
export interface Health { status: string; app: string; version: string; model_version: string }

export interface User { id: number; username: string; role: string; full_name: string }
export interface TokenResponse { access_token: string; token_type: string; user: User }

export interface PatientIn { pseudonym: string; age?: number | null; sex?: string | null; medication?: string | null; notes?: string | null }
export interface Patient extends PatientIn { id: number; created_at: string }

export interface AnalysisSummary {
  analysis_id: string; patient_id: number | null; created_at: string; status: string; input_kind: string; filename: string
  pretest_prevalence: number; condition?: string | null; p_calibrated?: number | null; p_posterior?: number | null
  tier_code?: string | null; model_version?: string | null; error?: string | null
}

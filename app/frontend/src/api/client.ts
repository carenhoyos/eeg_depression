import axios from 'axios'
import type { AnalysisStatus, Health, ModelCard, Report, ReportConfig } from '../types'

// Prefijo de la API desde el entorno de build (VITE_API_PREFIX); nunca fijo en el código.
export const API_PREFIX = import.meta.env.VITE_API_PREFIX ?? '/api/v1'
const http = axios.create({ baseURL: API_PREFIX })

export const api = {
  health: () => http.get<Health>('/health').then((r) => r.data),
  model: () => http.get<ModelCard>('/model').then((r) => r.data),
  reportConfig: () => http.get<ReportConfig>('/config/report').then((r) => r.data),
  createAnalysis: (form: FormData) => http.post<AnalysisStatus>('/analyses', form).then((r) => r.data),
  status: (id: string) => http.get<AnalysisStatus>(`/analyses/${id}/status`).then((r) => r.data),
  report: (id: string) => http.get<Report>(`/analyses/${id}`).then((r) => r.data),
  list: () => http.get<{ analysis_id: string; status: string; updated_at: string }[]>('/analyses').then((r) => r.data),
}

export function errorMessage(e: unknown): string {
  if (axios.isAxiosError(e)) return (e.response?.data as { detail?: string })?.detail ?? e.message
  return e instanceof Error ? e.message : String(e)
}

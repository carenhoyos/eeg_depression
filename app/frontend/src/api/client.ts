import axios from 'axios'
import type { AnalysisStatus, AnalysisSummary, Health, ModelCard, Patient, PatientIn, Report, ReportConfig, TokenResponse, User } from '../types'

// Prefijo de la API desde el entorno de build (VITE_API_PREFIX); nunca fijo en el código.
export const API_PREFIX = import.meta.env.VITE_API_PREFIX ?? '/api/v1'
const TOKEN_KEY = 'eegdep.token'

export const tokenStore = {
  get: () => localStorage.getItem(TOKEN_KEY),
  set: (t: string) => localStorage.setItem(TOKEN_KEY, t),
  clear: () => localStorage.removeItem(TOKEN_KEY),
}

const http = axios.create({ baseURL: API_PREFIX })
http.interceptors.request.use((cfg) => {
  const t = tokenStore.get()
  if (t) cfg.headers.Authorization = `Bearer ${t}`
  return cfg
})
http.interceptors.response.use(undefined, (err) => {
  if (axios.isAxiosError(err) && err.response?.status === 401) {
    tokenStore.clear()
    window.dispatchEvent(new Event('eegdep:unauthorized'))
  }
  return Promise.reject(err)
})

/** URL de un recurso protegido (figura, PDF) para usar en <img>/<a>: se obtiene como blob y se expone como object URL. */
export async function fetchBlobUrl(path: string): Promise<string> {
  const r = await http.get(path.startsWith(API_PREFIX) ? path.slice(API_PREFIX.length) : path, { responseType: 'blob' })
  return URL.createObjectURL(r.data)
}

export const api = {
  health: () => http.get<Health>('/health').then((r) => r.data),
  model: () => http.get<ModelCard>('/model').then((r) => r.data),
  reportConfig: () => http.get<ReportConfig>('/config/report').then((r) => r.data),
  login: (username: string, password: string) => http.post<TokenResponse>('/auth/login', { username, password }).then((r) => r.data),
  me: () => http.get<User>('/auth/me').then((r) => r.data),
  patients: () => http.get<Patient[]>('/patients').then((r) => r.data),
  patient: (id: number) => http.get<Patient>(`/patients/${id}`).then((r) => r.data),
  createPatient: (body: PatientIn) => http.post<Patient>('/patients', body).then((r) => r.data),
  updatePatient: (id: number, body: Partial<PatientIn>) => http.patch<Patient>(`/patients/${id}`, body).then((r) => r.data),
  deletePatient: (id: number) => http.delete(`/patients/${id}`),
  patientAnalyses: (id: number) => http.get<AnalysisSummary[]>(`/patients/${id}/analyses`).then((r) => r.data),
  createAnalysis: (form: FormData) => http.post<AnalysisStatus>('/analyses', form).then((r) => r.data),
  status: (id: string) => http.get<AnalysisStatus>(`/analyses/${id}/status`).then((r) => r.data),
  report: (id: string) => http.get<Report>(`/analyses/${id}`).then((r) => r.data),
  analyses: (patientId?: number) => http.get<AnalysisSummary[]>('/analyses', { params: patientId ? { patient_id: patientId } : {} }).then((r) => r.data),
}

export function errorMessage(e: unknown): string {
  if (axios.isAxiosError(e)) return (e.response?.data as { detail?: string })?.detail ?? e.message
  return e instanceof Error ? e.message : String(e)
}

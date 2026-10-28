import { Link, useNavigate, useParams } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, errorMessage } from '../api/client'
import UploadForm from '../components/UploadForm'
import { TrendChart } from '../components/charts'

const pct = (x?: number | null) => (typeof x === 'number' ? `${(100 * x).toFixed(0)} %` : '—')

export default function PatientDetail() {
  const { id = '' } = useParams()
  const pid = Number(id)
  const qc = useQueryClient()
  const nav = useNavigate()
  const patient = useQuery({ queryKey: ['patient', pid], queryFn: () => api.patient(pid) })
  const analyses = useQuery({ queryKey: ['patientAnalyses', pid], queryFn: () => api.patientAnalyses(pid), refetchInterval: 4000 })
  const cfg = useQuery({ queryKey: ['reportConfig'], queryFn: api.reportConfig })
  const del = useMutation({ mutationFn: () => api.deletePatient(pid), onSuccess: () => { qc.invalidateQueries({ queryKey: ['patients'] }); nav('/patients') } })

  if (patient.isLoading) return <section className="card"><p>Cargando…</p></section>
  if (patient.isError || !patient.data) return <section className="card"><p className="error">{errorMessage(patient.error)}</p></section>
  const p = patient.data
  const done = (analyses.data ?? []).filter((a) => a.status === 'done' && typeof a.p_posterior === 'number')
  const tierName = (code?: string | null) => cfg.data?.tiers.find((t) => t.code === code)?.name ?? code ?? '—'

  return (
    <>
      <section className="card">
        <div className="report-head">
          <div>
            <h2>Paciente {p.pseudonym}</h2>
            <p className="muted small">Edad {p.age ?? '—'} · Sexo {p.sex ?? '—'} · Medicación {p.medication ?? '—'} · Registrado {new Date(p.created_at).toLocaleDateString()}</p>
            {p.notes && <p className="small">{p.notes}</p>}
          </div>
          <div className="actions">
            <button className="secondary" onClick={() => { if (confirm(`¿Eliminar al paciente ${p.pseudonym} y todos sus análisis?`)) del.mutate() }}>Eliminar</button>
          </div>
        </div>
        <h4>Evolución de la probabilidad posterior</h4>
        {done.length >= 1 && cfg.data ? <TrendChart items={done} tiers={cfg.data.tiers} /> : <p className="muted small">Sin análisis finalizados todavía.</p>}
        <h4>Análisis</h4>
        {analyses.data?.length ? (
          <table>
            <thead><tr><th>Fecha</th><th>Archivo</th><th>Condición</th><th>Estado</th><th>Prob. posterior</th><th>Nivel</th><th>Modelo</th></tr></thead>
            <tbody>{analyses.data.map((a) => (
              <tr key={a.analysis_id}>
                <td>{new Date(a.created_at).toLocaleString()}</td>
                <td>{a.status === 'done' ? <Link to={`/analyses/${a.analysis_id}`}>{a.filename}</Link> : a.filename}</td>
                <td>{a.condition ?? '—'}</td>
                <td>{a.status}{a.error ? ` · ${a.error}` : ''}</td>
                <td>{pct(a.p_posterior)}</td>
                <td>{a.tier_code ? <span className={`badge tier-badge-${a.tier_code}`}>{tierName(a.tier_code)}</span> : '—'}</td>
                <td>{a.model_version ?? '—'}</td>
              </tr>))}
            </tbody>
          </table>
        ) : <p className="muted small">Sin análisis.</p>}
      </section>
      <UploadForm patientId={pid} onDone={(aid) => nav(`/analyses/${aid}`)} />
    </>
  )
}

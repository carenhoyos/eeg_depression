import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api, API_PREFIX, errorMessage } from '../api/client'
import type { RecordResult, Report } from '../types'
import { BandPowerChart, CoherenceHeatmap, ContributionsChart, ProbabilityBar, TopographyGrid } from './charts'

const pct = (x: number, d = 0) => `${(100 * x).toFixed(d)} %`
const f = (x: unknown, d = 2) => (typeof x === 'number' && !Number.isNaN(x) ? x.toFixed(d) : '?')

function RecordSection({ r, rep }: { r: RecordResult; rep: Report }) {
  const cfg = useQuery({ queryKey: ['reportConfig'], queryFn: api.reportConfig })
  const t = r.tier
  return (
    <article className="result">
      <h3>Registro {r.record_id}</h3>
      <div className={`tier tier-${t.code}`}>
        <div className="tier-name">{t.name}</div>
        <div className="tier-prob">{pct(r.p_posterior, 1)}</div>
        <div className="muted small">
          probabilidad posterior · prevalencia pre-test {pct(r.pretest_prevalence)} · razón de verosimilitud {f(r.likelihood_ratio)} · probabilidad calibrada {pct(r.p_calibrated, 1)}
        </div>
        <p className="impression">{t.text}</p>
      </div>
      {cfg.data && <ProbabilityBar r={r} tiers={cfg.data.tiers} />}

      <h4>Descripción cuantitativa</h4>
      <div className="figs">
        <figure><BandPowerChart r={r} /><figcaption>Potencia relativa por banda frente al grupo de referencia</figcaption></figure>
        <figure><TopographyGrid r={r} /><figcaption>Desviación (z) por canal y banda</figcaption></figure>
        {rep.figures[`topomaps_${r.record_id}`] && (
          <figure className="wide"><img src={rep.figures[`topomaps_${r.record_id}`]} alt="Topomapas interpolados" /><figcaption>Topografía interpolada (misma información que el mapa anterior)</figcaption></figure>
        )}
        <figure><CoherenceHeatmap m={r.coherence_matrix} /><figcaption>Coherencia entre canales</figcaption></figure>
        {r.contributions.length > 0 && (
          <figure><ContributionsChart items={r.contributions} /><figcaption>Características con mayor peso en la decisión del modelo (positivo = hacia depresión)</figcaption></figure>
        )}
      </div>

      <h4>Desviaciones destacadas respecto al grupo de referencia</h4>
      {r.findings.length ? (
        <table>
          <thead><tr><th>Tipo</th><th>Banda</th><th>Localización</th><th>Valor</th><th>z</th><th>Referencia p5–p95</th></tr></thead>
          <tbody>{r.findings.map((d) => (
            <tr key={d.feature}><td>{d.kind}</td><td>{d.band}</td><td>{d.location}</td><td>{d.value.toFixed(3)}</td><td>{d.z.toFixed(2)}</td><td>{d.reference_p5.toFixed(3)}–{d.reference_p95.toFixed(3)}</td></tr>
          ))}</tbody>
        </table>
      ) : <p className="muted">Sin desviaciones destacadas.</p>}
    </article>
  )
}

export default function ReportPage() {
  const { id = '' } = useParams()
  const q = useQuery({ queryKey: ['report', id], queryFn: () => api.report(id), enabled: !!id })
  if (q.isLoading) return <section className="card"><p>Cargando informe…</p></section>
  if (q.isError || !q.data) return <section className="card"><p className="error">No se pudo cargar el informe: {errorMessage(q.error)}</p></section>
  const rep = q.data
  const m = rep.model
  const cv = m.validation?.cv_per_repeat_mean ?? {}
  const ho = m.validation?.holdout ?? {}
  const hoci = m.validation?.holdout_ci95?.auc ?? [NaN, NaN]
  return (
    <section className="card">
      <div className="report-head">
        <div>
          <h2>Informe</h2>
          <p className="muted small">Análisis {rep.analysis_id} · {new Date(rep.created_at).toLocaleString()} · entrada: {rep.quality.input_kind} · registros: {rep.quality.n_records}</p>
        </div>
        <div className="actions">
          {rep.pdf_url && <a className="button" href={rep.pdf_url} target="_blank" rel="noopener">Descargar PDF</a>}
          <a className="button secondary" href={`${API_PREFIX}/analyses/${rep.analysis_id}`} target="_blank" rel="noopener">JSON</a>
        </div>
      </div>
      {rep.quality.warnings.length > 0 && (
        <div className="warn"><strong>Advertencias</strong><ul>{rep.quality.warnings.map((w) => <li key={w}>{w}</li>)}</ul></div>
      )}
      {Object.values(rep.recording).some(Boolean) && (
        <p className="muted small">
          {rep.recording.patient_ref && <>Paciente: {rep.recording.patient_ref} · </>}
          {rep.recording.age && <>Edad: {rep.recording.age} · </>}{rep.recording.sex && <>Sexo: {rep.recording.sex} · </>}
          {rep.recording.condition && <>Condición: {rep.recording.condition} · </>}{rep.recording.medication && <>Medicación: {rep.recording.medication}</>}
        </p>
      )}
      {rep.results.map((r) => <RecordSection key={r.record_id} r={r} rep={rep} />)}

      <h4>Modelo y validación</h4>
      <table><tbody>
        <tr><th>Modelo</th><td>{m.model_name} v{m.version}</td></tr>
        <tr><th>Entrenamiento</th><td>{m.n_train.cases} casos / {m.n_train.controls} controles</td></tr>
        <tr><th>Validación cruzada por sujeto</th><td>AUC {f(cv.auc)} · sensibilidad {f(cv.sensitivity)} · especificidad {f(cv.specificity)} · exactitud balanceada {f(cv.balanced_accuracy)}</td></tr>
        <tr><th>Holdout de sujetos</th><td>AUC {f(ho.auc)} (IC95 {f(hoci[0])}–{f(hoci[1])}) · Brier {f(ho.brier, 3)} · ECE {f(ho.ece, 3)}</td></tr>
      </tbody></table>
      <h4>Limitaciones</h4>
      <ul>{rep.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
      <p className="muted small disclaimer">{rep.disclaimer}</p>
    </section>
  )
}

import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api, errorMessage, fetchBlobUrl } from '../api/client'
import type { RecordResult, Report } from '../types'
import { BandPowerChart, CoherenceHeatmap, ContributionsChart, ProbabilityBar, PsdChart, SignalViewer, TopographyGrid } from './charts'

const pct = (x: number, d = 0) => `${(100 * x).toFixed(d)} %`
const f = (x: unknown, d = 2) => (typeof x === 'number' && !Number.isNaN(x) ? x.toFixed(d) : '?')

/** Imagen protegida por token: se descarga con el cliente autenticado y se muestra como object URL. */
function ProtectedImage({ src, alt }: { src: string; alt: string }) {
  const [url, setUrl] = useState<string>()
  useEffect(() => { let u: string; fetchBlobUrl(src).then((x) => { u = x; setUrl(x) }); return () => { if (u) URL.revokeObjectURL(u) } }, [src])
  return url ? <img src={url} alt={alt} /> : <div className="muted small">Cargando figura…</div>
}

function DownloadPdf({ src, id }: { src: string; id: string }) {
  const [busy, setBusy] = useState(false)
  return (
    <button className="button" disabled={busy} onClick={async () => {
      setBusy(true)
      try {
        const url = await fetchBlobUrl(src)
        const a = document.createElement('a'); a.href = url; a.download = `informe_${id}.pdf`; a.click(); URL.revokeObjectURL(url)
      } finally { setBusy(false) }
    }}>Descargar PDF</button>
  )
}

function RecordSection({ r, rep }: { r: RecordResult; rep: Report }) {
  const cfg = useQuery({ queryKey: ['reportConfig'], queryFn: api.reportConfig })
  const t = r.tier
  return (
    <article className="result">
      {rep.results.length > 1 && <h3>Registro {r.record_id}</h3>}
      <div className={`tier tier-${t.code}`}>
        <div className="tier-name">{t.name}</div>
        <div className="tier-prob">{pct(r.p_posterior, 1)}</div>
        <div className="muted small">
          probabilidad posterior de síntomas depresivos · prevalencia pre-test {pct(r.pretest_prevalence)} · razón de verosimilitud {f(r.likelihood_ratio)} · probabilidad calibrada {pct(r.p_calibrated, 1)}
        </div>
      </div>
      {cfg.data && <ProbabilityBar r={r} tiers={cfg.data.tiers} />}
      <h4>Valoración preliminar (apoyo al psiquiatra)</h4>
      <p className="assessment">{r.preliminary_assessment}</p>

      {r.signal && (
        <>
          <h4>Señal analizada</h4>
          <div className="figs">
            <figure className="wide"><SignalViewer s={r.signal} /><figcaption>Visor de la señal limpia (19 canales 10-20)</figcaption></figure>
            <figure><PsdChart s={r.signal} /><figcaption>Espectro de potencia medio de todos los canales</figcaption></figure>
          </div>
        </>
      )}

      <h4>Descripción cuantitativa</h4>
      <div className="figs">
        <figure><BandPowerChart r={r} /><figcaption>Potencia relativa por banda frente al grupo de referencia</figcaption></figure>
        <figure><TopographyGrid r={r} /><figcaption>Desviación (z) por canal y banda</figcaption></figure>
        {rep.figures[`topomaps_${r.record_id}`] && (
          <figure className="wide"><ProtectedImage src={rep.figures[`topomaps_${r.record_id}`]} alt="Topomapas interpolados" /><figcaption>Topografía interpolada (misma información que el mapa anterior)</figcaption></figure>
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
  const rec = rep.recording
  return (
    <section className="card">
      <div className="report-head">
        <div>
          <h2>Informe de apoyo al tamizaje</h2>
          <p className="muted small">
            {rec.patient_id ? <>Paciente <Link to={`/patients/${rec.patient_id}`}>{rec.patient_ref}</Link> · </> : null}
            Análisis {rep.analysis_id} · {new Date(rep.created_at).toLocaleString()} · entrada: {rep.quality.input_kind}
            {rec.condition ? ` · ${rec.condition}` : ''}{rec.medication ? ` · medicación: ${rec.medication}` : ''}
          </p>
        </div>
        <div className="actions">
          {rep.pdf_url && <DownloadPdf src={rep.pdf_url} id={rep.analysis_id} />}
        </div>
      </div>
      {rep.quality.warnings.length > 0 && (
        <div className="warn"><strong>Advertencias</strong><ul>{rep.quality.warnings.map((w) => <li key={w}>{w}</li>)}</ul></div>
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

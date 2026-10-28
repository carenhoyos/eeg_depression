import { Link, Route, Routes, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from './api/client'
import UploadForm from './components/UploadForm'
import ReportPage from './components/ReportView'

function Header() {
  const health = useQuery({ queryKey: ['health'], queryFn: api.health })
  const model = useQuery({ queryKey: ['model'], queryFn: api.model })
  const cv = model.data?.validation?.cv_per_repeat_mean
  return (
    <header className="header">
      <div>
        <Link to="/" className="brand">{health.data?.app ?? 'EEG · apoyo al tamizaje'}</Link>
        <p className="muted small">
          {model.data
            ? `Modelo ${model.data.model_name} v${model.data.version} · AUC en validación cruzada por sujeto ${cv?.auc?.toFixed(2) ?? '?'} · ${model.data.n_train.cases} casos / ${model.data.n_train.controls} controles`
            : health.isError ? 'API no disponible' : 'Cargando modelo…'}
        </p>
      </div>
      <nav><Link to="/">Nuevo análisis</Link><Link to="/history">Historial</Link></nav>
    </header>
  )
}

function History() {
  const list = useQuery({ queryKey: ['analyses'], queryFn: api.list, refetchInterval: 5000 })
  return (
    <section className="card">
      <h2>Análisis recientes</h2>
      {list.data?.length ? (
        <table><thead><tr><th>ID</th><th>Estado</th><th>Actualizado</th></tr></thead>
          <tbody>{list.data.map((a) => (
            <tr key={a.analysis_id}>
              <td>{a.status === 'done' ? <Link to={`/analyses/${a.analysis_id}`}>{a.analysis_id}</Link> : a.analysis_id}</td>
              <td>{a.status}</td><td>{a.updated_at ? new Date(a.updated_at).toLocaleString() : '-'}</td>
            </tr>))}
          </tbody></table>
      ) : <p className="muted">Sin análisis todavía.</p>}
    </section>
  )
}

export default function App() {
  const navigate = useNavigate()
  return (
    <div className="layout">
      <Header />
      <main>
        <Routes>
          <Route path="/" element={<UploadForm onDone={(id) => navigate(`/analyses/${id}`)} />} />
          <Route path="/history" element={<History />} />
          <Route path="/analyses/:id" element={<ReportPage />} />
        </Routes>
      </main>
    </div>
  )
}

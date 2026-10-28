import { Link, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from './api/client'
import { useAuth } from './auth'
import Login from './pages/Login'
import Patients from './pages/Patients'
import PatientDetail from './pages/PatientDetail'
import NewAnalysis from './pages/NewAnalysis'
import ReportPage from './components/ReportView'

function Header() {
  const { user, logout } = useAuth()
  const health = useQuery({ queryKey: ['health'], queryFn: api.health })
  const model = useQuery({ queryKey: ['model'], queryFn: api.model, enabled: !!user })
  const cv = model.data?.validation?.cv_per_repeat_mean
  return (
    <header className="header">
      <div>
        <Link to="/" className="brand">{health.data?.app ?? 'EEG · apoyo al tamizaje'}</Link>
        <p className="muted small">
          {model.data
            ? `Modelo ${model.data.model_name} v${model.data.version} · AUC por sujeto ${cv?.auc?.toFixed(2) ?? '?'} · ${model.data.n_train.cases} casos / ${model.data.n_train.controls} controles`
            : health.isError ? 'API no disponible' : 'Apoyo a la decisión clínica, no diagnóstico'}
        </p>
      </div>
      {user && (
        <nav>
          <Link to="/patients">Pacientes</Link>
          <Link to="/analyses/new">Nuevo análisis</Link>
          <span className="muted small">{user.full_name || user.username} · {user.role}</span>
          <button className="link" onClick={logout}>Salir</button>
        </nav>
      )}
    </header>
  )
}

function RequireAuth({ children }: { children: React.ReactElement }) {
  const { user, loading } = useAuth()
  const loc = useLocation()
  if (loading) return <section className="card"><p>Cargando…</p></section>
  if (!user) return <Navigate to="/login" state={{ from: loc.pathname }} replace />
  return children
}

export default function App() {
  return (
    <div className="layout">
      <Header />
      <main>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/" element={<Navigate to="/patients" replace />} />
          <Route path="/patients" element={<RequireAuth><Patients /></RequireAuth>} />
          <Route path="/patients/:id" element={<RequireAuth><PatientDetail /></RequireAuth>} />
          <Route path="/analyses/new" element={<RequireAuth><NewAnalysis /></RequireAuth>} />
          <Route path="/analyses/:id" element={<RequireAuth><ReportPage /></RequireAuth>} />
        </Routes>
      </main>
    </div>
  )
}

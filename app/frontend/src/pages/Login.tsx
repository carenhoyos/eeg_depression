import { useState, type FormEvent } from 'react'
import { Navigate, useLocation, useNavigate } from 'react-router-dom'
import { errorMessage } from '../api/client'
import { useAuth } from '../auth'

export default function Login() {
  const { user, login } = useAuth()
  const nav = useNavigate()
  const loc = useLocation() as { state?: { from?: string } }
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  if (user) return <Navigate to={loc.state?.from ?? '/patients'} replace />

  const submit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    const fd = new FormData(e.currentTarget)
    setBusy(true); setError('')
    try {
      await login(String(fd.get('username')), String(fd.get('password')))
      nav(loc.state?.from ?? '/patients', { replace: true })
    } catch (err) {
      setError(errorMessage(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="card login">
      <h2>Acceso para profesionales</h2>
      <p className="muted small">Herramienta de apoyo al tamizaje de depresión a partir de EEG. Uso exclusivo del profesional de salud tratante.</p>
      <form onSubmit={submit}>
        <label>Usuario<input name="username" autoComplete="username" required /></label>
        <label>Contraseña<input name="password" type="password" autoComplete="current-password" required /></label>
        <button type="submit" disabled={busy}>Entrar</button>
        {error && <p className="error">{error}</p>}
      </form>
    </section>
  )
}

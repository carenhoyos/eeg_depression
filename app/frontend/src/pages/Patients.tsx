import { useState, type FormEvent } from 'react'
import { Link } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, errorMessage } from '../api/client'
import type { PatientIn } from '../types'

export default function Patients() {
  const qc = useQueryClient()
  const patients = useQuery({ queryKey: ['patients'], queryFn: api.patients })
  const [error, setError] = useState('')
  const create = useMutation({
    mutationFn: (body: PatientIn) => api.createPatient(body),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['patients'] }); setError('') },
    onError: (e) => setError(errorMessage(e)),
  })

  const submit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    const fd = new FormData(e.currentTarget)
    const body: PatientIn = {
      pseudonym: String(fd.get('pseudonym')).trim(),
      age: fd.get('age') ? Number(fd.get('age')) : null,
      sex: String(fd.get('sex') || '') || null,
      medication: String(fd.get('medication') || '') || null,
      notes: String(fd.get('notes') || '') || null,
    }
    create.mutate(body)
    e.currentTarget.reset()
  }

  return (
    <>
      <section className="card">
        <h2>Pacientes</h2>
        <p className="muted small">Identifica a cada paciente solo por un pseudónimo. No registres nombres, documentos ni datos de contacto.</p>
        {patients.isLoading && <p>Cargando…</p>}
        {patients.data?.length ? (
          <table>
            <thead><tr><th>Pseudónimo</th><th>Edad</th><th>Sexo</th><th>Medicación</th><th>Creado</th></tr></thead>
            <tbody>{patients.data.map((p) => (
              <tr key={p.id}>
                <td><Link to={`/patients/${p.id}`}>{p.pseudonym}</Link></td>
                <td>{p.age ?? '—'}</td><td>{p.sex ?? '—'}</td><td>{p.medication ?? '—'}</td>
                <td>{new Date(p.created_at).toLocaleDateString()}</td>
              </tr>))}
            </tbody>
          </table>
        ) : !patients.isLoading && <p className="muted">Aún no hay pacientes registrados.</p>}
      </section>
      <section className="card">
        <h2>Nuevo paciente</h2>
        <form onSubmit={submit}>
          <div className="grid">
            <label>Pseudónimo *<input name="pseudonym" required maxLength={64} placeholder="p. ej. PX-0421" /></label>
            <label>Edad<input name="age" type="number" min={0} max={120} /></label>
            <label>Sexo<select name="sex" defaultValue=""><option value="">—</option><option>F</option><option>M</option><option>Otro</option></select></label>
            <label>Medicación actual<input name="medication" maxLength={200} /></label>
            <label>Notas clínicas<input name="notes" maxLength={2000} /></label>
          </div>
          <button type="submit" disabled={create.isPending}>Registrar paciente</button>
          {error && <p className="error">{error}</p>}
        </form>
      </section>
    </>
  )
}

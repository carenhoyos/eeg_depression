import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import UploadForm from '../components/UploadForm'

export default function NewAnalysis() {
  const nav = useNavigate()
  const patients = useQuery({ queryKey: ['patients'], queryFn: api.patients })
  const [patientId, setPatientId] = useState<number | undefined>(undefined)
  return (
    <>
      <section className="card">
        <h2>Nuevo análisis</h2>
        <label>Paciente
          <select value={patientId ?? ''} onChange={(e) => setPatientId(e.target.value ? Number(e.target.value) : undefined)}>
            <option value="">— sin asociar (análisis suelto) —</option>
            {patients.data?.map((p) => <option key={p.id} value={p.id}>{p.pseudonym}</option>)}
          </select>
          <small className="muted">Asociar el análisis a un paciente permite ver su evolución. <Link to="/patients">Registrar paciente</Link>.</small>
        </label>
      </section>
      <UploadForm patientId={patientId} onDone={(aid) => nav(`/analyses/${aid}`)} />
    </>
  )
}

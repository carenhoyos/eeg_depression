import { useEffect, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { api, errorMessage } from '../api/client'

interface Props { onDone: (analysisId: string) => void }

export default function UploadForm({ onDone }: Props) {
  const cfg = useQuery({ queryKey: ['reportConfig'], queryFn: api.reportConfig })
  const [status, setStatus] = useState<string>('')
  const [pretestPct, setPretestPct] = useState<number | ''>('')
  useEffect(() => { if (cfg.data && pretestPct === '') setPretestPct(Math.round(100 * cfg.data.pretest_prevalence_default)) }, [cfg.data, pretestPct])

  const submit = useMutation({
    mutationFn: async (form: FormData) => {
      setStatus('Enviando archivo…')
      const { analysis_id } = await api.createAnalysis(form)
      for (;;) {
        const s = await api.status(analysis_id)
        setStatus(`Estado: ${s.status}`)
        if (s.status === 'done') return analysis_id
        if (s.status === 'error') throw new Error(s.error ?? 'fallo en el análisis')
        await new Promise((r) => setTimeout(r, 1500))
      }
    },
    onSuccess: onDone,
    onError: (e) => setStatus(`Error: ${errorMessage(e)}`),
  })

  const exts = cfg.data ? Object.values(cfg.data.formats).flat() : []

  return (
    <section className="card">
      <h2>Nuevo análisis</h2>
      <form onSubmit={(e) => {
        e.preventDefault()
        const fd = new FormData(e.currentTarget)
        if (pretestPct !== '') fd.set('pretest_prevalence', String(Number(pretestPct) / 100))
        for (const [k, v] of Array.from(fd.entries())) if (v === '') fd.delete(k)
        submit.mutate(fd)
      }}>
        <label>Archivo EEG <span className="muted small">({exts.join(', ')}; máx. {cfg.data?.max_upload_mb ?? '?'} MB)</span>
          <input type="file" name="file" accept={exts.join(',')} required />
        </label>
        <div className="grid">
          <label>Prevalencia pre-test (%)
            <input type="number" min={1} max={99} value={pretestPct} onChange={(e) => setPretestPct(e.target.value === '' ? '' : Number(e.target.value))} />
            <small className="muted">Proporción esperada de depresión en tu contexto clínico; cambia la probabilidad posterior.</small>
          </label>
          <label>Frecuencia de muestreo (Hz, solo CSV crudo)<input type="number" name="sampling_rate_hz" min={1} step={0.1} /></label>
          <label>Referencia del paciente (pseudónimo)<input type="text" name="patient_ref" maxLength={40} /></label>
          <label>Edad<input type="number" name="age" min={0} max={120} /></label>
          <label>Sexo<select name="sex" defaultValue=""><option value="">—</option><option>F</option><option>M</option><option>Otro</option></select></label>
          <label>Condición del registro<select name="condition" defaultValue=""><option value="">—</option><option value="eyes_closed">Ojos cerrados</option><option value="eyes_open">Ojos abiertos</option></select></label>
          <label>Medicación actual<input type="text" name="medication" maxLength={120} /></label>
          <label>Notas<input type="text" name="notes" maxLength={200} /></label>
        </div>
        <button type="submit" disabled={submit.isPending || !cfg.data}>Analizar</button>
        <p className={`status ${submit.isError ? 'error' : ''}`} aria-live="polite">{status}</p>
      </form>
    </section>
  )
}

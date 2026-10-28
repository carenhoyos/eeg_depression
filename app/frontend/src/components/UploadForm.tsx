import { useEffect, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { api, errorMessage } from '../api/client'

interface Props { patientId?: number; onDone: (analysisId: string) => void }

const CONDITION_LABEL: Record<string, string> = { eyes_closed: 'Ojos cerrados', eyes_open: 'Ojos abiertos' }

export default function UploadForm({ patientId, onDone }: Props) {
  const cfg = useQuery({ queryKey: ['reportConfig'], queryFn: api.reportConfig })
  const [status, setStatus] = useState('')
  const [pretestPct, setPretestPct] = useState<number | ''>('')
  useEffect(() => { if (cfg.data && pretestPct === '') setPretestPct(Math.round(100 * cfg.data.pretest_prevalence_default)) }, [cfg.data, pretestPct])

  const submit = useMutation({
    mutationFn: async (form: FormData) => {
      setStatus('Enviando archivo…')
      const { analysis_id } = await api.createAnalysis(form)
      for (;;) {
        const s = await api.status(analysis_id)
        setStatus(s.status === 'running' ? 'Procesando señal (filtrado, ICA, características, modelo)…' : `Estado: ${s.status}`)
        if (s.status === 'done') return analysis_id
        if (s.status === 'error') throw new Error(s.error ?? 'fallo en el análisis')
        await new Promise((r) => setTimeout(r, 2000))
      }
    },
    onSuccess: onDone,
    onError: (e) => setStatus(`Error: ${errorMessage(e)}`),
  })

  const exts = cfg.data ? Object.values(cfg.data.formats).flat() : []

  return (
    <section className="card">
      <h2>Señal de EEG</h2>
      <p className="muted small">
        Entrada recomendada: registro de EEG en reposo (EDF/BDF, EEGLAB .set + .fdt, o FIF), 19 canales 10-20 o más, al menos un minuto.
        También se acepta un CSV de características con la estructura de entrenamiento.
      </p>
      <form onSubmit={(e) => {
        e.preventDefault()
        const form = e.currentTarget
        const fd = new FormData()
        const input = form.elements.namedItem('files') as HTMLInputElement
        Array.from(input.files ?? []).forEach((f) => fd.append('files', f))
        if (patientId) fd.set('patient_id', String(patientId))
        if (pretestPct !== '') fd.set('pretest_prevalence', String(Number(pretestPct) / 100))
        for (const k of ['sampling_rate_hz', 'condition', 'medication', 'notes']) {
          const v = (form.elements.namedItem(k) as HTMLInputElement | HTMLSelectElement | null)?.value
          if (v) fd.set(k, v)
        }
        submit.mutate(fd)
      }}>
        <label>Archivo(s) <span className="muted small">({exts.join(', ')}; máx. {cfg.data?.max_upload_mb ?? '?'} MB). Para EEGLAB selecciona .set y .fdt a la vez.</span>
          <input type="file" name="files" accept={[...exts, '.fdt'].join(',')} multiple required />
        </label>
        <div className="grid">
          <label>Condición del registro
            <select name="condition" defaultValue="">
              <option value="">—</option>
              {cfg.data?.conditions.map((c) => <option key={c} value={c}>{CONDITION_LABEL[c] ?? c}</option>)}
            </select>
          </label>
          <label>Prevalencia pre-test (%)
            <input type="number" min={1} max={99} value={pretestPct} onChange={(e) => setPretestPct(e.target.value === '' ? '' : Number(e.target.value))} />
            <small className="muted">Proporción esperada de depresión en tu contexto clínico; cambia la probabilidad posterior.</small>
          </label>
          <label>Frecuencia de muestreo (Hz, solo CSV crudo)<input type="number" name="sampling_rate_hz" min={1} step={0.1} /></label>
          <label>Medicación en el momento del registro<input type="text" name="medication" maxLength={120} /></label>
          <label>Notas<input type="text" name="notes" maxLength={200} /></label>
        </div>
        <button type="submit" disabled={submit.isPending || !cfg.data}>Analizar</button>
        <p className={`status ${submit.isError ? 'error' : ''}`} aria-live="polite">{status}</p>
      </form>
    </section>
  )
}

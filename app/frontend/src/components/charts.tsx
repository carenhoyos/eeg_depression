// Gráficos interactivos construidos desde el JSON de la API (Recharts + SVG). Un solo eje por gráfico;
// color por entidad; leyenda cuando hay más de una serie; sin valores en cada punto.
import { useMemo, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Cell, ComposedChart, Legend, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis,
} from 'recharts'
import type { AnalysisSummary, Contribution, RecordResult, SignalExcerpt, Tier } from '../types'

const fmtVal = (v: unknown, d = 3): string => (Array.isArray(v) ? `${Number(v[0]).toFixed(d)} – ${Number(v[1]).toFixed(d)}` : typeof v === 'number' ? v.toFixed(d) : String(v ?? ''))
const css = (v: string) => getComputedStyle(document.documentElement).getPropertyValue(v).trim() || undefined
const tierVar = (code: string) => css(`--${code}`)

export function ProbabilityBar({ r, tiers }: { r: RecordResult; tiers: Tier[] }) {
  const p = 100 * r.p_posterior
  return (
    <div className="prob-bar" role="img" aria-label={`Probabilidad posterior ${p.toFixed(0)} %`}>
      <div className="prob-track">
        {tiers.map((t) => (
          <div key={t.code} className={`prob-seg tier-bg-${t.code}`} style={{ width: `${100 * (t.max - t.min)}%` }} title={`${t.name}: ${100 * t.min}–${100 * t.max} %`}>
            <span>{t.name}</span>
          </div>
        ))}
        <div className="prob-marker" style={{ left: `${p}%` }} />
      </div>
      <div className="prob-axis"><span>0 %</span><span>50 %</span><span>100 %</span></div>
    </div>
  )
}

export function BandPowerChart({ r }: { r: RecordResult }) {
  const data = Object.keys(r.band_relative_power).map((b) => {
    const ref = r.band_reference[b]
    return { band: b, iqr: [ref.p25, ref.p75], range: [ref.p5, ref.p95], patient: r.band_relative_power[b] }
  })
  return (
    <ResponsiveContainer width="100%" height={260}>
      <ComposedChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={css('--line')} vertical={false} />
        <XAxis dataKey="band" tickLine={false} axisLine={false} />
        <YAxis tickLine={false} axisLine={false} width={44} tickFormatter={(v) => v.toFixed(2)} />
        <Tooltip formatter={(v) => fmtVal(v)} />
        <Legend />
        <Bar dataKey="range" name="Referencia p5–p95" fill={css('--ref')} fillOpacity={0.25} radius={4} />
        <Bar dataKey="iqr" name="Referencia p25–p75" fill={css('--ref')} fillOpacity={0.55} radius={4} />
        <Scatter dataKey="patient" name="Paciente" fill={css('--primary')} r={7} />
      </ComposedChart>
    </ResponsiveContainer>
  )
}

export function ContributionsChart({ items }: { items: Contribution[] }) {
  const data = [...items].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
  return (
    <ResponsiveContainer width="100%" height={28 * data.length + 40}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 0 }}>
        <CartesianGrid stroke={css('--line')} horizontal={false} />
        <XAxis type="number" tickLine={false} axisLine={false} />
        <YAxis type="category" dataKey="feature" width={190} tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
        <Tooltip formatter={(v) => fmtVal(v)} />
        <ReferenceLine x={0} stroke={css('--ref')} />
        <Bar dataKey="contribution" name="Contribución al logit" radius={4}>
          {data.map((d) => <Cell key={d.feature} fill={d.contribution > 0 ? css('--elevated') : css('--primary')} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

const POS: Record<string, [number, number]> = {
  Fp1: [-0.3, 0.9], Fp2: [0.3, 0.9], F7: [-0.75, 0.55], F3: [-0.4, 0.5], Fz: [0, 0.5], F4: [0.4, 0.5], F8: [0.75, 0.55],
  T7: [-0.9, 0], C3: [-0.45, 0], Cz: [0, 0], C4: [0.45, 0], T8: [0.9, 0],
  P7: [-0.75, -0.55], P3: [-0.4, -0.5], Pz: [0, -0.5], P4: [0.4, -0.5], P8: [0.75, -0.55], O1: [-0.3, -0.9], O2: [0.3, -0.9],
}

export function TopographyGrid({ r, zMax = 3 }: { r: RecordResult; zMax?: number }) {
  const bands = Object.keys(r.topography_z)
  const color = (z: number) => {
    const t = Math.max(-1, Math.min(1, z / zMax))
    const a = Math.abs(t) * 0.85
    return t >= 0 ? `rgba(209,73,91,${a})` : `rgba(42,111,219,${a})`
  }
  return (
    <div className="topo-grid">
      {bands.map((b) => (
        <figure key={b} className="topo">
          <svg viewBox="-1.15 -1.15 2.3 2.3" role="img" aria-label={`z por canal, banda ${b}`}>
            <circle cx={0} cy={0} r={1.05} fill="none" stroke="currentColor" strokeWidth={0.02} />
            <path d="M -0.12 -1.05 L 0 -1.18 L 0.12 -1.05" fill="none" stroke="currentColor" strokeWidth={0.02} />
            {r.topography_z[b].map(({ channel, value }) => {
              const [x, y] = POS[channel] ?? [0, 0]
              return (
                <g key={channel}>
                  <circle cx={x} cy={-y} r={0.13} fill={color(value)} stroke="currentColor" strokeWidth={0.01}><title>{`${channel}: z = ${value.toFixed(2)}`}</title></circle>
                  <text x={x} y={-y + 0.03} fontSize={0.09} textAnchor="middle" fill="currentColor">{channel}</text>
                </g>
              )
            })}
          </svg>
          <figcaption>{b}</figcaption>
        </figure>
      ))}
      <p className="muted small">Rojo: por encima del grupo de referencia; azul: por debajo (|z| hasta {zMax}).</p>
    </div>
  )
}

export function CoherenceHeatmap({ m }: { m: RecordResult['coherence_matrix'] }) {
  if (!m.channels || !m.values) return null
  const n = m.channels.length
  const cell = 100 / n
  return (
    <div className="heatmap">
      <svg viewBox={`0 0 ${100 + 8} ${100 + 8}`} role="img" aria-label={`Coherencia ${m.band}`}>
        {m.values.map((row, i) => row.map((v, j) => (
          <rect key={`${i}-${j}`} x={8 + j * cell} y={8 + i * cell} width={cell} height={cell} fill={`rgba(42,111,219,${Math.min(1, v / 100)})`}>
            <title>{`${m.channels![i]} – ${m.channels![j]}: ${v.toFixed(1)} %`}</title>
          </rect>
        )))}
        {m.channels.map((c, i) => (
          <text key={c} x={8 + i * cell + cell / 2} y={6} fontSize={2.6} textAnchor="middle" fill="currentColor" transform={`rotate(-90 ${8 + i * cell + cell / 2} 6)`}>{c}</text>
        ))}
        {m.channels.map((c, i) => (
          <text key={`y-${c}`} x={6} y={8 + i * cell + cell / 2 + 1} fontSize={2.6} textAnchor="end" fill="currentColor">{c}</text>
        ))}
      </svg>
      <p className="muted small">Coherencia {m.band} entre pares de canales (0–100 %).</p>
    </div>
  )
}

/** Visor de señal: trazos apilados del fragmento limpio, con selección de canales y ganancia. */
export function SignalViewer({ s }: { s: SignalExcerpt }) {
  const [gain, setGain] = useState(1)
  const [selected, setSelected] = useState<string[]>(s.channels)
  const spacing = 100
  const n = s.data[0]?.length ?? 0
  const w = 1000, h = Math.max(200, selected.length * 26)
  const paths = useMemo(() => selected.map((ch, i) => {
    const row = s.data[s.channels.indexOf(ch)]
    const y0 = 14 + i * (h - 28) / Math.max(1, selected.length - 1)
    const pts = row.map((v, k) => `${(k / Math.max(1, n - 1)) * w},${y0 - (v * gain) / spacing * 24}`)
    return { ch, y0, d: `M${pts.join('L')}` }
  }), [s, selected, gain, h, n])
  return (
    <div className="signal-viewer">
      <div className="signal-controls">
        <label className="inline">Ganancia
          <input type="range" min={0.25} max={4} step={0.25} value={gain} onChange={(e) => setGain(Number(e.target.value))} /> ×{gain}
        </label>
        <button type="button" className="link" onClick={() => setSelected(s.channels)}>todos</button>
        <button type="button" className="link" onClick={() => setSelected(s.channels.filter((c) => /^(F|Fp)/.test(c)))}>frontales</button>
        <button type="button" className="link" onClick={() => setSelected(s.channels.filter((c) => /^(P|O)/.test(c)))}>posteriores</button>
      </div>
      <svg viewBox={`0 0 ${w + 50} ${h}`} role="img" aria-label="Señal EEG limpia" className="signal-svg">
        {paths.map(({ ch, y0, d }) => (
          <g key={ch}>
            <text x={0} y={y0 + 3} fontSize={10} fill="currentColor">{ch}</text>
            <path d={d} transform="translate(45,0)" fill="none" stroke={css('--primary')} strokeWidth={0.8} />
          </g>
        ))}
      </svg>
      <p className="muted small">Fragmento de {s.duration_s.toFixed(0)} s desde t = {s.t0.toFixed(0)} s tras filtrado, referencia promedio e ICA. Escala: {spacing} µV entre trazos (× ganancia).</p>
    </div>
  )
}

export function PsdChart({ s }: { s: SignalExcerpt }) {
  const mean = s.psd_freqs.map((f, i) => ({ f, mean: s.psd.reduce((a, row) => a + row[i], 0) / s.psd.length }))
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={mean} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={css('--line')} vertical={false} />
        <XAxis dataKey="f" tickLine={false} axisLine={false} label={{ value: 'Hz', position: 'insideRight', offset: -4 }} />
        <YAxis tickLine={false} axisLine={false} width={44} />
        <Tooltip formatter={(v) => fmtVal(v, 2)} labelFormatter={(l) => `${Number(l).toFixed(1)} Hz`} />
        <ReferenceArea x1={8} x2={12} fill={css('--ref')} fillOpacity={0.12} label={{ value: 'alfa', fontSize: 10 }} />
        <Line type="monotone" dataKey="mean" name="log10 PSD media (µV²/Hz)" stroke={css('--primary')} dot={false} strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  )
}

export function TrendChart({ items, tiers }: { items: AnalysisSummary[]; tiers: Tier[] }) {
  const data = [...items].sort((a, b) => a.created_at.localeCompare(b.created_at)).map((a) => ({
    date: new Date(a.created_at).toLocaleDateString(), p: 100 * (a.p_posterior ?? 0), id: a.analysis_id,
  }))
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={css('--line')} vertical={false} />
        {tiers.map((t) => <ReferenceArea key={t.code} y1={100 * t.min} y2={100 * t.max} fill={tierVar(t.code)} fillOpacity={0.08} />)}
        <XAxis dataKey="date" tickLine={false} axisLine={false} />
        <YAxis domain={[0, 100]} tickLine={false} axisLine={false} width={40} unit=" %" />
        <Tooltip formatter={(v) => `${fmtVal(v, 0)} %`} />
        <Line type="monotone" dataKey="p" name="Probabilidad posterior" stroke={css('--primary')} strokeWidth={2} dot={{ r: 5 }} />
      </LineChart>
    </ResponsiveContainer>
  )
}

// Gráficos interactivos construidos desde el JSON de la API (Recharts). Un solo eje por gráfico;
// color por entidad, no por rango; leyenda cuando hay más de una serie; sin valores en cada punto.
import {
  Bar, BarChart, CartesianGrid, Cell, ComposedChart, Legend, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis,
} from 'recharts'
import type { Contribution, RecordResult, Tier } from '../types'

const fmtVal = (v: unknown, d = 3): string => (Array.isArray(v) ? `${Number(v[0]).toFixed(d)} – ${Number(v[1]).toFixed(d)}` : typeof v === 'number' ? v.toFixed(d) : String(v ?? ''))
const css = (v: string) => getComputedStyle(document.documentElement).getPropertyValue(v).trim() || undefined

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
    return { band: b, p5: ref.p5, iqr: [ref.p25, ref.p75], range: [ref.p5, ref.p95], patient: r.band_relative_power[b], p50: ref.p50 }
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

// Posiciones 2D aproximadas (10-20) para un mapa esquemático interactivo; la topografía interpolada viene del PNG.
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
                  <circle cx={x} cy={-y} r={0.13} fill={color(value)} stroke="currentColor" strokeWidth={0.01}>
                    <title>{`${channel}: z = ${value.toFixed(2)}`}</title>
                  </circle>
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

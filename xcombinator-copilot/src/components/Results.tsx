import results from '../data/results.json'

// Comparison-story view: real eval metrics from extras/results (regenerate with `npm run build:results`).
// Base = n-gram classical baseline; "best" = the best each task achieves across our fine-tuned models.

const R = results as any
const pc = (x: number | null | undefined) => (x == null ? '—' : (x * 100).toFixed(1) + '%')

const C = {
  ns: '#0f9d6b', // next-step (green)
  cp: '#3b82f6', // completion (blue)
  an: '#f59e0b', // anomaly (amber)
  base: '#94a3b8', // baseline (grey)
}

function Card({ title, metric, base, best }: { title: string; metric: string; base: number; best: number }) {
  const win = best >= base
  const w = (v: number) => Math.max(3, Math.min(100, v * 100))
  return (
    <div className="rs-card">
      <div className="rs-card-t">{title}</div>
      <div className="rs-card-m">{metric}</div>
      <div className="rs-row"><span>Baseline · n-gram</span><b>{pc(base)}</b></div>
      <div className="rs-bar"><i style={{ width: w(base) + '%', background: C.base }} /></div>
      <div className="rs-row">
        <span>Best · fine-tuned</span>
        <span><b>{pc(best)}</b> <em className={'rs-pill ' + (win ? 'win' : 'lose')}>{win ? 'win' : 'trails'}</em></span>
      </div>
      <div className="rs-bar"><i style={{ width: w(best) + '%', background: win ? C.ns : '#ef4444' }} /></div>
    </div>
  )
}

type Series = { name: string; color: string; dashed?: boolean; vals: (number | null)[] }
function LineChart({ labels, series }: { labels: string[]; series: Series[] }) {
  const W = 520, H = 230, padL = 40, padR = 16, padT = 14, padB = 36
  const iw = W - padL - padR, ih = H - padT - padB
  const x = (i: number) => (labels.length < 2 ? padL + iw / 2 : padL + (iw * i) / (labels.length - 1))
  const y = (v: number) => padT + ih * (1 - Math.max(0, Math.min(1, v)))
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img">
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <line x1={padL} x2={W - padR} y1={y(g)} y2={y(g)} stroke="#e2e8f0" strokeWidth={1} />
          <text x={6} y={y(g) + 3} fontSize={11} fill="#94a3b8">{Math.round(g * 100)}%</text>
        </g>
      ))}
      {labels.map((l, i) => (
        <text key={l + i} x={x(i)} y={H - 12} fontSize={11} fill="#64748b" textAnchor="middle">{l}</text>
      ))}
      {series.map((s) => {
        const pts = s.vals.map((v, i) => (v == null ? null : [x(i), y(v)] as [number, number])).filter(Boolean) as [number, number][]
        return (
          <g key={s.name}>
            {pts.length > 1 && (
              <polyline points={pts.map((p) => p.join(',')).join(' ')} fill="none" stroke={s.color}
                strokeWidth={s.dashed ? 1.5 : 2.5} strokeDasharray={s.dashed ? '6 5' : undefined} />
            )}
            {!s.dashed && pts.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={3.5} fill={s.color} />)}
          </g>
        )
      })}
    </svg>
  )
}

function Legend({ series }: { series: Series[] }) {
  return (
    <div className="rs-legend">
      {series.map((s) => (
        <span key={s.name}><i style={{ background: s.color, opacity: s.dashed ? 0.6 : 1 }} />{s.name}</span>
      ))}
    </div>
  )
}

export default function Results() {
  const scalingSeries: Series[] = [
    { name: 'Next-step top-1', color: C.ns, vals: R.scaling.map((p: any) => p.nextstepTop1) },
    { name: 'Completion block-acc', color: C.cp, vals: R.scaling.map((p: any) => p.completionBlockAcc) },
    { name: 'n-gram next-step', color: C.base, dashed: true, vals: R.scaling.map(() => R.nextstep.baseline) },
    { name: 'n-gram completion', color: '#a78bfa', dashed: true, vals: R.scaling.map(() => R.completion.baseline) },
  ]
  const sizeSeries: Series[] = [
    { name: 'Next-step top-1', color: C.ns, vals: R.modelSize.map((p: any) => p.nextstepTop1) },
    { name: 'Completion block-acc', color: C.cp, vals: R.modelSize.map((p: any) => p.completionBlockAcc) },
    { name: 'Anomaly F1', color: C.an, vals: R.modelSize.map((p: any) => p.anomalyF1) },
  ]
  return (
    <div className="rs-wrap">
      <div className="rs-head">
        <div className="rs-eyebrow">Infineon Industrial-AI · real eval metrics from extras/results</div>
        <h1>Teaching a small model fab <em>process logic</em></h1>
        <p className="rs-lede">Qwen2.5-1.5B fine-tuned on fab routes, vs a classical n-gram baseline and the frozen base model. MOSFET labeled eval, n=200.</p>
      </div>

      <h2 className="rs-h2">Headline — best fine-tuned (per task) vs n-gram baseline</h2>
      <div className="rs-cards">
        <Card title="Next-step" metric="Top-1 accuracy" base={R.nextstep.baseline} best={R.nextstep.finetuned} />
        <Card title="Completion" metric="Block accuracy" base={R.completion.baseline} best={R.completion.finetuned} />
        <Card title="Anomaly" metric="F1 (rule violation)" base={R.anomaly.baseline} best={R.anomaly.finetuned} />
      </div>

      <h2 className="rs-h2">Does more training data help? (1-epoch data-scaling)</h2>
      <div className="rs-panel">
        <p className="rs-sub">Each point is a fine-tune on N routes/family. Completion <b>crosses the baseline</b> with enough data.</p>
        <LineChart labels={R.scaling.map((p: any) => String(p.size))} series={scalingSeries} />
        <Legend series={scalingSeries} />
      </div>

      <h2 className="rs-h2">Does model size help? (0.5B → 3B, full data)</h2>
      <div className="rs-panel">
        <p className="rs-sub">Same JSON corpus + ~3 epochs; only the base model size changes. Bigger doesn't help — <b>data matters more than capacity</b>.</p>
        <LineChart labels={R.modelSize.map((p: any) => p.label)} series={sizeSeries} />
        <Legend series={sizeSeries} />
      </div>

      <h2 className="rs-h2">Generalizes across all three families</h2>
      <div className="rs-panel rs-fam">
        <div className="rs-frow rs-fhead"><span>Family</span><span>Next-step</span><span>Completion</span><span>Anomaly F1</span></div>
        {R.perFamily.map((f: any) => (
          <div className="rs-frow" key={f.family}>
            <b>{f.family}</b><span>{pc(f.nextstepTop1)}</span><span>{pc(f.completionBlockAcc)}</span><span>{pc(f.anomalyF1)}</span>
          </div>
        ))}
        <p className="rs-sub" style={{ marginTop: 10 }}>One all-family model. MOSFET strongest, IGBT hardest; anomaly transfers to IGBT — not a MOSFET-only model.</p>
      </div>

      <h2 className="rs-h2">Takeaways</h2>
      <div className="rs-panel">
        <ul className="rs-take">
          <li>With enough data the fine-tuned LLM <b>beats the strong n-gram</b> on completion ({pc(R.completion.finetuned)} vs {pc(R.completion.baseline)}).</li>
          <li>Fine-tuning teaches <b>rule-violation detection from scratch</b> (anomaly F1 0 → {pc(R.anomaly.finetuned)}); the frozen base scores 0.</li>
          <li>The n-gram is a strong classical baseline on next-step; the LLM's edge is completion coherence + one promptable, reasoning model across all tasks.</li>
          <li>All numbers are real eval metrics — no placeholders. The live copilot runs this model on-device.</li>
        </ul>
      </div>
      <div className="rs-src">Generated {R.generatedAt} · best={R.bestName} · baseline={R.baselineName}</div>
    </div>
  )
}

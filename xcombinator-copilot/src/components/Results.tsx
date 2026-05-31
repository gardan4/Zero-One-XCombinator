import results from '../data/results.json'

// Comparison-story view: real eval metrics from extras/results (regenerate with `npm run build:results`).
// Headline = DeepSeek zero-shot vs fine-tuned; table includes n-gram and other baselines.

const R = results as any
const pc = (x: number | null | undefined) => (x == null ? '—' : (x * 100).toFixed(1) + '%')

const C = {
  ns: '#0f9d6b', // next-step (green)
  cp: '#3b82f6', // completion (blue)
  an: '#f59e0b', // anomaly (amber)
  base: '#94a3b8', // baseline (grey)
}

function Card({
  title,
  metric,
  base,
  best,
  baseLabel,
  bestLabel,
}: {
  title: string
  metric: string
  base: number
  best: number
  baseLabel: string
  bestLabel: string
}) {
  const win = best >= base
  const w = (v: number) => Math.max(3, Math.min(100, v * 100))
  return (
    <div className="rs-card">
      <div className="rs-card-t">{title}</div>
      <div className="rs-card-m">{metric}</div>
      <div className="rs-row"><span>{baseLabel}</span><b>{pc(base)}</b></div>
      <div className="rs-bar"><i style={{ width: w(base) + '%', background: C.base }} /></div>
      <div className="rs-row">
        <span>{bestLabel}</span>
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
  const H = R.headline || {
    baselineName: 'DeepSeek-V4-Flash (zero-shot)',
    finetunedName: R.finetuned?.name || 'SFT instruct-all',
    nextstep: R.nextstep,
    completion: R.completion,
    anomaly: R.anomaly,
  }
  const tableRows = [
    ...(R.llmBaselines || []),
    R.ngram,
    R.finetuned,
    ...(R.featuredModels || []),
  ].filter(Boolean)

  const finSlugs = new Set(
    [R.finetuned?.slug, ...(R.featuredModels || []).map((m: any) => m.slug)].filter(Boolean),
  )
  const bestNextstep = R.bestByTask?.nextstep?.nextstepTop1 ?? R.finetuned?.nextstepTop1
  const bestCompletion =
    R.bestByTask?.completion?.completionBlockAcc ??
    Math.max(...(R.scaling || []).map((p: any) => p.completionBlockAcc ?? 0), R.finetuned?.completionBlockAcc ?? 0)
  const bestAnomaly = R.bestByTask?.anomaly?.anomalyF1 ?? R.finetuned?.anomalyF1

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
        <p className="rs-lede">
          Qwen2.5-1.5B fine-tuned on fab routes vs zero-shot baselines (Qwen 1.5B on the same gold-200 set, DeepSeek on mosfet40).
          Labeled MOSFET eval with gold labels.
        </p>
      </div>

      <h2 className="rs-h2">Headline — fine-tuned vs DeepSeek zero-shot</h2>
      <div className="rs-cards">
        <Card
          title="Next-step"
          metric="Top-1 accuracy"
          base={H.nextstep.baseline}
          best={H.nextstep.finetuned}
          baseLabel={H.baselineName}
          bestLabel={H.finetunedName}
        />
        <Card
          title="Completion"
          metric="Block accuracy"
          base={H.completion.baseline}
          best={H.completion.finetuned}
          baseLabel={H.baselineName}
          bestLabel={H.finetunedName}
        />
        <Card
          title="Anomaly"
          metric="F1 (rule violation)"
          base={H.anomaly.baseline}
          best={H.anomaly.finetuned}
          baseLabel={H.baselineName}
          bestLabel={H.finetunedName}
        />
      </div>
      {(H.baselineEvalNote || H.finetunedEvalNote) && (
        <p className="rs-sub" style={{ marginTop: -8, marginBottom: 20 }}>
          DeepSeek: {H.baselineEvalNote || 'mosfet40 subset'}. Fine-tuned: {H.finetunedEvalNote || 'full eval_local'}.
        </p>
      )}

      {tableRows.length > 0 && (
        <>
          <h2 className="rs-h2">All models (labeled eval, gold.json)</h2>
          <div className="rs-panel rs-fam">
            <div className="rs-frow rs-fhead"><span>Model</span><span>n</span><span>Next-step</span><span>Completion</span><span>Anomaly F1</span></div>
            {tableRows.map((row: any) => (
              <div
                className={
                  'rs-frow' +
                  (finSlugs.has(row.slug) ? ' rs-frow-best' : '') +
                  (row.highlight === 'completion' ? ' rs-frow-highlight' : '')
                }
                key={row.slug}
              >
                <span>
                  {finSlugs.has(row.slug) ? <b>{row.name}</b> : row.name}
                  {row.evalNote ? <em className="rs-note"> · {row.evalNote}</em> : null}
                  {row.highlight === 'completion' ? (
                    <em className="rs-pill win" style={{ marginLeft: 6 }}>
                      best completion
                    </em>
                  ) : null}
                </span>
                <span>{row.nNextstep ?? '—'}</span>
                <span>
                  {row.nextstepTop1 === bestNextstep && row.nextstepTop1 != null ? (
                    <b>{pc(row.nextstepTop1)}</b>
                  ) : (
                    pc(row.nextstepTop1)
                  )}
                </span>
                <span>
                  {row.completionBlockAcc === bestCompletion && row.completionBlockAcc != null ? (
                    <b>{pc(row.completionBlockAcc)}</b>
                  ) : (
                    pc(row.completionBlockAcc)
                  )}
                </span>
                <span>
                  {row.anomalyF1 === bestAnomaly && row.anomalyF1 != null ? (
                    <b>{pc(row.anomalyF1)}</b>
                  ) : (
                    pc(row.anomalyF1)
                  )}
                </span>
              </div>
            ))}
          </div>
        </>
      )}

      {R.wandb?.runs?.length > 0 && (
        <>
          <h2 className="rs-h2">W&B runs (eval metrics)</h2>
          <div className="rs-panel">
            <ul className="rs-take">
              {R.wandb.runs.map((r: any) => (
                <li key={r.runId}>
                  <a href={r.url} target="_blank" rel="noreferrer">{r.label}</a>
                  <span className="rs-note"> · {r.runId}</span>
                </li>
              ))}
            </ul>
          </div>
        </>
      )}

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
          <li>
            Fine-tuning a 1.5B model <b>doubles next-step accuracy</b> vs DeepSeek zero-shot ({pc(H.nextstep.finetuned)} vs{' '}
            {pc(H.nextstep.baseline)}).
          </li>
          {R.llmBaselines?.find((b: any) => b.slug === 'qwen-zeroshot-local200') && (
            <li>
              Same-model comparison on gold-200: Qwen zero-shot ({pc(R.llmBaselines.find((b: any) => b.slug === 'qwen-zeroshot-local200').nextstepTop1)}{' '}
              next-step) vs fine-tuned ({pc(R.finetuned?.nextstepTop1)}).
            </li>
          )}
          <li>
            Completion block accuracy jumps from {pc(H.completion.baseline)} → {pc(H.completion.finetuned)}; anomaly F1 from{' '}
            {pc(H.anomaly.baseline)} → {pc(H.anomaly.finetuned)}.
          </li>
          {R.ngram && (
            <li>
              The n-gram baseline remains strong on next-step ({pc(R.ngram.nextstepTop1)}). Fine-tuned models win on
              completion ({pc(bestCompletion)} block-acc at 2000 routes vs {pc(R.ngram.completionBlockAcc)} n-gram) and
              anomaly detection ({pc(R.bestByTask?.anomaly?.anomalyF1 ?? R.finetuned?.anomalyF1)} F1).
            </li>
          )}
          <li>All numbers are real labeled eval metrics — no placeholders. The live copilot runs the fine-tuned model on-device.</li>
        </ul>
      </div>
      <div className="rs-src">Generated {R.generatedAt} · best={R.bestName} · baseline={R.baselineName}</div>
    </div>
  )
}

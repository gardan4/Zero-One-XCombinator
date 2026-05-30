import {
  benchmarks,
  delta,
  deltaNum,
  deltaPct,
  num,
  pct,
  scaling,
  type AnomalyMetrics,
  type CompletionMetrics,
  type FamilyTasks,
  type ModelMetrics,
  type NextStepMetrics,
  type ScalingPoint,
} from '../lib/benchmarks'

// ===========================================================================
// Benchmarks — the base-model vs best-model process-logic eval, inside the
// copilot. Reads the bundled results.json (built from extras/results). Renders
// in BOTH themes via the shared tokens / .glass primitive. Degrades to a
// base-only view with "pending" badges when no finetuned run is promoted.
// ===========================================================================

const { base, best, oracle, perFamily, idOod, generatedAt } = benchmarks

/** A horizontal 0..1 bar (themed). `tone` switches the accent for the best run. */
function Bar({ value, tone = 'base' }: { value: number | null | undefined; tone?: 'base' | 'best' }) {
  const w = value == null ? 0 : Math.max(0, Math.min(1, value)) * 100
  return (
    <div className="bm-bar">
      <div className={`bm-bar-fill is-${tone}`} style={{ width: `${w}%` }} />
    </div>
  )
}

/** One metric row in the headline cards: label, base/best values + bars, delta. */
function MetricRow({
  label,
  hint,
  baseVal,
  bestVal,
  fmt,
  fmtDelta,
  higherIsBetter = true,
}: {
  label: string
  hint?: string
  baseVal: number | null | undefined
  bestVal: number | null | undefined
  fmt: (x: number | null | undefined) => string
  fmtDelta: (d: number | null) => string
  higherIsBetter?: boolean
}) {
  const d = delta(bestVal, baseVal)
  const good = d == null ? null : higherIsBetter ? d >= 0 : d <= 0
  return (
    <div className="bm-metric">
      <div className="bm-metric-head">
        <span className="bm-metric-label">{label}</span>
        {hint && <span className="bm-metric-hint">{hint}</span>}
      </div>
      <div className="bm-metric-grid">
        <div className="bm-cell">
          <span className="bm-val mono">{fmt(baseVal)}</span>
          <Bar value={baseVal} tone="base" />
        </div>
        <div className="bm-cell">
          {bestVal == null ? (
            <span className="bm-pending">pending</span>
          ) : (
            <>
              <span className="bm-val mono">{fmt(bestVal)}</span>
              <Bar value={bestVal} tone="best" />
            </>
          )}
        </div>
        <div className="bm-cell bm-delta-cell">
          {d == null ? (
            <span className="bm-delta is-none mono">—</span>
          ) : (
            <span className={`bm-delta mono ${good ? 'is-up' : 'is-down'}`}>{fmtDelta(d)}</span>
          )}
        </div>
      </div>
    </div>
  )
}

function TaskCard({
  eyebrow,
  title,
  lede,
  children,
}: {
  eyebrow: string
  title: string
  lede?: string
  children: React.ReactNode
}) {
  return (
    <section className="bm-card glass">
      <div className="bm-card-head">
        <span className="bm-eyebrow">{eyebrow}</span>
        <h3 className="bm-card-title">{title}</h3>
        {lede && <p className="bm-card-lede">{lede}</p>}
      </div>
      <div className="bm-colhead">
        <span />
        <span className="bm-colhead-base">Base</span>
        <span className="bm-colhead-best">Best{best == null ? ' (pending)' : ''}</span>
        <span className="bm-colhead-delta">Δ</span>
      </div>
      {children}
    </section>
  )
}

function NextStepCard({ b, f }: { b: NextStepMetrics | null; f: NextStepMetrics | null }) {
  return (
    <TaskCard eyebrow="Task 1 · next-step prediction" title="Predict the next process step" lede="Top-k accuracy + MRR over the held-out step vocabulary.">
      <MetricRow label="Top-1" baseVal={b?.top1} bestVal={f?.top1} fmt={(x) => pct(x)} fmtDelta={deltaPct} />
      <MetricRow label="Top-3" baseVal={b?.top3} bestVal={f?.top3} fmt={(x) => pct(x)} fmtDelta={deltaPct} />
      <MetricRow label="Top-5" baseVal={b?.top5} bestVal={f?.top5} fmt={(x) => pct(x)} fmtDelta={deltaPct} />
      <MetricRow label="MRR" hint="mean reciprocal rank" baseVal={b?.mrr} bestVal={f?.mrr} fmt={(x) => num(x)} fmtDelta={deltaNum} />
    </TaskCard>
  )
}

function CompletionCard({ b, f }: { b: CompletionMetrics | null; f: CompletionMetrics | null }) {
  // Lead with NED + Block Acc — exact-match is ~0 for everyone (expected on 100+ token routes).
  return (
    <TaskCard
      eyebrow="Task 2 · sequence completion"
      title="Complete the rest of the route"
      lede="Lead metrics: normalized edit distance & block accuracy. Exact-match is ~0 for all — expected on long routes."
    >
      <MetricRow
        label="Norm. edit dist"
        hint="higher = closer"
        baseVal={b?.normEditDist}
        bestVal={f?.normEditDist}
        fmt={(x) => num(x)}
        fmtDelta={deltaNum}
      />
      <MetricRow label="Block accuracy" baseVal={b?.blockAcc} bestVal={f?.blockAcc} fmt={(x) => pct(x)} fmtDelta={deltaPct} />
      <MetricRow label="Token accuracy" baseVal={b?.tokenAcc} bestVal={f?.tokenAcc} fmt={(x) => pct(x)} fmtDelta={deltaPct} />
      <MetricRow
        label="Exact match"
        hint="~0 for all — expected"
        baseVal={b?.exactMatch}
        bestVal={f?.exactMatch}
        fmt={(x) => pct(x, 2)}
        fmtDelta={(d) => deltaPct(d, 2)}
      />
    </TaskCard>
  )
}

function AnomalyCard({ b, f, oracleA }: { b: AnomalyMetrics | null; f: AnomalyMetrics | null; oracleA: AnomalyMetrics | null }) {
  // Confusion matrix — prefer best, fall back to base so it always renders.
  const conf = (f ?? b)?.confusion
  const confLabel = f ? 'Best' : 'Base'
  return (
    <TaskCard
      eyebrow="Task 3 · anomaly detection"
      title="Flag process-logic violations"
      lede="Binary accuracy, F1 & ROC-AUC against the rule verifier."
    >
      <MetricRow label="Binary acc" baseVal={b?.binAcc} bestVal={f?.binAcc} fmt={(x) => pct(x)} fmtDelta={deltaPct} />
      <MetricRow label="F1" baseVal={b?.f1} bestVal={f?.f1} fmt={(x) => num(x)} fmtDelta={deltaNum} />
      <MetricRow label="ROC-AUC" baseVal={b?.rocAuc} bestVal={f?.rocAuc} fmt={(x) => num(x)} fmtDelta={deltaNum} />
      <MetricRow label="Precision" baseVal={b?.precision} bestVal={f?.precision} fmt={(x) => num(x)} fmtDelta={deltaNum} />
      <MetricRow label="Recall" baseVal={b?.recall} bestVal={f?.recall} fmt={(x) => num(x)} fmtDelta={deltaNum} />

      {oracleA && (
        <div className="bm-oracle">
          <span className="bm-oracle-dot" />
          <span className="bm-oracle-text">
            <b>Oracle ceiling</b> — the submitted anomaly score is the rule verifier itself
            (bin-acc {num(oracleA.binAcc, 2)}, F1 {num(oracleA.f1, 2)}, ROC-AUC {num(oracleA.rocAuc, 2)} ≈ 1.0).
          </span>
        </div>
      )}

      {conf && (
        <div className="bm-confusion">
          <div className="bm-confusion-head">
            <span className="d-label">Confusion matrix</span>
            <span className="bm-confusion-src mono">{confLabel}</span>
          </div>
          <div className="bm-conf-grid">
            <div className="bm-conf-cell is-tp">
              <span className="bm-conf-k">TP</span>
              <span className="bm-conf-v mono">{conf.tp}</span>
            </div>
            <div className="bm-conf-cell is-fp">
              <span className="bm-conf-k">FP</span>
              <span className="bm-conf-v mono">{conf.fp}</span>
            </div>
            <div className="bm-conf-cell is-fn">
              <span className="bm-conf-k">FN</span>
              <span className="bm-conf-v mono">{conf.fn}</span>
            </div>
            <div className="bm-conf-cell is-tn">
              <span className="bm-conf-k">TN</span>
              <span className="bm-conf-v mono">{conf.tn}</span>
            </div>
          </div>
        </div>
      )}
    </TaskCard>
  )
}

/** Per-family table: base vs best, Top-1 / Block-acc / NED across MOSFET/IGBT/IC. */
function PerFamilyTable() {
  if (!perFamily.length) {
    return (
      <section className="bm-card glass">
        <div className="bm-card-head">
          <span className="bm-eyebrow">Per family</span>
          <h3 className="bm-card-title">By product family</h3>
        </div>
        <div className="bm-placeholder">No per-family slices promoted yet.</div>
      </section>
    )
  }
  const showBest = perFamily.some((r) => r.best)
  return (
    <section className="bm-card glass">
      <div className="bm-card-head">
        <span className="bm-eyebrow">Per family</span>
        <h3 className="bm-card-title">By product family</h3>
        <p className="bm-card-lede">Next-step Top-1 and completion block-accuracy, per family.</p>
      </div>
      <div className="bm-table">
        <div className="bm-tr bm-thead">
          <span className="bm-td">Family</span>
          <span className="bm-td bm-num">Top-1 base</span>
          {showBest && <span className="bm-td bm-num">Top-1 best</span>}
          <span className="bm-td bm-num">Block base</span>
          {showBest && <span className="bm-td bm-num">Block best</span>}
        </div>
        {perFamily.map((r) => (
          <div className="bm-tr" key={String(r.family)}>
            <span className="bm-td bm-fam">{r.family}</span>
            <span className="bm-td bm-num mono">{pct(r.base?.nextstep?.top1)}</span>
            {showBest && <span className="bm-td bm-num mono">{r.best ? pct(r.best.nextstep?.top1) : <em className="bm-pending">pending</em>}</span>}
            <span className="bm-td bm-num mono">{pct(r.base?.completion?.blockAcc)}</span>
            {showBest && <span className="bm-td bm-num mono">{r.best ? pct(r.best.completion?.blockAcc) : <em className="bm-pending">pending</em>}</span>}
          </div>
        ))}
      </div>
    </section>
  )
}

/** ID → OOD: "told the rules" (all-family, in-distribution) vs "learned the rules" (LOFO held-out). */
function IdOodTable() {
  const head = (
    <div className="bm-card-head">
      <span className="bm-eyebrow">Generalization · ID → OOD</span>
      <h3 className="bm-card-title">Told the rules vs learned the rules</h3>
      <p className="bm-card-lede">
        In-distribution (all families) vs leave-one-family-out. The OOD column is the family the model never trained on.
      </p>
    </div>
  )
  if (!idOod.length) {
    return (
      <section className="bm-card glass">
        {head}
        <div className="bm-placeholder">
          <span className="bm-placeholder-badge">Awaiting LOFO eval</span>
          Promote a <span className="mono">split:ood</span> finetuned run per held-out <span className="mono">family:X</span> to populate the
          generalization gap.
        </div>
      </section>
    )
  }
  const cell = (t: FamilyTasks) => ({
    top1: t.nextstep?.top1,
    block: t.completion?.blockAcc,
  })
  return (
    <section className="bm-card glass">
      {head}
      <div className="bm-table">
        <div className="bm-tr bm-thead">
          <span className="bm-td">Held-out family</span>
          <span className="bm-td bm-num">Top-1 ID</span>
          <span className="bm-td bm-num">Top-1 OOD</span>
          <span className="bm-td bm-num">Δ Top-1</span>
          <span className="bm-td bm-num">Block ID</span>
          <span className="bm-td bm-num">Block OOD</span>
        </div>
        {idOod.map((r) => {
          const id = cell(r.id)
          const ood = cell(r.ood)
          const d = delta(ood.top1, id.top1)
          return (
            <div className="bm-tr" key={String(r.family)}>
              <span className="bm-td bm-fam">{r.family}</span>
              <span className="bm-td bm-num mono">{pct(id.top1)}</span>
              <span className="bm-td bm-num mono">{pct(ood.top1)}</span>
              <span className={`bm-td bm-num mono ${d == null ? '' : d >= 0 ? 'bm-up' : 'bm-down'}`}>{deltaPct(d)}</span>
              <span className="bm-td bm-num mono">{pct(id.block)}</span>
              <span className="bm-td bm-num mono">{pct(ood.block)}</span>
            </div>
          )
        })}
      </div>
    </section>
  )
}

/**
 * Data scaling — does more training data help? Accuracy vs #training sequences.
 * Inline SVG line chart, no chart lib. X is ORDINAL (evenly spaced category slots,
 * not linear pixels) since sizes span 100→2000; Y is 0..1. Two series: next-step
 * Top-1 (teal) and completion block-acc (amber). Per-point values are labeled.
 */
function ScalingChart({ points }: { points: ScalingPoint[] }) {
  // viewBox geometry — scales responsively; px here are SVG units.
  const W = 560
  const H = 220
  const padL = 38 // room for the y-axis %
  const padR = 16
  const padT = 14
  const padB = 34 // room for the x size labels
  const plotW = W - padL - padR
  const plotH = H - padT - padB

  // Ordinal x: one evenly spaced slot per point (NOT proportional to size).
  const n = points.length
  const x = (i: number) => (n === 1 ? padL + plotW / 2 : padL + (plotW * i) / (n - 1))
  const y = (v: number) => padT + plotH * (1 - Math.max(0, Math.min(1, v)))

  const yTicks = [0, 0.25, 0.5, 0.75, 1]

  // Build a polyline for one series, skipping points whose value is null.
  const series = (key: 'nextstepTop1' | 'completionBlockAcc') =>
    points
      .map((p, i) => ({ i, v: p[key] }))
      .filter((d): d is { i: number; v: number } => d.v != null)

  const top1 = series('nextstepTop1')
  const block = series('completionBlockAcc')
  const poly = (pts: { i: number; v: number }[]) => pts.map((d) => `${x(d.i)},${y(d.v)}`).join(' ')

  return (
    <div className="bm-scaling">
      <div className="bm-scaling-legend">
        <span className="bm-scaling-key">
          <span className="bm-scaling-swatch is-nextstep" /> Next-step Top-1
        </span>
        <span className="bm-scaling-key">
          <span className="bm-scaling-swatch is-block" /> Completion Block-Acc
        </span>
      </div>
      <svg className="bm-scaling-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Accuracy vs number of training sequences" preserveAspectRatio="xMidYMid meet">
        {/* y gridlines + % labels */}
        {yTicks.map((t) => (
          <g key={t}>
            <line className="bm-sc-grid" x1={padL} x2={W - padR} y1={y(t)} y2={y(t)} />
            <text className="bm-sc-ytick mono" x={padL - 6} y={y(t) + 3} textAnchor="end">
              {Math.round(t * 100)}
            </text>
          </g>
        ))}

        {/* completion block-acc (amber) — drawn first so top-1 sits on top */}
        {block.length > 1 && <polyline className="bm-sc-line is-block" points={poly(block)} />}
        {block.map((d) => (
          <g key={`b-${d.i}`}>
            <circle className="bm-sc-dot is-block" cx={x(d.i)} cy={y(d.v)} r={3.4} />
            <text className="bm-sc-val is-block mono" x={x(d.i)} y={y(d.v) - 8} textAnchor="middle">
              {pct(d.v, 0)}
            </text>
          </g>
        ))}

        {/* next-step top-1 (teal) */}
        {top1.length > 1 && <polyline className="bm-sc-line is-nextstep" points={poly(top1)} />}
        {top1.map((d) => (
          <g key={`t-${d.i}`}>
            <circle className="bm-sc-dot is-nextstep" cx={x(d.i)} cy={y(d.v)} r={3.4} />
            <text className="bm-sc-val is-nextstep mono" x={x(d.i)} y={y(d.v) - 8} textAnchor="middle">
              {pct(d.v, 0)}
            </text>
          </g>
        ))}

        {/* x size labels (ordinal slots) */}
        {points.map((p, i) => (
          <text key={`x-${i}`} className="bm-sc-xtick mono" x={x(i)} y={H - 12} textAnchor="middle">
            {p.size.toLocaleString()}
          </text>
        ))}
      </svg>
      <div className="bm-scaling-axis">training sequences →</div>
    </div>
  )
}

function ScalingCard() {
  const head = (
    <div className="bm-card-head">
      <span className="bm-eyebrow">Data scaling</span>
      <h3 className="bm-card-title">Does more training data help?</h3>
      <p className="bm-card-lede">Overall accuracy vs the number of training sequences. Each point is a promoted run tagged <span className="mono">data-size:N</span>.</p>
    </div>
  )
  if (!scaling.length) {
    return (
      <section className="bm-card glass">
        {head}
        <div className="bm-placeholder">
          <span className="bm-placeholder-badge">Awaiting scaling runs…</span>
          Promote runs tagged <span className="mono">data-size:&lt;N&gt;</span> (e.g. <span className="mono">data-size:100</span>,{' '}
          <span className="mono">data-size:2000</span>) then run <span className="mono">npm run build:benchmarks</span>.
        </div>
      </section>
    )
  }
  return (
    <section className="bm-card glass">
      {head}
      <ScalingChart points={scaling} />
    </section>
  )
}

export default function Benchmarks() {
  // base should always exist once anything is promoted; guard anyway.
  const b: ModelMetrics | null = base
  const f: ModelMetrics | null = best
  const gen = generatedAt ? new Date(generatedAt) : null

  return (
    <div className="bm-view">
      <header className="bm-hero glass">
        <div className="bm-hero-main">
          <span className="bm-eyebrow">Process-logic eval · base vs best</span>
          <h2 className="bm-hero-title">
            Base model <span className="bm-vs">vs</span> best model
          </h2>
          <p className="bm-hero-lede">
            Labeled eval over fab step sequences — next-step prediction, sequence completion, and anomaly detection. Numbers are
            generated from <span className="mono">extras/results</span>, not placeholders.
          </p>
        </div>
        <div className="bm-hero-side">
          <div className="bm-legend">
            <span className="bm-legend-item">
              <span className="bm-swatch is-base" /> {b?.label ?? 'Base'}
            </span>
            <span className="bm-legend-item">
              <span className="bm-swatch is-best" /> {f?.label ?? 'Best (pending)'}
            </span>
          </div>
          {f == null && <span className="bm-await-badge">Finetuned run pending</span>}
          {gen && <span className="bm-gen mono">generated {gen.toISOString().slice(0, 10)}</span>}
        </div>
      </header>

      {b == null && f == null ? (
        <section className="bm-card glass">
          <div className="bm-placeholder">
            <span className="bm-placeholder-badge">No results yet</span>
            Promote eval results (<span className="mono">just promote</span>) then run{' '}
            <span className="mono">npm run build:benchmarks</span>.
          </div>
        </section>
      ) : (
        <>
          <div className="bm-grid">
            <NextStepCard b={b?.nextstep ?? null} f={f?.nextstep ?? null} />
            <CompletionCard b={b?.completion ?? null} f={f?.completion ?? null} />
            <AnomalyCard b={b?.anomaly ?? null} f={f?.anomaly ?? null} oracleA={oracle?.anomaly ?? null} />
          </div>

          <div className="bm-grid bm-grid-2">
            <PerFamilyTable />
            <IdOodTable />
          </div>

          <ScalingCard />
        </>
      )}
    </div>
  )
}

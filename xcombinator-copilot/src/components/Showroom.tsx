import { useEffect, useRef, useState } from 'react'
import type { Family, Prediction, Violation } from '../types'
import { defaultSample, expectedLength } from '../lib/data'
import { validateRoute } from '../lib/rules'
import { predictNextStep } from '../lib/model'
import { modelLabel } from './TopBar'
import samplesRaw from '../data/eval_samples.json'

// Live showcase: take a RANDOM real route from a track eval-input set and walk it FROM THE START. At
// each step every served model predicts the next step; we compare each guess to the route's actual
// next step and light it green (correct) or red (wrong). The fine-tune answers instantly; the hosted
// DeepSeek reasons for a beat, so the cards fill in live.

type EvalSample = { id: string; family: Family; steps: string[] }
const SAMPLES = samplesRaw as unknown as { valid: EvalSample[]; anomaly: EvalSample[] }

const DATASETS: { key: 'valid' | 'anomaly'; label: string; blurb: string }[] = [
  { key: 'valid', label: 'Valid routes', blurb: 'real routes from eval_input_valid.csv — every step follows the process logic' },
  { key: 'anomaly', label: 'Anomaly set', blurb: 'routes from eval_input_anomaly.csv that may hide one rule violation' },
]

const START = 5 // steps revealed before the first prediction — start near the beginning of the route
const ORDER = ['qwen-base', 'deepseek-v4-flash', 'sft-best']
const isOurs = (id: string) => id === 'sft-best' || id.includes('sft')
const isHosted = (id: string) => id.includes('deepseek')

const norm = (s: string) => (s || '').toUpperCase().replace(/_/g, ' ').replace(/\s+/g, ' ').trim()
function prettyRule(rule: string): string {
  return rule.replace(/^RULE_/, '').replace(/_/g, ' ').toLowerCase()
}

interface Props {
  models: string[]
}

type Cell = { pred: Prediction | null; loading: boolean }
type Score = { correct: number; total: number }

export default function Showroom({ models }: Props) {
  const ordered = [...models].sort((a, b) => {
    const ia = ORDER.indexOf(a),
      ib = ORDER.indexOf(b)
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib)
  })

  function pick(ds: 'valid' | 'anomaly'): EvalSample {
    const pool = SAMPLES[ds]
    if (pool && pool.length) return pool[Math.floor(Math.random() * pool.length)]
    const s = defaultSample('MOSFET') // fallback if the samples JSON wasn't built
    return { id: s.id, family: 'MOSFET', steps: s.steps }
  }

  const [dataset, setDataset] = useState<'valid' | 'anomaly'>('valid')
  const [sample, setSample] = useState<EvalSample>(() => pick('valid'))
  const [cursor, setCursor] = useState(START)
  const [cells, setCells] = useState<Record<string, Cell>>({})
  const [score, setScore] = useState<Record<string, Score>>({})
  const token = useRef(0)

  const family = sample.family
  const full = sample.steps
  const revealed = full.slice(0, cursor)
  const expected = full[cursor] // the route's actual next step (ground truth); undefined at the end
  const atEnd = cursor >= full.length

  function runAll(seq: string[], fam: Family, ids: string[]) {
    const t = ++token.current
    setCells(Object.fromEntries(ids.map((id) => [id, { pred: null, loading: true }])))
    for (const id of ids) {
      predictNextStep(fam, seq, id)
        .then((pred) => {
          if (t === token.current) setCells((c) => ({ ...c, [id]: { pred, loading: false } }))
        })
        .catch(() => {
          if (t === token.current) setCells((c) => ({ ...c, [id]: { pred: null, loading: false } }))
        })
    }
  }

  // (re)run whenever the served model list, the chosen route, or the cursor changes
  useEffect(() => {
    if (ordered.length && !atEnd) runAll(revealed, family, ordered)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models.join(','), sample.id, cursor])

  function load(ds: 'valid' | 'anomaly') {
    setDataset(ds)
    setSample(pick(ds))
    setCursor(START)
    setScore({})
  }
  function shuffle() {
    setSample(pick(dataset))
    setCursor(START)
    setScore({})
  }
  // Step forward along the REAL route: bank each model's correctness for this position, then reveal
  // the actual next step (so all models are scored against the same ground-truth procedure).
  function advance() {
    if (atEnd) return
    setScore((prev) => {
      const next = { ...prev }
      for (const id of ordered) {
        const p = cells[id]?.pred
        const s = next[id] || { correct: 0, total: 0 }
        if (p && p.source === id) next[id] = { correct: s.correct + (norm(p.step) === norm(expected) ? 1 : 0), total: s.total + 1 }
      }
      return next
    })
    setCursor((c) => c + 1)
  }

  const baseViol = validateRoute(revealed)
  const total = expectedLength(family)
  const tail = revealed.slice(-6)
  const anyLoading = ordered.some((id) => cells[id]?.loading)

  function verdict(pred: Prediction | null) {
    if (!pred) return null
    const before = new Set(baseViol.map((x) => `${x.rule}@${x.stepIndex}`))
    const after = validateRoute([...revealed, pred.step])
    const fresh = (after.find((x) => !before.has(`${x.rule}@${x.stepIndex}`)) ?? null) as Violation | null
    const correct = expected != null && norm(pred.step) === norm(expected)
    return { correct, breaksRule: fresh }
  }

  if (!models.length) {
    return (
      <div className="rs-wrap">
        <div className="rs-head">
          <div className="rs-eyebrow">Live model compare</div>
          <h1>No inference server reachable</h1>
          <p className="rs-lede">
            Start the local model server (<code>scripts/serve_copilot_mac.py</code>, port 8001) — or run the VS Code
            “Demo: start all” task — then reload. This page walks a real fab route through every served model at once.
          </p>
        </div>
      </div>
    )
  }

  const dsBlurb = DATASETS.find((d) => d.key === dataset)?.blurb
  return (
    <div className="rs-wrap">
      <div className="rs-head">
        <div className="rs-eyebrow">Live model compare · same route, every model</div>
        <h1>Walking a real fab route — who predicts the <em>next step</em>?</h1>
        <p className="rs-lede">
          A random route from the track eval inputs, replayed from the start. At each step every served model predicts
          what comes next; we score each guess against the route’s real next step — green for a hit, red for a miss.
        </p>
      </div>

      <div className="sr-bar">
        <div className="sr-fam" role="tablist" aria-label="Eval dataset">
          {DATASETS.map((d) => (
            <button key={d.key} role="tab" aria-selected={d.key === dataset} className={`sr-seg${d.key === dataset ? ' on' : ''}`} onClick={() => (d.key === dataset ? shuffle() : load(d.key))}>
              {d.label}
            </button>
          ))}
        </div>
        <div className="sr-ctx mono">
          <span className="sr-fam-chip">{family}</span> {sample.id} · step <b>{cursor + 1}</b> / ~{total}
        </div>
        <div className="sr-actions">
          <button className="sr-btn primary" onClick={shuffle} disabled={anyLoading}>
            🎲 Random route
          </button>
          <button className="sr-btn" onClick={() => runAll(revealed, family, ordered)} disabled={anyLoading || atEnd}>
            {anyLoading ? 'Predicting…' : 'Re-run'}
          </button>
          <button className="sr-btn ghost" onClick={advance} disabled={anyLoading || atEnd}>
            Next step →
          </button>
        </div>
      </div>
      <p className="sr-dsblurb">{dsBlurb}</p>

      <div className="sr-seq">
        <span className="sr-seq-l">Route so far</span>
        <span className="sr-seq-steps mono">
          {revealed.length > tail.length && <span className="sr-ell">… +{revealed.length - tail.length} </span>}
          {tail.map((s, i) => (
            <span key={i} className="sr-chip">
              {s}
            </span>
          ))}
          {expected != null ? (
            <span className="sr-truth mono" title="the route’s real next step (ground truth)">
              actual next → {expected}
            </span>
          ) : (
            <span className="sr-truth done">route complete</span>
          )}
        </span>
      </div>

      <div className="sr-cards">
        {ordered.map((id) => {
          const cell = cells[id] || { pred: null, loading: true }
          const pred = cell.pred
          const v = verdict(pred)
          const sc = score[id]
          return (
            <div key={id} className={`sr-card${isOurs(id) ? ' ours' : ''}`}>
              <div className="sr-card-head">
                <span className="sr-name">{modelLabel(id)}</span>
                <span className={`sr-tag ${isHosted(id) ? 'hosted' : 'local'}`}>{isHosted(id) ? 'hosted' : 'on-device'}</span>
                {isOurs(id) && <span className="sr-tag ours">ours</span>}
                {sc && sc.total > 0 && (
                  <span className="sr-score mono" title="correct next-step predictions on this route">
                    {sc.correct}/{sc.total}
                  </span>
                )}
              </div>

              {cell.loading ? (
                <div className="sr-pred loading">
                  <span className="sr-spinner" />
                  <span className="sr-thinking">thinking…</span>
                </div>
              ) : pred && pred.source !== id ? (
                <div className="sr-pred err">
                  couldn’t reach {isHosted(id) ? 'Featherless' : 'the model server'} — hit “Re-run”.
                </div>
              ) : pred && v ? (
                <>
                  <div className={`sr-pred${v.correct ? ' correct' : ' wrong'}`}>
                    <span className="sr-step mono">{pred.step}</span>
                    <span className="sr-mark">{v.correct ? '✓' : '✗'}</span>
                  </div>
                  <div className={`sr-valid${v.correct ? '' : ' bad'}`}>
                    {v.correct
                      ? '✓ correct — matches the real next step'
                      : v.breaksRule
                        ? `✗ wrong — would break ${prettyRule(v.breaksRule.rule)}`
                        : `✗ differs — route’s next is ${expected}`}
                  </div>
                  {pred.reasoning ? (
                    <p className="sr-reason">{pred.reasoning}</p>
                  ) : (
                    <p className="sr-reason muted">No chain-of-thought — answers the step directly.</p>
                  )}
                  {pred.alternates && pred.alternates.length > 0 && (
                    <div className="sr-alts">
                      <span className="sr-alts-l">alternates</span>
                      {pred.alternates.map((a) => (
                        <span key={a} className={`sr-alt mono${expected != null && norm(a) === norm(expected) ? ' hit' : ''}`}>
                          {a}
                        </span>
                      ))}
                    </div>
                  )}
                </>
              ) : (
                <div className="sr-pred err">no prediction</div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

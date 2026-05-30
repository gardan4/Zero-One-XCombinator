import { useEffect, useRef } from 'react'
import type { Prediction, Violation } from '../types'
import type { Phase, RoadmapPhase } from '../lib/grammar'
import { RULE_SHORT } from '../lib/rules'

const HEAD_CAP = 9 // when the head phase is long, show only the last N steps (compressed)

interface Props {
  phases: Phase[]
  headIdx: number
  headPhaseIndex: number
  expandedIndex: number
  selectedIdx: number
  prediction: Prediction | null
  predicting: boolean
  /** does appending prediction.step keep the route grammar-valid? */
  predValid: boolean
  /** the new violation appending the step would introduce, if any */
  predViolation: Violation | null
  roadmap: RoadmapPhase[]
  violations: Violation[]
  onSelectStep: (idx: number) => void
  onExpandPhase: (phaseIndex: number) => void
}

export default function ProcessRoute(props: Props) {
  const { phases, headIdx, headPhaseIndex, expandedIndex, selectedIdx, prediction, predicting, predValid, predViolation, roadmap, violations, onSelectStep, onExpandPhase } = props
  const wrapRef = useRef<HTMLDivElement>(null)
  const expandedRef = useRef<HTMLDivElement>(null)
  const headRef = useRef<HTMLButtonElement>(null)
  const badIdx = new Set(violations.map((v) => v.stepIndex))

  // Frame the current window: keep the head (and its predicted ghost just right
  // of it) centred as the route grows. Uses rects so it's robust to offsetParent.
  useEffect(() => {
    const wrap = wrapRef.current
    const anchor = headRef.current ?? expandedRef.current
    if (!wrap || !anchor) return
    const wr = wrap.getBoundingClientRect()
    const ar = anchor.getBoundingClientRect()
    const delta = ar.left - wr.left - wrap.clientWidth * 0.44
    wrap.scrollBy({ left: delta, behavior: 'smooth' })
  }, [headIdx, expandedIndex, phases.length, prediction?.step, predicting])

  const firstStep = phases[0]?.steps[0]?.step ?? 'RECEIVE WAFER LOT'
  const lastActualIndex = phases[phases.length - 1]?.index ?? 0
  const future = roadmap.filter((r) => r.index > lastActualIndex)

  const items: React.ReactNode[] = []
  items.push(
    <div className="r-anchor" key="anchor">
      <span className="a-cap">Route start</span>
      <span className="a-step mono">{firstStep}</span>
    </div>,
  )

  phases.forEach((p, pi) => {
    const isHeadPhase = p.index === headPhaseIndex
    const isExpanded = p.index === expandedIndex
    items.push(<span className={`r-conn${isExpanded ? ' live' : ''}`} key={`c${pi}`} />)

    if (isExpanded) {
      const cap = isHeadPhase ? HEAD_CAP : p.steps.length
      const shown = p.steps.slice(Math.max(0, p.steps.length - cap))
      const hidden = p.steps.length - shown.length
      items.push(
        <div className="r-expanded" key={`e${pi}`} ref={expandedRef}>
          <div className="r-phase-label">
            <span className="lbl-line" />
            {p.name}
            {isHeadPhase ? ' — current phase' : ''}
            {hidden > 0 && <span className="r-trunc mono">last {shown.length} of {p.steps.length}</span>}
          </div>
          <div className="r-steps">
            {shown.map((s) => {
              const isHead = s.idx === headIdx
              const isSel = s.idx === selectedIdx
              const isBad = badIdx.has(s.idx)
              return (
                <button
                  key={s.idx}
                  ref={isHead ? headRef : undefined}
                  className={`r-step${isHead ? ' head' : ''}${isSel ? ' sel' : ''}${isBad ? ' bad' : ''}`}
                  onClick={() => onSelectStep(s.idx)}
                >
                  {isHead && <span className="r-badge">Head</span>}
                  <span className="r-node" />
                  <span className="r-label mono">{s.step}</span>
                  <span className="r-idx mono">{s.idx + 1}</span>
                </button>
              )
            })}

            {isHeadPhase && (predicting || prediction) && (
              <div
                className={`r-predict${predicting ? ' forming' : ''}${!predicting && prediction && !predValid ? ' breaks' : ''}`}
                key="ghost"
              >
                <span className="r-ptag">{predicting ? 'Predicting' : 'Predicted next'}</span>
                <span className="r-ghost" />
                {!predicting && prediction && <span className="r-plabel mono">{prediction.step}</span>}
                {!predicting && prediction && (
                  <span
                    className={`r-pchip${predValid ? ' ok' : ' warn'}`}
                    title={
                      predValid
                        ? 'Appending this step keeps the route grammar-valid.'
                        : predViolation
                          ? RULE_SHORT[predViolation.rule] ?? predViolation.description
                          : 'Appending this step would break a process-logic rule.'
                    }
                  >
                    {predValid ? '✓ valid next step' : `⚠ would break ${predViolation ? prettyRule(predViolation.rule) : 'a rule'}`}
                  </span>
                )}
                {!predicting && prediction && (
                  <span className="r-pconf mono">
                    <span className="pc-k">model confidence</span>
                    <span className="pc-bar"><span className="pc-fill" style={{ width: `${Math.round(prediction.confidence * 100)}%` }} /></span>
                    <span className="pc-v">{Math.round(prediction.confidence * 100)}%</span>
                  </span>
                )}
              </div>
            )}
          </div>
        </div>,
      )
    } else {
      items.push(
        <button
          className={`r-chip${isHeadPhase ? ' current' : ' done'}`}
          key={`p${pi}`}
          onClick={() => onExpandPhase(p.index)}
          title={`Expand ${p.name}`}
        >
          <span className="pc-name">{p.name}</span>
          <span className="pc-meta mono">
            {p.repeats && p.repeats > 1 ? `×${p.repeats} ↻` : `· ${p.steps.length}`}
          </span>
        </button>,
      )
    }
  })

  if (future.length) {
    items.push(<span className="r-conn future" key="cf" />)
    items.push(
      <div className="r-future" key="future">
        {future.map((f, i) => (
          <span className="r-future-chip" key={f.index}>
            {i > 0 && <span className="fc-conn" />}
            {f.name}
          </span>
        ))}
      </div>,
    )
  }

  return (
    <div className="track-wrap" ref={wrapRef}>
      <div className="track-fade l" />
      <div className="track-fade r" />
      <div className="track">{items}</div>
    </div>
  )
}

/** Compact, human-readable name for a rule id, e.g. RULE_DEP_NO_CLEAN → "dep no clean". */
function prettyRule(rule: string): string {
  return rule.replace(/^RULE_/, '').replace(/_/g, ' ').toLowerCase()
}

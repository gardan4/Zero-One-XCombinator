import { useEffect, useRef } from 'react'
import type { Prediction, Violation } from '../types'
import { RULE_SHORT } from '../lib/rules'

interface Props {
  steps: string[]
  headIdx: number
  selectedIdx: number
  prediction: Prediction | null
  predicting: boolean
  /** does appending prediction.step keep the route grammar-valid? */
  predValid: boolean
  /** the new violation appending the step would introduce, if any */
  predViolation: Violation | null
  violations: Violation[]
  onSelectStep: (idx: number) => void
}

/** Compact, human-readable name for a rule id, e.g. RULE_DEP_NO_CLEAN → "dep no clean". */
function prettyRule(rule: string): string {
  return rule.replace(/^RULE_/, '').replace(/_/g, ' ').toLowerCase()
}

export default function ProcessRoute({ steps, headIdx, selectedIdx, prediction, predicting, predValid, predViolation, violations, onSelectStep }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const selRef = useRef<HTMLButtonElement>(null)
  const headRef = useRef<HTMLButtonElement>(null)
  const badIdx = new Set(violations.map((v) => v.stepIndex))

  // keep the selected step centred (it follows the head as the route grows,
  // and follows a click when inspecting an earlier step or seeking on the bar)
  useEffect(() => {
    const wrap = wrapRef.current
    const anchor = selRef.current ?? headRef.current
    if (!wrap || !anchor) return
    const wr = wrap.getBoundingClientRect()
    const ar = anchor.getBoundingClientRect()
    wrap.scrollBy({ left: ar.left - wr.left - wrap.clientWidth * 0.42, behavior: 'smooth' })
  }, [selectedIdx, headIdx, prediction?.step, predicting])

  return (
    <div className="track-frame">
      {/* fades live in the non-scrolling frame so they stay pinned to the
          visible edges instead of scrolling away with the track content */}
      <div className="track-fade l" />
      <div className="track-fade r" />
      <div className="track-wrap" ref={wrapRef}>
        <div className="track">
          <div className="r-flat">
            {steps.map((step, idx) => {
              const isHead = idx === headIdx
              const isSel = idx === selectedIdx
              const isBad = badIdx.has(idx)
              return (
                <button
                  key={idx}
                  ref={isSel ? selRef : isHead ? headRef : undefined}
                  className={`r-step${isHead ? ' head' : ''}${isSel ? ' sel' : ''}${isBad ? ' bad' : ''}`}
                  onClick={() => onSelectStep(idx)}
                >
                  {isHead && <span className="r-badge">Head</span>}
                  <span className="r-node" />
                  <span className="r-label mono">{step}</span>
                  <span className="r-idx mono">{idx + 1}</span>
                </button>
              )
            })}

            {(predicting || prediction) && (
              <div className={`r-predict${predicting ? ' forming' : ''}${!predicting && prediction && !predValid ? ' breaks' : ''}`}>
                <span className="r-ptag">{predicting ? 'Predicting' : 'Predicted next'}</span>
                <span className="r-ghost" />
                {!predicting && prediction && <span className="r-plabel mono">{prediction.step}</span>}
                {!predicting && prediction && (
                  <div className="r-pcard">
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
                    <span className="r-pconf">
                      <span className="pc-k">{prediction.confidenceKnown === false ? 'hosted model · no logprobs' : 'model confidence'}</span>
                      <span className="pc-bar"><span className="pc-fill" style={{ width: prediction.confidenceKnown === false ? '0%' : `${Math.round(prediction.confidence * 100)}%` }} /></span>
                      <span className="pc-v mono">{prediction.confidenceKnown === false ? '—' : `${Math.round(prediction.confidence * 100)}%`}</span>
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

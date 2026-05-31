import { useEffect, useRef } from 'react'
import type { Prediction, Violation } from '../types'

interface Props {
  steps: string[]
  headIdx: number
  selectedIdx: number
  prediction: Prediction | null
  predicting: boolean
  violations: Violation[]
  onSelectStep: (idx: number) => void
}

export default function ProcessRoute({ steps, headIdx, selectedIdx, prediction, predicting, violations, onSelectStep }: Props) {
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
              <div className={`r-predict${predicting ? ' forming' : ''}`}>
                <span className="r-ptag">{predicting ? 'Predicting' : 'Predicted next'}</span>
                <span className="r-ghost" />
                {!predicting && prediction && <span className="r-plabel mono">{prediction.step}</span>}
                {!predicting && prediction && <span className="r-pconf mono">{Math.round(prediction.confidence * 100)}%</span>}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

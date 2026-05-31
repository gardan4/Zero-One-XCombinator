import type { Prediction } from '../types'
import type { Category } from '../lib/grammar'
import type { StepRule } from '../lib/rules'

interface Props {
  step: string
  idx: number
  total: number
  category: Category
  description: string | null
  prediction: Prediction | null
  rule: StepRule | null
  fraction: number
  isHead: boolean
  complete: boolean
}

export default function StepDetail({ step, idx, total, category, description, prediction, rule, fraction, isHead, complete }: Props) {
  const pct = Math.round(Math.min(1, fraction) * 100)
  return (
    <aside className="detail glass">
      <div className="d-head">
        <span className="d-eyebrow">Step detail</span>
        {isHead && (
          <span className="d-live">
            <span className="ld" />
            {complete ? 'Final' : 'Head'}
          </span>
        )}
      </div>

      <div className="d-token mono">{step}</div>
      <div className="d-pos mono">
        step <b>{idx + 1}</b> / {total} · {category}
      </div>

      <div className="divider" />

      {description && (
        <div className="d-row">
          <div className="d-label">Description</div>
          <div className="d-desc">{description}</div>
        </div>
      )}

      <div className="d-row">
        <div className="d-label">Category</div>
        <span className="d-cat">
          <span className="cd" />
          {category}
        </span>
      </div>

      {isHead && prediction && !complete && (
        <div className="d-row">
          <div className="d-label">Predicted next</div>
          <div className="d-pred">
            <span className="pp-ghost" />
            <span className="pp-body">
              <span className="pp-name mono">{prediction.step}</span>
              <span className="pp-conf">confidence · {prediction.source}</span>
            </span>
            <span className="pp-pct mono">{Math.round(prediction.confidence * 100)}%</span>
          </div>
        </div>
      )}

      {rule && (
        <div className="d-row">
          <div className="d-label">Process logic</div>
          <div className={`d-rule${rule.satisfied ? '' : ' is-bad'}`}>
            <span className="rk">
              {rule.satisfied ? (
                <svg width="8" height="8" viewBox="0 0 10 10" fill="none">
                  <path d="M1.6 5.2l2.1 2.1L8.4 2.6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              ) : (
                <svg width="9" height="9" viewBox="0 0 12 12" fill="none">
                  <path d="M6 1.6l4.6 8H1.4l4.6-8z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
                  <path d="M6 5v2.1" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
                  <circle cx="6" cy="8.7" r="0.6" fill="currentColor" />
                </svg>
              )}
            </span>
            <span>
              <span className="rr-id mono">{rule.rule}</span>
              <span className="rr-desc">{rule.text}</span>
            </span>
          </div>
        </div>
      )}

      <div className="grow" />

      <div className="divider" />

      <div className="progress">
        <div className="d-label">Route position</div>
        <div className="pbar">
          <div className="pfill" style={{ width: `${pct}%` }} />
        </div>
        <div className="pcap mono">
          <span className="start">RECEIVE WAFER LOT</span>
          <span className="now">{idx + 1}</span>
          <span className="end">SHIP LOT</span>
        </div>
      </div>
    </aside>
  )
}

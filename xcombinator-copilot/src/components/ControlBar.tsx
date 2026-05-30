import type { Violation } from '../types'

interface Props {
  onPredict: () => void
  predicting: boolean
  complete: boolean
  violations: Violation[]
}

export default function ControlBar({ onPredict, predicting, complete, violations }: Props) {
  const bad = violations.length > 0
  return (
    <div className="control-bar glass">
      <button className="btn-primary" onClick={onPredict} disabled={predicting || complete}>
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <path d="M3 2.5l7 4.5-7 4.5V2.5z" fill="currentColor" />
        </svg>
        {complete ? 'Route complete' : predicting ? 'Predicting…' : 'Predict next step'}
      </button>

      <div className="toggle-soon" title="Auto-advance every step to SHIP LOT — coming soon">
        <span className="switch" />
        Auto-run to SHIP LOT
        <span className="soon">Soon</span>
      </div>

      <div className="spacer" />

      {bad ? (
        <div className="anomaly is-bad">
          <span className="warn">
            <svg width="10" height="10" viewBox="0 0 12 12" fill="none">
              <path d="M6 1.6l4.6 8H1.4l4.6-8z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
              <path d="M6 5v2.1" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
              <circle cx="6" cy="8.7" r="0.6" fill="currentColor" />
            </svg>
          </span>
          {violations.length} issue{violations.length > 1 ? 's' : ''} · <span className="mono rule">{violations[0].rule}</span>
        </div>
      ) : (
        <div className="anomaly">
          <span className="ok">
            <svg width="9" height="9" viewBox="0 0 10 10" fill="none">
              <path d="M1.6 5.2l2.1 2.1L8.4 2.6" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </span>
          Process logic valid
        </div>
      )}
    </div>
  )
}

import type { Family } from '../types'
import { FAMILIES } from '../lib/data'

interface Props {
  family: Family
  onFamily: (f: Family) => void
  onImport: () => void
  live: boolean
}

export default function TopBar({ family, onFamily, onImport, live }: Props) {
  return (
    <header className="topbar glass">
      <div className="wordmark">
        <span className="mark-dot" />
        <span className="name">XCombinator</span>
        <span className="tag">Fab Process Copilot</span>
      </div>

      <div className="spacer" />

      <span className="seg-label">Family</span>
      <div className="segmented" role="tablist" aria-label="Product family">
        {FAMILIES.map((f) => (
          <button
            key={f}
            role="tab"
            aria-selected={f === family}
            className={`seg${f === family ? ' active' : ''}`}
            onClick={() => onFamily(f)}
          >
            {f}
          </button>
        ))}
      </div>

      <button className="btn-ghost" onClick={onImport}>
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <path d="M7 9.2V1.8M7 1.8L4.2 4.6M7 1.8l2.8 2.8" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M2 9v2.2c0 .55.45 1 1 1h8c.55 0 1-.45 1-1V9" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
        </svg>
        Import sequence
      </button>

      <div className={`status-pill${live ? ' is-live' : ''}`}>
        <span className="dot" />
        {live ? 'Live · model' : 'Simulated'}
        {!live && <span className="future">Live · model</span>}
      </div>
    </header>
  )
}

import type { Family } from '../types'
import { FAMILIES } from '../lib/data'

interface Props {
  family: Family
  onFamily: (f: Family) => void
  onImport: () => void
  live: boolean
  models: string[]
  selectedModel: string
  onSelectModel: (m: string) => void
  theme: 'dark' | 'light'
  onToggleTheme: () => void
}

/** Map a served model id to a friendly label, falling back to the raw id. */
export function modelLabel(id: string): string {
  const lc = id.toLowerCase()
  if (id === 'base-qwen' || lc.includes('base')) return 'Base · Qwen2.5-1.5B'
  if (id === 'sft-best' || id === 'sft-fab-all' || lc.includes('sft')) return 'Fine-tuned · SFT'
  if (lc.includes('grpo')) return 'RL · GRPO'
  if (lc.includes('lofo')) return 'OOD · LOFO'
  return id
}

export default function TopBar({ family, onFamily, onImport, live, models, selectedModel, onSelectModel, theme, onToggleTheme }: Props) {
  return (
    <header className="topbar glass">
      <div className="wordmark">
        <span className="mark-dot" />
        <span className="name">XCombinator</span>
        <span className="tag">Process Copilot</span>
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

      <span className="seg-label">Model</span>
      {models.length > 0 ? (
        <div className="model-select">
          <select aria-label="Inference model" value={selectedModel} onChange={(e) => onSelectModel(e.target.value)}>
            {models.map((m) => (
              <option key={m} value={m}>
                {modelLabel(m)}
              </option>
            ))}
          </select>
          <svg className="ms-caret" width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden>
            <path d="M2.4 3.8L5 6.4l2.6-2.6" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
      ) : (
        <span className="model-chip" title="No inference server reachable">
          Simulated (no server)
        </span>
      )}

      <button className="btn-ghost" onClick={onImport}>
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <path d="M7 9.2V1.8M7 1.8L4.2 4.6M7 1.8l2.8 2.8" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M2 9v2.2c0 .55.45 1 1 1h8c.55 0 1-.45 1-1V9" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
        </svg>
        Import sequence
      </button>

      <button
        className="icon-btn"
        onClick={onToggleTheme}
        aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
        title={theme === 'dark' ? 'Light theme' : 'Dark theme'}
      >
        {theme === 'dark' ? (
          <svg width="15" height="15" viewBox="0 0 16 16" fill="none">
            <circle cx="8" cy="8" r="3.1" stroke="currentColor" strokeWidth="1.3" />
            <path d="M8 1.4v1.6M8 13v1.6M1.4 8H3M13 8h1.6M3.3 3.3l1.1 1.1M11.6 11.6l1.1 1.1M12.7 3.3l-1.1 1.1M4.4 11.6l-1.1 1.1" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
          </svg>
        ) : (
          <svg width="15" height="15" viewBox="0 0 16 16" fill="none">
            <path d="M13.4 9.6A5.6 5.6 0 0 1 6.4 2.6a5.6 5.6 0 1 0 7 7z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
          </svg>
        )}
      </button>

      <div className={`status-pill${live && selectedModel ? ' is-live' : ''}`}>
        <span className="dot" />
        {live && selectedModel ? `Live · ${modelLabel(selectedModel)}` : 'Simulated'}
      </div>
    </header>
  )
}

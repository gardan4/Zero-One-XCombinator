import { useState } from 'react'
import type { Family } from '../types'
import { FAMILIES, SAMPLES } from '../lib/data'

export interface ImportPayload {
  family: Family
  steps: string[]
  routeId: string
  label?: string
}

interface Props {
  family: Family
  onClose: () => void
  onImport: (p: ImportPayload) => void
}

export default function ImportDialog({ family, onClose, onImport }: Props) {
  const [pasteFamily, setPasteFamily] = useState<Family>(family)
  const [routeName, setRouteName] = useState('')
  const [text, setText] = useState('')

  const parsed = text
    .split(/[|\n]/)
    .map((s) => s.trim())
    .filter(Boolean)

  function importPasted() {
    if (parsed.length < 2) return
    const label = routeName.trim() || `${pasteFamily} custom route`
    onImport({ family: pasteFamily, steps: parsed, routeId: 'PASTED', label })
  }

  return (
    <div className="modal-scrim" onClick={onClose}>
      <div className="modal glass" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <span className="d-eyebrow">Import sequence</span>
          <button className="modal-close" onClick={onClose} aria-label="Close">
            <svg width="13" height="13" viewBox="0 0 14 14" fill="none">
              <path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        <div className="modal-label">Sample routes</div>
        <div className="sample-grid">
          {SAMPLES.map((s) => (
            <button
              key={s.id}
              className={`sample${s.invalid ? ' is-faulty' : ''}`}
              onClick={() => onImport({ family: s.family, steps: s.steps, routeId: s.id, label: s.label })}
            >
              <span className="sample-name">{s.label}</span>
              <span className="sample-meta mono">
                {s.steps.length} steps{s.invalid ? ' · contains fault' : ''}
              </span>
            </button>
          ))}
        </div>

        <div className="modal-divider" />

        <div className="modal-label">Paste a partial sequence</div>
        <div className="paste-row">
          <input
            className="paste-name"
            type="text"
            value={routeName}
            onChange={(e) => setRouteName(e.target.value)}
            placeholder="Route name"
            aria-label="Custom sequence name"
          />
          <select className="paste-family" value={pasteFamily} onChange={(e) => setPasteFamily(e.target.value as Family)}>
            {FAMILIES.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
          <span className="paste-hint mono">steps separated by | or newlines</span>
        </div>
        <textarea
          className="paste-area mono"
          placeholder={'RECEIVE WAFER LOT | LOT IDENTIFICATION | PRE CLEAN WAFER | …'}
          value={text}
          onChange={(e) => setText(e.target.value)}
          spellCheck={false}
        />
        <div className="paste-actions">
          <span className="paste-count mono">{parsed.length} steps parsed</span>
          <button className="btn-primary sm" disabled={parsed.length < 2} onClick={importPasted}>
            Import &amp; predict
          </button>
        </div>
      </div>
    </div>
  )
}

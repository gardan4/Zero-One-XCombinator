import { useRef, type CSSProperties, type MouseEvent } from 'react'

interface Props {
  total: number // expected steps for the family = number of cells
  done: number // steps completed so far
  selectedIdx: number // currently selected step
  headIdx: number // last completed step index (max selectable)
  onSeek: (stepIdx: number) => void
}

export default function FullRouteRail({ total, done, selectedIdx, headIdx, onSeek }: Props) {
  const meterRef = useRef<HTMLDivElement>(null)
  const cells = Math.max(1, total)
  const filled = Math.min(done, cells)
  const fillPct = (filled / cells) * 100
  const selPct = ((Math.min(selectedIdx, headIdx) + 0.5) / cells) * 100
  const showSel = selectedIdx !== headIdx && selectedIdx >= 0 && selectedIdx <= headIdx

  // click anywhere on the bar to seek to that step (locked beyond the head)
  function handleClick(e: MouseEvent<HTMLDivElement>) {
    const el = meterRef.current
    if (!el) return
    const r = el.getBoundingClientRect()
    const idx = Math.floor(((e.clientX - r.left) / r.width) * cells)
    onSeek(Math.max(0, Math.min(idx, headIdx)))
  }

  const style = { '--cells': cells, '--fill': `${fillPct}%` } as CSSProperties

  return (
    <div className="flow-rail">
      <div className="rail-head">
        <span className="rail-title">Full route</span>
        <span className="rail-sub mono">
          RECEIVE WAFER LOT <b>→</b> SHIP LOT &nbsp;·&nbsp; {done} / ~{total} steps
        </span>
      </div>

      <div className="meter" ref={meterRef} style={style} onClick={handleClick}>
        <span className="m-fill" />
        <span className="m-seam" />
        {showSel && <span className="m-sel" style={{ left: `${selPct}%` }} />}
      </div>
    </div>
  )
}

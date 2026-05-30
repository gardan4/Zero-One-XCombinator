import type { CSSProperties } from 'react'
import type { RoadmapPhase } from '../lib/grammar'

interface Props {
  roadmap: RoadmapPhase[]
  headPhaseIndex: number
  expectedTotal: number
  currentCount: number
  currentPhaseDone: number
}

export default function FullRouteRail({ roadmap, headPhaseIndex, expectedTotal, currentCount, currentPhaseDone }: Props) {
  return (
    <div className="flow-rail">
      <div className="rail-head">
        <span className="rail-title">Full route</span>
        <span className="rail-sub mono">
          RECEIVE WAFER LOT <b>→</b> SHIP LOT &nbsp;·&nbsp; ~{expectedTotal} steps
        </span>
      </div>

      <div className="rail">
        {roadmap.map((p) => {
          const state = p.index < headPhaseIndex ? 'done' : p.index === headPhaseIndex ? 'current' : 'future'

          if (state === 'current') {
            // per-step meter: solid teal up to the steps done, faint split cells beyond.
            const expected = Math.max(1, Math.round(p.weight))
            const done = Math.max(0, currentPhaseDone)
            const cells = Math.max(expected, done, 1)
            const fillPct = Math.min(100, (done / cells) * 100)
            const pitch = cells > 22 ? 7 : 10
            const minWidth = cells * pitch + 14 // expand instead of crushing the splits
            const style = {
              flex: p.weight,
              minWidth: `${minWidth}px`,
              '--cells': cells,
              '--fill': `${fillPct}%`,
            } as CSSProperties

            return (
              <div key={p.index} className="seg current" style={style}>
                <span className="seg-fill" />
                <span className="seg-seam" />
                <span className="seg-name">
                  {p.name} <span className="seg-count mono">{currentCount}</span>
                </span>
              </div>
            )
          }

          return (
            <div key={p.index} className={`seg ${state}`} style={{ flex: p.weight }}>
              <span className="seg-name">{p.name}</span>
            </div>
          )
        })}
      </div>

      <div className="rail-scale">
        <span className="cap start mono">
          <span className="tk" />
          01
        </span>
        <span className="cap end mono">
          <span className="tk" />
          {expectedTotal}
        </span>
      </div>
    </div>
  )
}

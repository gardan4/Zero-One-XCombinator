import type { RoadmapPhase } from '../lib/grammar'

interface Props {
  roadmap: RoadmapPhase[]
  headPhaseIndex: number
  fraction: number
  expectedTotal: number
  currentCount: number
}

export default function FullRouteRail({ roadmap, headPhaseIndex, fraction, expectedTotal, currentCount }: Props) {
  const totalWeight = roadmap.reduce((a, p) => a + p.weight, 0) || 1
  const herePct = Math.min(99, Math.max(1, fraction * 100))

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
        <span className="here" style={{ left: `${herePct}%` }}>
          <span className="nib" />
          <span className="htk" />
          <span className="hnum mono">{currentCount}</span>
        </span>
        <span className="cap end mono">
          <span className="tk" />
          {expectedTotal}
        </span>
      </div>
    </div>
  )
}

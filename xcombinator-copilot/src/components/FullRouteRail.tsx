import { useEffect, useRef, type CSSProperties } from 'react'
import type { Phase, RoadmapPhase } from '../lib/grammar'

interface Props {
  roadmap: RoadmapPhase[]
  phases: Phase[]
  headPhaseIndex: number
  expectedTotal: number
  currentCount: number
}

export default function FullRouteRail({ roadmap, phases, headPhaseIndex, expectedTotal, currentCount }: Props) {
  const railRef = useRef<HTMLDivElement>(null)
  const curRef = useRef<HTMLDivElement>(null)

  // keep the current phase centred as the route advances and the row scrolls
  useEffect(() => {
    const wrap = railRef.current
    const el = curRef.current
    if (!wrap || !el) return
    const wr = wrap.getBoundingClientRect()
    const er = el.getBoundingClientRect()
    wrap.scrollBy({ left: er.left - wr.left - wrap.clientWidth * 0.4, behavior: 'smooth' })
  }, [headPhaseIndex, currentCount])

  const curRoadmap = roadmap.find((r) => r.index === headPhaseIndex)
  const future = roadmap.filter((r) => r.index > headPhaseIndex)

  return (
    <div className="flow-rail">
      <div className="rail-head">
        <span className="rail-title">Full route</span>
        <span className="rail-sub mono">
          RECEIVE WAFER LOT <b>→</b> SHIP LOT &nbsp;·&nbsp; ~{expectedTotal} steps expected
        </span>
      </div>

      <div className="rail" ref={railRef}>
        {phases.map((p) => {
          // past + current phases: real, observed step counts
          if (p.index === headPhaseIndex) {
            const expected = Math.max(1, Math.round(curRoadmap?.weight ?? p.steps.length))
            const done = p.steps.length
            const cells = Math.max(expected, done, 1)
            const fillPct = Math.min(100, (done / cells) * 100)
            const pitch = cells > 22 ? 7 : 10
            const style = {
              minWidth: `${cells * pitch + 14}px`,
              '--cells': cells,
              '--fill': `${fillPct}%`,
            } as CSSProperties
            return (
              <div key={p.index} className="rc current" ref={curRef} style={style}>
                <span className="rc-fill" />
                <span className="rc-seam" />
                <span className="rc-name">{p.name}</span>
                <span className="rc-meta mono">{currentCount}</span>
              </div>
            )
          }
          return (
            <div key={p.index} className="rc done">
              <span className="rc-name">{p.name}</span>
              <span className="rc-meta mono">{p.steps.length}</span>
            </div>
          )
        })}

        {/* future phases: expected sequence, estimated counts */}
        {future.map((f) => (
          <div key={f.index} className="rc future">
            <span className="rc-name">{f.name}</span>
            <span className="rc-meta mono">~{f.weight}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

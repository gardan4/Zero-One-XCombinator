import { useEffect, useMemo, useRef, useState } from 'react'
import type { Family, Prediction } from './types'
import { defaultSample, describe, expectedLength } from './lib/data'
import { segmentRoute, buildRoadmap, categoryOf } from './lib/grammar'
import { validateRoute, ruleForStep } from './lib/rules'
import { predictNextStep, LIVE } from './lib/model'
import TopBar from './components/TopBar'
import Results from './components/Results'
import ProcessRoute from './components/ProcessRoute'
import FullRouteRail from './components/FullRouteRail'
import StepDetail from './components/StepDetail'
import ControlBar from './components/ControlBar'
import ImportDialog, { type ImportPayload } from './components/ImportDialog'

export default function App() {
  const init = defaultSample('MOSFET')
  const [family, setFamily] = useState<Family>('MOSFET')
  const [steps, setSteps] = useState<string[]>(init.steps)
  const [routeLabel, setRouteLabel] = useState<string>(init.label)
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [predicting, setPredicting] = useState(false)
  const [selOverride, setSelOverride] = useState<number | null>(null) // null = follow head
  const [expOverride, setExpOverride] = useState<number | null>(null) // null = follow head phase
  const [importOpen, setImportOpen] = useState(false)
  const [view, setView] = useState<'copilot' | 'results'>('copilot')
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    try {
      return (localStorage.getItem('xc-theme') as 'dark' | 'light') || 'dark'
    } catch {
      return 'dark'
    }
  })
  const token = useRef(0)

  const phases = useMemo(() => segmentRoute(steps), [steps])
  const violations = useMemo(() => validateRoute(steps), [steps])
  const roadmap = useMemo(() => buildRoadmap(family), [family])

  const headIdx = steps.length - 1
  const headPhase = phases[phases.length - 1]
  const headPhaseIndex = headPhase?.index ?? 0
  const expandedIndex = expOverride ?? headPhaseIndex
  const selectedIdx = Math.min(selOverride ?? headIdx, steps.length - 1)
  const complete = steps[headIdx] === 'SHIP LOT'
  const total = expectedLength(family)

  function runPredict(fam: Family, seq: string[]) {
    const t = ++token.current
    setPredicting(true)
    setPrediction(null)
    predictNextStep(fam, seq).then((p) => {
      if (t === token.current) {
        setPrediction(p)
        setPredicting(false)
      }
    })
  }

  // first prediction on mount
  useEffect(() => {
    runPredict(family, steps)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // apply + persist the theme
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    try {
      localStorage.setItem('xc-theme', theme)
    } catch {
      /* ignore */
    }
  }, [theme])

  function applyRoute(fam: Family, seq: string[], label: string) {
    setFamily(fam)
    setSteps(seq)
    setRouteLabel(label)
    setSelOverride(null)
    setExpOverride(null)
    setImportOpen(false)
    runPredict(fam, seq)
  }

  function onFamily(f: Family) {
    if (f === family) return
    const s = defaultSample(f)
    applyRoute(f, s.steps, s.label)
  }

  function onImport(p: ImportPayload) {
    const label = p.routeId === 'PASTED' ? 'Pasted route' : p.label ?? `${p.family} route`
    applyRoute(p.family, p.steps, label)
  }

  function advance() {
    if (!prediction || complete) return
    const next = [...steps, prediction.step]
    setSteps(next)
    setSelOverride(null)
    setExpOverride(null)
    runPredict(family, next)
  }

  function onExpandPhase(phaseIndex: number) {
    setExpOverride(phaseIndex)
    const p = phases.find((x) => x.index === phaseIndex)
    if (p) setSelOverride(p.steps[p.steps.length - 1].idx)
  }

  const selStep = steps[selectedIdx]
  const selPhase = phases.find((p) => p.steps.some((s) => s.idx === selectedIdx))
  const stepRule = useMemo(() => ruleForStep(steps, selectedIdx, violations), [steps, selectedIdx, violations])
  const detailFraction = Math.min(1, (selectedIdx + 1) / total)

  return (
    <div className="app">
      <div className="grid-lines" />
      <div className="grid-dots" />
      <div className="bloom" />
      <span className="reg-tick tl" />
      <span className="reg-tick tr" />
      <span className="reg-tick bl" />
      <span className="reg-tick br" />

      <TopBar
        family={family}
        onFamily={onFamily}
        onImport={() => setImportOpen(true)}
        live={LIVE}
        theme={theme}
        onToggleTheme={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
      />

      <div className="rs-nav-fixed"><div className="rs-nav">
        <button className={view === 'copilot' ? 'on' : ''} onClick={() => setView('copilot')}>Copilot</button>
        <button className={view === 'results' ? 'on' : ''} onClick={() => setView('results')}>Results</button>
      </div></div>
      {view === 'results' && <div className="rs-overlay"><Results /></div>}

      <div className="workspace">
        <div className="left-col">
          <div className="flow-canvas glass">
            <div className="flow-head">
              <div className="title">Process Route</div>
              <div className="route-id mono">
                {routeLabel} &nbsp;·&nbsp; <b>{steps.length}</b> / ~{total} steps
              </div>
            </div>

            <ProcessRoute
              phases={phases}
              headIdx={headIdx}
              headPhaseIndex={headPhaseIndex}
              expandedIndex={expandedIndex}
              selectedIdx={selectedIdx}
              prediction={prediction}
              predicting={predicting}
              roadmap={roadmap}
              violations={violations}
              onSelectStep={(i) => setSelOverride(i)}
              onExpandPhase={onExpandPhase}
            />

            <FullRouteRail
              roadmap={roadmap}
              headPhaseIndex={headPhaseIndex}
              expectedTotal={total}
              currentCount={steps.length}
              currentPhaseDone={headPhase?.steps.length ?? 0}
            />
          </div>

          <ControlBar onPredict={advance} predicting={predicting} complete={complete} violations={violations} />
        </div>

        <StepDetail
          step={selStep}
          idx={selectedIdx}
          total={total}
          phaseName={selPhase?.name ?? '—'}
          category={categoryOf(selStep)}
          description={describe(selStep)}
          prediction={prediction}
          rule={stepRule}
          fraction={detailFraction}
          isHead={selectedIdx === headIdx}
          complete={complete}
        />
      </div>

      {importOpen && <ImportDialog family={family} onClose={() => setImportOpen(false)} onImport={onImport} />}
    </div>
  )
}

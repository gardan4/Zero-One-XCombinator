import type { Family } from '../types'
import { fab } from './data'
import { ELECTRICAL_TEST_STEPS } from './rules'

// ---------------------------------------------------------------------------
// Macro-phase model — the named blocks of a fab route, in canonical order.
// A monotonic state machine walks the steps and assigns each to a phase, so a
// 100+ step route collapses into a handful of legible, evenly-named groups.
// ---------------------------------------------------------------------------

export const PHASE_ORDER = [
  'Receive & Inspect',
  'Clean',
  'Family Prep',
  'Oxidation',
  'Litho Cycles',
  'Dielectric & CMP',
  'Via',
  'Metal',
  'Passivation',
  'Backside',
  'Final & Test',
  'Ship',
] as const

export type PhaseName = (typeof PHASE_ORDER)[number]

export interface Phase {
  name: PhaseName
  index: number // position in PHASE_ORDER
  startIdx: number // first step index in the route
  steps: { step: string; idx: number }[]
  repeats?: number // for Litho Cycles: number of litho blocks inside
}

const FAMILY_PREP_ENTRY = new Set([
  'SUBSTRATE CHECK', 'EPITAXY PREP', 'EPITAXIAL WAFER CHECK', 'EPITAXIAL LAYER PREP',
  'WAFER CLEAN PRE-GRIND', 'GRINDING WAFER BACKSIDE',
])
const VIA_ENTRY = new Set([
  'VIA ETCH', 'VIA ETCH THROUGH DIELECTRIC', 'DIELECTRIC ETCH VIA',
  'VIA INSPECTION', 'VIA OPENING INSPECTION', 'MEASURE VIA CD',
  'FILL VIA METAL', 'FILL VIA TUNGSTEN',
])
const BACKSIDE_ENTRY = new Set([
  'BACKSIDE GRIND', 'BACKSIDE THINNING CHECK', 'DEPOSIT BACKSIDE PROTECTION',
  'DEPOSIT BACKSIDE METAL', 'BACKSIDE METALLIZATION PREP', 'BACKSIDE ETCH CLEAN',
  'BACKSIDE ANNEAL', 'MEASURE WAFER THICKNESS', 'BACKSIDE CLEAN FINAL',
  'BACKSIDE RINSE', 'BACKSIDE DRY',
])
const SHIP_ENTRY = new Set(['LOT RELEASE', 'FINAL LOT RELEASE', 'PACKAGE PREPARATION', 'SHIP LOT'])

function entryCandidates(step: string, cur: number): number[] {
  const c: number[] = []
  if (step === 'RECEIVE WAFER LOT' || step === 'LOT IDENTIFICATION') c.push(0)
  if (step === 'PRE CLEAN WAFER' || step === 'WAFER CLEAN PRE PROCESS') c.push(1)
  if (FAMILY_PREP_ENTRY.has(step)) c.push(2)
  if (step === 'THERMAL OXIDATION') c.push(3)
  if (step === 'SPIN COAT PHOTORESIST') {
    c.push(4)
    if (cur === 5) c.push(6) // first litho after ILD = the via litho
  }
  if (step === 'DEPOSIT INTERLAYER DIELECTRIC' || step === 'DEPOSIT INTERLEVEL DIELECTRIC') c.push(5)
  if (VIA_ENTRY.has(step)) c.push(6)
  if (step === 'DEPOSIT METAL 1' || step === 'DEPOSIT TOP METAL') c.push(7)
  if (step === 'DEPOSIT PASSIVATION' || step === 'DEPOSIT PASSIVATION LAYER') c.push(8)
  if (BACKSIDE_ENTRY.has(step) && cur >= 8) c.push(9)
  if ((step.startsWith('FINAL') || ELECTRICAL_TEST_STEPS.has(step) || step === 'WAFER SORT TEST' || step === 'YIELD ANALYSIS') && cur >= 7) c.push(10)
  if (SHIP_ENTRY.has(step)) c.push(11)
  return c
}

/** Segment a route into ordered, contiguous macro-phases (only those present). */
export function segmentRoute(steps: string[]): Phase[] {
  const phases: Phase[] = []
  let cur = 0
  let active: Phase | null = null

  steps.forEach((step, i) => {
    const cands = entryCandidates(step, cur).filter((p) => p > cur)
    if (cands.length) cur = Math.max(...cands)
    if (!active || active.index !== cur) {
      active = { name: PHASE_ORDER[cur], index: cur, startIdx: i, steps: [] }
      phases.push(active)
    }
    active.steps.push({ step, idx: i })
  })

  for (const p of phases) {
    if (p.name === 'Litho Cycles') {
      p.repeats = p.steps.filter((s) => s.step === 'SPIN COAT PHOTORESIST').length || undefined
    }
  }
  return phases
}

// ---------------------------------------------------------------------------
// Roadmap — the expected phase sequence + proportional weights for a family,
// derived from the reference corpus so the full-route rail is data-accurate.
// ---------------------------------------------------------------------------

export interface RoadmapPhase {
  name: PhaseName
  index: number
  weight: number // average step count
}

const roadmapCache = new Map<Family, RoadmapPhase[]>()

export function buildRoadmap(family: Family): RoadmapPhase[] {
  const cached = roadmapCache.get(family)
  if (cached) return cached

  const routes = fab.corpus[family] ?? []
  const sums = new Map<number, { total: number; n: number }>()
  for (const route of routes) {
    const counts = new Map<number, number>()
    for (const p of segmentRoute(route)) counts.set(p.index, (counts.get(p.index) ?? 0) + p.steps.length)
    for (const [idx, cnt] of counts) {
      const s = sums.get(idx) ?? { total: 0, n: 0 }
      s.total += cnt
      s.n += 1
      sums.set(idx, s)
    }
  }
  const roadmap: RoadmapPhase[] = [...sums.entries()]
    .filter(([, s]) => s.n >= routes.length / 2)
    .map(([idx, s]) => ({ name: PHASE_ORDER[idx], index: idx, weight: Math.max(1, Math.round(s.total / s.n)) }))
    .sort((a, b) => a.index - b.index)

  roadmapCache.set(family, roadmap)
  return roadmap
}

// ---------------------------------------------------------------------------
// Category — a coarse functional bucket per step, for the detail panel tag.
// ---------------------------------------------------------------------------

export type Category =
  | 'Lithography' | 'Etch' | 'Strip' | 'Clean' | 'Implant & Diffusion'
  | 'Thermal & Anneal' | 'Deposition' | 'Planarization' | 'Via' | 'Passivation'
  | 'Backside' | 'Metrology' | 'Electrical Test' | 'Logistics'

export function categoryOf(step: string): Category {
  const s = step
  if (/\b(LOT|SHIP|RELEASE|PACKAGE)\b/.test(s) || s === 'LOT IDENTIFICATION' || s === 'RECEIVE WAFER LOT') return 'Logistics'
  if (s.includes('PASSIVATION') || s.includes('PAD WINDOW') || s === 'CURE PASSIVATION') return 'Passivation'
  if (s.includes('BACKSIDE') || s.includes('GRIND')) return 'Backside'
  if (s.startsWith('CMP') || s.includes('PLANAR')) return 'Planarization'
  if (s.includes('VIA')) return 'Via'
  if (s.startsWith('SPIN COAT') || s.includes('BAKE') || s.includes('MASK LEVEL') || s.startsWith('EXPOSE LITHO') || s === 'DEVELOP PHOTORESIST' || s === 'DEVELOP PAD WINDOW' || s.includes('PATTERN INSPECTION') || s.includes('PATTERN LEVEL')) return 'Lithography'
  if (s.startsWith('STRIP')) return 'Strip'
  if (s.includes('ETCH')) return 'Etch'
  if (s.startsWith('IMPLANT') || s.includes('DIFFUSION') || s.includes('DRIVE IN')) return 'Implant & Diffusion'
  if (s.startsWith('DEPOSIT') || s.startsWith('FILL VIA') || s === 'EPITAXIAL DEPOSITION') return 'Deposition'
  if (s.includes('OXIDATION') || s.includes('ANNEAL') || s.includes('OXIDE GROWTH') || s.includes('GATE OXIDE') || s.includes('DENSIFY')) return 'Thermal & Anneal'
  if (ELECTRICAL_TEST_STEPS.has(s) || s === 'WAFER SORT TEST') return 'Electrical Test'
  if (s.includes('CLEAN') || s.includes('RCA') || s === 'HF DIP' || s.includes('DRY WAFER') || s.includes('RINSE') || s.includes('SURFACE PREP')) return 'Clean'
  if (s.startsWith('MEASURE') || s.includes('INSPECT') || s.includes('CHECK') || s.includes('ANALYSIS')) return 'Metrology'
  return 'Metrology'
}

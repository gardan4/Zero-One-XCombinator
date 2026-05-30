import type { Violation } from '../types'

// ---------------------------------------------------------------------------
// Vocabulary sets — ported verbatim from the track's generate_sequences.py so
// the dashboard's anomaly check matches the official validator step-for-step.
// ---------------------------------------------------------------------------

export const DEPOSITION_STEPS = new Set([
  'THERMAL OXIDATION', 'GATE OXIDE GROWTH', 'DEPOSIT PAD OXIDE', 'EPITAXIAL DEPOSITION',
  'DEPOSIT POLYSILICON', 'DEPOSIT SPACER DIELECTRIC', 'DEPOSIT FIELD OXIDE',
  'DEPOSIT GATE OXIDE OR DIELECTRIC', 'DEPOSIT INTERLAYER DIELECTRIC', 'DEPOSIT INTERLEVEL DIELECTRIC',
  'DEPOSIT BARRIER METAL', 'DEPOSIT METAL SEED', 'DEPOSIT METAL 1', 'DEPOSIT TOP METAL',
  'DEPOSIT BACKSIDE METAL', 'DEPOSIT TUNGSTEN SEED', 'DEPOSIT PASSIVATION',
  'DEPOSIT PASSIVATION LAYER', 'DEPOSIT BACKSIDE PROTECTION',
])

export const CLEAN_STEPS = new Set([
  'PRE CLEAN WAFER', 'WAFER CLEAN PRE PROCESS', 'WAFER SURFACE CLEAN',
  'RCA CLEAN 1', 'RCA CLEAN 2', 'WET CLEAN RCA1', 'WET CLEAN RCA2',
  'HF DIP', 'OXIDE STRIP', 'SURFACE PREP FOR DEPOSITION',
  'FRONTSIDE CLEAN', 'BACKSIDE CLEAN', 'FRONTSIDE CLEAN FINAL',
  'BACKSIDE CLEAN FINAL', 'WAFER CLEAN PRE-GRIND',
  'DRY WAFER', 'DRY WAFER BACKSIDE',
  'CLEAN AFTER ETCH', 'CLEAN AFTER OXIDE ETCH', 'CLEAN AFTER POLY ETCH',
  'CLEAN AFTER VIA ETCH', 'CLEAN AFTER METAL ETCH',
  'CLEAN AFTER WINDOW ETCH', 'CLEAN AFTER FIELD ETCH',
  'CLEAN PAD OPENING', 'BACKSIDE ETCH CLEAN', 'BACKSIDE RINSE',
  'THERMAL OXIDATION', 'GATE OXIDE PREP', 'RAPID THERMAL ANNEAL',
  'EPITAXY ANNEAL', 'ANNEAL OXIDE',
])

export const ETCH_STEPS = new Set([
  'OXIDE ETCH', 'OXIDE ETCH DRY', 'POLYSILICON ETCH', 'POLYSILICON ETCH DRY',
  'ETCH SILICON OR OXIDE WINDOW', 'FIELD OXIDE ETCH',
  'VIA ETCH', 'VIA ETCH THROUGH DIELECTRIC', 'DIELECTRIC ETCH VIA',
  'METAL ETCH', 'METAL ETCH DRY', 'PASSIVATION ETCH PAD OPENING', 'PASSIVATION ETCH',
])

export const METAL_ETCH_STEPS = new Set(['METAL ETCH', 'METAL ETCH DRY'])

export const IMPLANT_STEPS = new Set([
  'IMPLANT WELL', 'IMPLANT SOURCE DRAIN', 'IMPLANT SOURCE REGION', 'IMPLANT LDD',
  'IMPLANT P BODY', 'IMPLANT N BUFFER', 'IMPLANT CHANNEL STOP',
  'IMPLANT DRAIN / CATHODE REGION', 'IMPLANT N-TYPE',
])

export const IMPLANT_OPENER_STEPS = new Set([
  'OXIDE ETCH', 'OXIDE ETCH DRY', 'ETCH SILICON OR OXIDE WINDOW', 'DEVELOP PHOTORESIST',
])

export const CMP_STEPS = new Set([
  'CMP DIELECTRIC', 'CMP INTERLAYER DIELECTRIC', 'CMP METAL', 'CMP VIA FILL',
])

export const FILL_STEPS = new Set([
  'FILL VIA METAL', 'FILL VIA TUNGSTEN', ...DEPOSITION_STEPS,
])

export const PAD_WINDOW_STEPS = new Set([
  'OPEN PAD WINDOW', 'OPEN BOND PAD WINDOW', 'PAD WINDOW LITHO', 'OPEN PAD WINDOW LITHO',
])

export const ELECTRICAL_TEST_STEPS = new Set([
  'PARAMETRIC TEST', 'ELECTRICAL PARAMETRIC TEST', 'THRESHOLD VOLTAGE TEST',
  'BREAKDOWN VOLTAGE TEST', 'LEAKAGE TEST', 'SWITCHING TEST',
])

export const BACKSIDE_METAL_STEPS = new Set(['DEPOSIT BACKSIDE METAL'])

/** One-line, plain-English summary per rule (for legends and tooltips). */
export const RULE_SHORT: Record<string, string> = {
  RULE_DEP_NO_CLEAN: 'A deposition needs a clean surface within the prior 12 steps.',
  RULE_METAL_ETCH_NO_LITHO: 'Metal etch needs an expose + develop mask within the prior 15 steps.',
  RULE_ETCH_NO_MASK: 'A patterned etch needs a developed photoresist mask within the prior 12 steps.',
  RULE_LITHO_LEVEL_SKIP: 'Mask levels must run in sequential order, no skips.',
  RULE_IMPLANT_NO_MASK: 'An implant needs an open window (etch or develop) within the prior 15 steps.',
  RULE_CMP_NO_DEP: 'CMP needs deposited/filled material within the prior 6 steps.',
  RULE_PAD_OPEN_BEFORE_DEP: 'Pad windows open only after passivation is deposited and cured.',
  RULE_TEST_BEFORE_PASSIVATION: 'Electrical tests run only after passivation is cured.',
  RULE_SHIP_BEFORE_TEST: 'A lot ships only after wafer sort test.',
  RULE_BACKSIDE_BEFORE_PASSIVATION: 'Backside metal goes on only after the front is passivated.',
}

// ---------------------------------------------------------------------------
// validateRoute — mirrors validate_sequence(); empty array means valid.
// ---------------------------------------------------------------------------

export function validateRoute(steps: string[]): Violation[] {
  const out: Violation[] = []
  const win = (i: number, size: number) => steps.slice(Math.max(0, i - size), i)
  const anyIn = (i: number, size: number, set: Set<string>) => win(i, size).some((s) => set.has(s))

  steps.forEach((step, i) => {
    if (DEPOSITION_STEPS.has(step) && !anyIn(i, 12, CLEAN_STEPS)) {
      out.push(v('RULE_DEP_NO_CLEAN', `Deposition step '${step}' has no clean step in the prior 12 steps. A clean surface is required before any deposition.`, i, step))
    }
    if (METAL_ETCH_STEPS.has(step)) {
      const w = win(i, 15)
      const hasExpose = w.some((s) => s.startsWith('EXPOSE LITHO LEVEL'))
      const hasDevelop = w.includes('DEVELOP PHOTORESIST') || w.includes('DEVELOP PAD WINDOW')
      if (!(hasExpose && hasDevelop)) {
        out.push(v('RULE_METAL_ETCH_NO_LITHO', `Metal etch '${step}' is missing EXPOSE LITHO LEVEL or DEVELOP PHOTORESIST in the prior 15 steps. Metal cannot be etched without a photoresist mask.`, i, step))
      }
    }
    if (ETCH_STEPS.has(step)) {
      const w = win(i, 12)
      if (!(w.includes('DEVELOP PHOTORESIST') || w.includes('DEVELOP PAD WINDOW'))) {
        out.push(v('RULE_ETCH_NO_MASK', `Etch step '${step}' has no DEVELOP PHOTORESIST in the prior 12 steps. A photoresist mask must be patterned before etching.`, i, step))
      }
    }
    if (IMPLANT_STEPS.has(step) && !anyIn(i, 15, IMPLANT_OPENER_STEPS)) {
      out.push(v('RULE_IMPLANT_NO_MASK', `Implant step '${step}' has no oxide etch or DEVELOP PHOTORESIST in the prior 15 steps. An open implant window is required.`, i, step))
    }
    if (CMP_STEPS.has(step) && !anyIn(i, 6, FILL_STEPS)) {
      out.push(v('RULE_CMP_NO_DEP', `CMP step '${step}' has no deposition or fill step in the prior 6 steps. There must be material to planarize.`, i, step))
    }
  })

  // RULE_LITHO_LEVEL_SKIP — mask levels must be non-decreasing and step by one.
  const aligns: [number, number][] = []
  steps.forEach((s, i) => {
    if (s.startsWith('ALIGN MASK LEVEL ')) {
      const n = s.slice('ALIGN MASK LEVEL '.length)
      if (/^\d+$/.test(n)) aligns.push([i, parseInt(n, 10)])
    }
  })
  for (let k = 1; k < aligns.length; k++) {
    const [pi, pl] = aligns[k - 1]
    const [ci, cl] = aligns[k]
    if (cl > pl + 1) out.push(v('RULE_LITHO_LEVEL_SKIP', `Litho level jumps from ${pl} (step ${pi}) to ${cl} (step ${ci}), skipping level ${pl + 1}.`, ci, steps[ci]))
    if (cl < pl) out.push(v('RULE_LITHO_LEVEL_SKIP', `Litho level decreases from ${pl} (step ${pi}) to ${cl} (step ${ci}). Levels must be non-decreasing.`, ci, steps[ci]))
  }

  // RULE_PAD_OPEN_BEFORE_DEP
  let passDep: number | null = null
  let curePass: number | null = null
  steps.forEach((step, i) => {
    if (step === 'DEPOSIT PASSIVATION' || step === 'DEPOSIT PASSIVATION LAYER') passDep = i
    if (step === 'CURE PASSIVATION') curePass = i
    if (PAD_WINDOW_STEPS.has(step)) {
      if (passDep === null || i < passDep) {
        out.push(v('RULE_PAD_OPEN_BEFORE_DEP', `Pad window step '${step}' at index ${i} appears before DEPOSIT PASSIVATION (index ${passDep}). You cannot open a window in passivation that has not been deposited.`, i, step))
      } else if (curePass === null || i < curePass) {
        out.push(v('RULE_PAD_OPEN_BEFORE_DEP', `Pad window step '${step}' at index ${i} appears before CURE PASSIVATION (index ${curePass}). Passivation must be cured before the pad window is opened.`, i, step))
      }
    }
  })

  const cureIdx = idxOf(steps, 'CURE PASSIVATION')

  // RULE_TEST_BEFORE_PASSIVATION
  steps.forEach((step, i) => {
    if (ELECTRICAL_TEST_STEPS.has(step) && (cureIdx === null || i < cureIdx)) {
      out.push(v('RULE_TEST_BEFORE_PASSIVATION', `Electrical test '${step}' at index ${i} appears before CURE PASSIVATION (index ${cureIdx}). Devices must be passivated before electrical characterization.`, i, step))
    }
  })

  // RULE_SHIP_BEFORE_TEST
  const shipIdx = idxOf(steps, 'SHIP LOT')
  const sortIdx = idxOf(steps, 'WAFER SORT TEST')
  if (shipIdx !== null && (sortIdx === null || shipIdx < sortIdx)) {
    out.push(v('RULE_SHIP_BEFORE_TEST', `SHIP LOT at index ${shipIdx} appears before WAFER SORT TEST (index ${sortIdx}). Lots must pass sort testing before they can be shipped.`, shipIdx, 'SHIP LOT'))
  }

  // RULE_BACKSIDE_BEFORE_PASSIVATION
  steps.forEach((step, i) => {
    if (BACKSIDE_METAL_STEPS.has(step) && (cureIdx === null || i < cureIdx)) {
      out.push(v('RULE_BACKSIDE_BEFORE_PASSIVATION', `'${step}' at index ${i} appears before CURE PASSIVATION (index ${cureIdx}). The frontside must be passivated before backside metallization.`, i, step))
    }
  })

  return out
}

function v(rule: string, description: string, stepIndex: number, stepName: string): Violation {
  return { rule, description, stepIndex, stepName }
}

function idxOf(steps: string[], target: string): number | null {
  const i = steps.indexOf(target)
  return i === -1 ? null : i
}

// ---------------------------------------------------------------------------
// Per-step governing rule — what process-logic constraint applies to a step,
// and whether it is currently satisfied. Drives the detail panel's logic line.
// ---------------------------------------------------------------------------

export interface StepRule {
  rule: string
  satisfied: boolean
  text: string
}

export function ruleForStep(steps: string[], i: number, violations: Violation[]): StepRule | null {
  const step = steps[i]
  const violated = violations.find((x) => x.stepIndex === i)
  let rule: string | null = null

  if (DEPOSITION_STEPS.has(step)) rule = 'RULE_DEP_NO_CLEAN'
  else if (METAL_ETCH_STEPS.has(step)) rule = 'RULE_METAL_ETCH_NO_LITHO'
  else if (ETCH_STEPS.has(step)) rule = 'RULE_ETCH_NO_MASK'
  else if (IMPLANT_STEPS.has(step)) rule = 'RULE_IMPLANT_NO_MASK'
  else if (CMP_STEPS.has(step)) rule = 'RULE_CMP_NO_DEP'
  else if (ELECTRICAL_TEST_STEPS.has(step)) rule = 'RULE_TEST_BEFORE_PASSIVATION'
  else if (PAD_WINDOW_STEPS.has(step)) rule = 'RULE_PAD_OPEN_BEFORE_DEP'
  else if (step === 'SHIP LOT') rule = 'RULE_SHIP_BEFORE_TEST'
  else if (BACKSIDE_METAL_STEPS.has(step)) rule = 'RULE_BACKSIDE_BEFORE_PASSIVATION'
  else if (step.startsWith('ALIGN MASK LEVEL')) rule = 'RULE_LITHO_LEVEL_SKIP'

  if (violated) {
    return { rule: violated.rule, satisfied: false, text: violated.description }
  }
  if (rule) {
    return { rule, satisfied: true, text: `${RULE_SHORT[rule]} Satisfied here.` }
  }
  return null
}

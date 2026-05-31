import { ELECTRICAL_TEST_STEPS } from './rules'

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

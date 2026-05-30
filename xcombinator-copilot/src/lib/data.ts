import fabRaw from '../data/fab.json'
import type { Family } from '../types'

export interface Sample {
  id: string
  family: Family
  fraction: number
  label: string
  steps: string[]
  invalid: boolean
  expectViolation?: string | null
}

interface Fab {
  families: Family[]
  corpus: Record<Family, string[][]>
  samples: Sample[]
  descriptions: Record<string, string>
  vocab: string[]
  expectedLength: Record<Family, number>
}

export const fab = fabRaw as unknown as Fab
export const FAMILIES: Family[] = fab.families
export const VOCAB: string[] = fab.vocab
export const SAMPLES: Sample[] = fab.samples

export function describe(step: string): string | null {
  return fab.descriptions[step] ?? null
}

export function expectedLength(family: Family): number {
  return fab.expectedLength[family] ?? 125
}

export function defaultSample(family: Family): Sample {
  return SAMPLES.find((s) => s.family === family && !s.invalid) ?? SAMPLES[0]
}

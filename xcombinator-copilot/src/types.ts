export type Family = 'MOSFET' | 'IGBT' | 'IC'

/** A single next-step prediction from the model (or the local simulator). */
export interface Prediction {
  step: string
  confidence: number
  /** where it came from, e.g. "4-gram" (simulated) or "model" (served checkpoint) */
  source: string
}

/** A process-logic rule violation, mirroring the track's validate_sequence(). */
export interface Violation {
  rule: string
  description: string
  stepIndex: number
  stepName: string
}

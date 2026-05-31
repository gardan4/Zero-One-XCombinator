export type Family = 'MOSFET' | 'IGBT' | 'IC'

/** A single next-step prediction from the model (or the local simulator). */
export interface Prediction {
  step: string
  confidence: number
  /** false when the source can't report a real confidence (e.g. a hosted model without logprobs) */
  confidenceKnown?: boolean
  /** where it came from, e.g. "4-gram" (simulated) or a served model id */
  source: string
  /** the model's own chain-of-thought, when it emits one (rich for DeepSeek, empty for the fine-tune) */
  reasoning?: string
  /** ranked alternate next steps for this same position */
  alternates?: string[]
}

/** A process-logic rule violation, mirroring the track's validate_sequence(). */
export interface Violation {
  rule: string
  description: string
  stepIndex: number
  stepName: string
}

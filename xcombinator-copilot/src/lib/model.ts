import type { Family, Prediction } from '../types'
import { fab, VOCAB } from './data'

// ===========================================================================
// THE MODEL SEAM.
//
// predictNextStep() is the ONE place the dashboard asks "what comes next?".
// Today it runs a local back-off n-gram over a handful of reference routes
// (honest, deterministic, no network) and is labelled "Simulated".
//
// To go live on the team's served checkpoint, set two env vars and nothing
// else changes — the UI, the contract, the shapes all stay identical:
//
//   VITE_MODEL_BASE_URL=http://localhost:8001/v1     # vLLM OpenAI-compatible
//   VITE_MODEL_NAME=XCombinator/sft-fab-all
//
// ===========================================================================

const BASE = import.meta.env.VITE_MODEL_BASE_URL as string | undefined
const MODEL = (import.meta.env.VITE_MODEL_NAME as string) || 'default'
const API_KEY = (import.meta.env.VITE_MODEL_API_KEY as string) || 'EMPTY'

export const LIVE = Boolean(BASE)

/** Optional default model id (VITE_MODEL_NAME), exposed so the UI can prefer it in the picker. */
export const DEFAULT_MODEL = (import.meta.env.VITE_MODEL_NAME as string) || ''

/**
 * List the model ids the local server has loaded (GET /v1/models).
 * Returns [] on any error or when VITE_MODEL_BASE_URL is unset, so the UI can fall back to the
 * "Simulated (no server)" state without throwing.
 */
export async function listModels(): Promise<string[]> {
  if (!BASE) return []
  try {
    const res = await fetch(`${BASE.replace(/\/$/, '')}/models`)
    const json = await res.json()
    const data: unknown = json?.data
    if (!Array.isArray(data)) return []
    return data.map((m) => (m as { id?: string })?.id).filter((id): id is string => Boolean(id))
  } catch {
    return []
  }
}

export async function predictNextStep(family: Family, steps: string[], modelName?: string): Promise<Prediction | null> {
  if (steps.length === 0 || steps[steps.length - 1] === 'SHIP LOT') return null

  if (LIVE) {
    try {
      return await predictLive(family, steps, modelName)
    } catch {
      // network/model hiccup: fall through to the local simulator so the demo never stalls
    }
  }

  await delay(280) // a beat of "thinking" so the prediction reads as inference, not a lookup
  return predictNgram(family, steps)
}

// ---------------------------------------------------------------------------
// Live path — the served model, OpenAI-compatible chat (matches zo_common.llm)
// ---------------------------------------------------------------------------

async function predictLive(family: Family, steps: string[], modelName?: string): Promise<Prediction> {
  const prompt = `Product family: ${family}\nProcess so far: ${steps.join(' | ')}\n\nNext process step?`
  const res = await fetch(`${BASE!.replace(/\/$/, '')}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${API_KEY}` },
    body: JSON.stringify({
      model: modelName || MODEL,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0,
      max_tokens: 24,
    }),
  })
  const json = await res.json()
  const raw: string = json?.choices?.[0]?.message?.content ?? ''
  // The local server returns the model's real per-step confidence (geometric mean of the
  // greedy tokens' probabilities). Read it; fall back only if an older server omits it.
  const conf = typeof json?.confidence === 'number' ? Math.max(0.05, Math.min(0.99, json.confidence)) : 0.82
  return { step: snapToVocab(raw), confidence: conf, source: 'model' }
}

function snapToVocab(raw: string): string {
  const t = raw.split('|')[0].trim().replace(/[.\s]+$/, '')
  if (!t) return raw.trim()
  const up = t.toUpperCase()
  const exact = VOCAB.find((v) => v === up)
  if (exact) return exact
  const contained = VOCAB.filter((v) => up.includes(v)).sort((a, b) => b.length - a.length)[0]
  if (contained) return contained
  const within = VOCAB.filter((v) => v.includes(up)).sort((a, b) => a.length - b.length)[0]
  return within ?? t
}

// ---------------------------------------------------------------------------
// Simulated path — back-off n-gram over the reference corpus
// ---------------------------------------------------------------------------

const MAXK = 6
type Gram = Map<string, Map<string, number>>
const indexCache = new Map<Family, Gram[]>()

function indexFor(family: Family): Gram[] {
  const cached = indexCache.get(family)
  if (cached) return cached
  const grams: Gram[] = Array.from({ length: MAXK + 1 }, () => new Map())
  for (const route of fab.corpus[family] ?? []) {
    for (let i = 0; i < route.length - 1; i++) {
      const next = route[i + 1]
      for (let k = 1; k <= MAXK; k++) {
        if (i - k + 1 < 0) break
        const ctx = route.slice(i - k + 1, i + 1).join('')
        const m = grams[k]
        const counts = m.get(ctx) ?? new Map<string, number>()
        counts.set(next, (counts.get(next) ?? 0) + 1)
        m.set(ctx, counts)
      }
    }
  }
  indexCache.set(family, grams)
  return grams
}

function predictNgram(family: Family, steps: string[]): Prediction {
  const grams = indexFor(family)
  for (let k = Math.min(MAXK, steps.length); k >= 1; k--) {
    const ctx = steps.slice(steps.length - k).join('')
    const counts = grams[k].get(ctx)
    if (counts && counts.size) {
      let best = ''
      let bestC = 0
      let total = 0
      for (const [step, c] of counts) {
        total += c
        if (c > bestC) {
          bestC = c
          best = step
        }
      }
      const purity = bestC / total
      const confidence = clamp(0.5 + 0.3 * purity + 0.04 * (k - 1), 0.45, 0.96)
      return { step: best, confidence, source: `${k}-gram` }
    }
  }
  // prior fallback — most common step overall in the family corpus
  const freq = new Map<string, number>()
  for (const route of fab.corpus[family] ?? []) for (const s of route) freq.set(s, (freq.get(s) ?? 0) + 1)
  const best = [...freq.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? 'MEASURE THICKNESS'
  return { step: best, confidence: 0.42, source: 'prior' }
}

const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x))
const delay = (ms: number) => new Promise((r) => setTimeout(r, ms))

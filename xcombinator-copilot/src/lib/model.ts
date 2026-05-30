import type { Family, Prediction } from '../types'
import { fab, VOCAB } from './data'

// ===========================================================================
// THE MODEL SEAM.
//
// predictNextStep() is the ONE place the dashboard asks "what comes next?".
// Today it runs a local back-off n-gram over a handful of reference routes
// (honest, deterministic, no network) and is labelled "Simulated".
//
// To go live on the team's served checkpoint, set the base URL and nothing
// else changes — the UI, the contract, the shapes all stay identical:
//
//   VITE_MODEL_BASE_URL=http://localhost:8001/v1     # vLLM OpenAI-compatible
//
// The server exposes every loaded checkpoint via GET /v1/models. The UI reads
// that list (listModels) into a dropdown, and predictNextStep(..., modelName)
// routes the request to whichever the presenter picked. VITE_MODEL_NAME is an
// optional default for when no model is selected; it ultimately falls back to
// the simulator on any error.
//
//   VITE_MODEL_NAME=sft-fab-all                       # optional default id
//
// ===========================================================================

const BASE = import.meta.env.VITE_MODEL_BASE_URL as string | undefined
const MODEL = (import.meta.env.VITE_MODEL_NAME as string) || 'default'
const API_KEY = (import.meta.env.VITE_MODEL_API_KEY as string) || 'EMPTY'

export const LIVE = Boolean(BASE)

/**
 * List the model ids the local server has loaded (GET /v1/models).
 * Returns [] on any error or when VITE_MODEL_BASE_URL is unset, so the UI can
 * fall back to the "Simulated (no server)" state without throwing.
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

export async function predictNextStep(
  family: Family,
  steps: string[],
  modelName?: string,
): Promise<Prediction | null> {
  if (steps.length === 0 || steps[steps.length - 1] === 'SHIP LOT') return null

  if (BASE) {
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
      model: modelName ?? MODEL,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0,
      max_tokens: 24,
    }),
  })
  const json = await res.json()
  const raw: string = json?.choices?.[0]?.message?.content ?? ''
  // The local server returns the model's real per-step confidence (geometric mean
  // of the greedy tokens' probabilities). Fall back to a fixed value only if an
  // older server without the field answers.
  const conf = typeof json?.confidence === 'number' ? clamp(json.confidence, 0.05, 0.99) : 0.82
  return { step: snapToVocab(raw), confidence: conf, source: 'model' }
}

function snapToVocab(raw: string): string {
  const t = raw.split('|')[0].trim().replace(/[.\s]+$/, '')
  if (!t) return nearestVocab(raw)
  const up = t.toUpperCase()
  const exact = VOCAB.find((v) => v === up)
  if (exact) return exact
  const contained = VOCAB.filter((v) => up.includes(v)).sort((a, b) => b.length - a.length)[0]
  if (contained) return contained
  const within = VOCAB.filter((v) => v.includes(up)).sort((a, b) => a.length - b.length)[0]
  if (within) return within
  // Safety net: no exact/substring hit, so the model produced garbage or prose.
  // Snap to the closest real vocab entry so the UI never shows a non-vocab token.
  return nearestVocab(up)
}

/**
 * Closest VOCAB entry to an arbitrary string. Ranks by shared-word count, then
 * Levenshtein distance, then absolute length difference — all over the
 * uppercased forms. Always returns a real VOCAB entry (VOCAB is non-empty).
 */
function nearestVocab(raw: string): string {
  const up = raw.trim().toUpperCase()
  const words = new Set(up.split(/[^A-Z0-9]+/).filter(Boolean))
  let best = VOCAB[0]
  let bestKey: [number, number, number] = [-1, Infinity, Infinity]
  for (const v of VOCAB) {
    const vWords = v.split(/[^A-Z0-9]+/).filter(Boolean)
    const overlap = vWords.reduce((n, w) => (words.has(w) ? n + 1 : n), 0)
    const key: [number, number, number] = [overlap, levenshtein(up, v), Math.abs(up.length - v.length)]
    // higher overlap wins; then lower distance; then smaller length difference
    if (key[0] > bestKey[0] || (key[0] === bestKey[0] && (key[1] < bestKey[1] || (key[1] === bestKey[1] && key[2] < bestKey[2])))) {
      best = v
      bestKey = key
    }
  }
  return best
}

/** Standard Levenshtein edit distance (two-row, O(n·m) time, O(m) space). */
function levenshtein(a: string, b: string): number {
  if (a === b) return 0
  if (!a.length) return b.length
  if (!b.length) return a.length
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j)
  let cur = new Array<number>(b.length + 1)
  for (let i = 1; i <= a.length; i++) {
    cur[0] = i
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
    }
    ;[prev, cur] = [cur, prev]
  }
  return prev[b.length]
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

import { FAB_DATA } from "./data/fab-data.js";
import {
  RULE_IDS,
  ID_TO_STEP,
  rankCandidates,
  completeRouteBaseline,
  checkAnomaly,
  parseSequence,
  encodeSteps,
  normalizeFamily,
} from "./engine.js";

const CORS = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET,POST,OPTIONS",
  "access-control-allow-headers": "content-type,authorization",
};
const JSON_HEADERS = { "content-type": "application/json; charset=utf-8", ...CORS };

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") return new Response(null, { headers: CORS });

    const url = new URL(request.url);
    try {
      if (url.pathname.startsWith("/api/")) return await handleApi(request, env, url);
      return env.ASSETS.fetch(request);
    } catch (error) {
      return json({ error: error.message || "Unexpected error" }, 500);
    }
  },
};

async function handleApi(request, env, url) {
  if (url.pathname === "/api/health") {
    return json({
      ok: true,
      modelConfigured: Boolean(env.MODEL_API_KEY && env.MODEL_NAME),
      modelName: env.MODEL_NAME || null,
      productLines: FAB_DATA.families,
      stats: FAB_DATA.stats,
      vocabSize: FAB_DATA.vocab.length,
      ruleIds: RULE_IDS,
    });
  }

  if (url.pathname === "/api/vocab") {
    return json({ productLines: FAB_DATA.families, steps: FAB_DATA.vocab });
  }

  if (url.pathname === "/api/sample") {
    const family = normalizeFamily(url.searchParams.get("family") || "MOSFET");
    const fraction = Number(url.searchParams.get("fraction") || "0.6");
    return json(sampleRoute(family, fraction));
  }

  if (request.method !== "POST") return json({ error: "Method not allowed" }, 405);
  const body = await request.json().catch(() => ({}));
  const family = normalizeFamily(body.family || body.productLine || "MOSFET");

  if (url.pathname === "/api/predict") {
    const partial = parseSequence(body.partial_sequence ?? body.partialSequence ?? body.sequence);
    return json(await predictNext(family, partial, env));
  }

  if (url.pathname === "/api/predict-batch") {
    return json(predictBatch(body.csv ?? body.csvText ?? ""));
  }

  if (url.pathname === "/api/complete") {
    const partial = parseSequence(body.partial_sequence ?? body.partialSequence ?? body.sequence);
    return json(await completeSequence(family, partial, env, Number(body.max_steps || body.maxSteps || 80)));
  }

  if (url.pathname === "/api/anomaly") {
    const sequence = parseSequence(body.sequence);
    return json(checkAnomaly(family, sequence));
  }

  return json({ error: "Not found" }, 404);
}

async function predictNext(family, partialSteps, env) {
  const { ids, unknownSteps } = encodeSteps(partialSteps);
  const ranked = rankCandidates(family, ids, 12);
  const modelResult = await rerankWithModel(env, family, partialSteps, ranked);
  const modelApplied = Boolean(modelResult?.appliedCount);
  const predictions = modelApplied ? modelResult.predictions : ranked.slice(0, 5);

  return {
    family,
    partialLength: partialSteps.length,
    unknownSteps,
    source: modelApplied ? "model_reranked_retrieval" : ranked[0]?.source || "retrieval",
    modelAttempted: Boolean(modelResult?.attempted),
    modelApplied,
    predictions: predictions.slice(0, 5),
    reason: modelApplied && modelResult?.reason
      ? modelResult.reason
      : explainPrediction(family, partialSteps, predictions),
  };
}

function predictBatch(csvText) {
  const { rows, errors } = parseBatchCsv(csvText);
  const results = rows.map((row, i) => {
    try {
      const family = normalizeFamily(row.family);
      const partial = parseSequence(row.partial_sequence);
      if (!partial.length) throw new Error("Empty PARTIAL_SEQUENCE");
      const { ids, unknownSteps } = encodeSteps(partial);
      const ranked = rankCandidates(family, ids, 5);
      const top = ranked[0];
      return {
        example_id: row.example_id || `row_${i + 1}`,
        family,
        completion_fraction: row.completion_fraction,
        partial_length: partial.length,
        predicted_next_step: top?.step || null,
        confidence: top?.confidence ?? null,
        source: top?.source || "position",
        top5: ranked,
        unknown_steps: unknownSteps,
      };
    } catch (error) {
      return {
        example_id: row.example_id || `row_${i + 1}`,
        family: row.family,
        completion_fraction: row.completion_fraction,
        error: error.message || "Could not predict",
      };
    }
  });
  return { count: results.length, parseErrors: errors, results };
}

// Parses the EXAMPLE_ID,FAMILY,COMPLETION_FRACTION,PARTIAL_SEQUENCE format.
// PARTIAL_SEQUENCE is the last column and holds pipe-joined steps, so the
// final cell absorbs any stray commas.
function parseBatchCsv(text) {
  const lines = String(text || "").split(/\r?\n/).filter((line) => line.trim());
  if (!lines.length) return { rows: [], errors: ["File is empty."] };

  const header = lines[0].split(",").map((h) => h.trim().toUpperCase());
  const idx = {
    id: header.indexOf("EXAMPLE_ID"),
    family: header.indexOf("FAMILY"),
    fraction: header.indexOf("COMPLETION_FRACTION"),
    seq: header.indexOf("PARTIAL_SEQUENCE"),
  };
  if (idx.family < 0 || idx.seq < 0) {
    return { rows: [], errors: ["Header must include FAMILY and PARTIAL_SEQUENCE columns."] };
  }

  const rows = [];
  for (let i = 1; i < lines.length; i += 1) {
    const cells = splitCsvRow(lines[i], header.length);
    rows.push({
      example_id: idx.id >= 0 ? cells[idx.id] : `row_${i}`,
      family: cells[idx.family],
      completion_fraction: idx.fraction >= 0 ? cells[idx.fraction] : "",
      partial_sequence: cells[idx.seq],
    });
  }
  return { rows, errors: [] };
}

function splitCsvRow(line, columns) {
  const parts = line.split(",");
  if (parts.length <= columns) return parts.map((c) => c.trim());
  const head = parts.slice(0, columns - 1).map((c) => c.trim());
  head.push(parts.slice(columns - 1).join(",").trim());
  return head;
}

async function completeSequence(family, partialSteps, env, maxSteps) {
  const result = completeRouteBaseline(family, partialSteps, maxSteps);
  return {
    family,
    predicted_sequence: result.predicted_sequence,
    full_sequence: result.full_sequence,
    completionMode: result.completionMode,
    anomaly: checkAnomaly(family, result.full_sequence),
  };
}

async function rerankWithModel(env, family, partialSteps, candidates) {
  if (!env.MODEL_API_KEY || !env.MODEL_NAME || !candidates.length) return null;

  const baseUrl = (env.MODEL_BASE_URL || "http://localhost:8001/v1").replace(/\/$/, "");
  const user = {
    product_line: family,
    partial_sequence: partialSteps.slice(-35),
    candidate_next_steps: candidates
      .slice(0, 10)
      .map((p) => p.step)
      .sort((a, b) => a.localeCompare(b)),
    rules: [
      "Routes start with logistics and inspection, then measurements, clean, product-line prep, oxidation, process cycles, ILD, via, metal, passivation, backside, final inspection, test, ship.",
      "Return only exact step names from candidate_next_steps.",
      "Respect product-line-specific sequences: IC, IGBT, and MOSFET have different route structure.",
    ],
    output_schema: {
      ranked_next_steps: ["exact candidate step", "exact candidate step"],
      reason: "short process-logic explanation",
    },
  };

  const response = await fetch(`${baseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${env.MODEL_API_KEY}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: env.MODEL_NAME,
      temperature: 0,
      max_tokens: 350,
      chat_template_kwargs: { enable_thinking: false },
      messages: [
        {
          role: "system",
          content:
            "You rank semiconductor fabrication next-step candidates. Do not use thinking text. Return strict JSON only. /no_think",
        },
        { role: "user", content: JSON.stringify(user) },
      ],
    }),
  });

  if (!response.ok) return { attempted: true, appliedCount: 0 };
  const data = await response.json();
  const text = data.choices?.[0]?.message?.content || "";
  const parsed = extractJson(text);
  if (!parsed?.ranked_next_steps?.length) return { attempted: true, appliedCount: 0 };

  const allowed = new Set(candidates.map((p) => p.step));
  const byStep = new Map(candidates.map((p) => [p.step, p]));
  const ordered = [];
  let appliedCount = 0;

  for (const step of parsed.ranked_next_steps) {
    if (allowed.has(step) && !ordered.some((p) => p.step === step)) {
      ordered.push(byStep.get(step));
      appliedCount += 1;
    }
  }
  for (const candidate of candidates) {
    if (ordered.length >= 5) break;
    if (!ordered.some((p) => p.step === candidate.step)) ordered.push(candidate);
  }

  return {
    attempted: true,
    appliedCount,
    predictions: ordered.map((p, i) => ({ ...p, rank: i + 1 })),
    reason: typeof parsed.reason === "string" ? parsed.reason : "",
  };
}

function sampleRoute(family, fraction) {
  const seq = FAB_DATA.sequences[family][0].map((id) => ID_TO_STEP[id]);
  const cut = Math.max(3, Math.min(seq.length - 1, Math.round(seq.length * fraction)));
  const invalid = [...seq];
  const ship = invalid.pop();
  invalid.splice(Math.max(4, cut - 2), 0, ship);
  return {
    family,
    fraction,
    partial_sequence: seq.slice(0, cut),
    invalid_sequence: invalid,
    stats: FAB_DATA.stats[family],
  };
}

function explainPrediction(family, partialSteps, predictions) {
  const top = predictions[0]?.step || "no step";
  return `${family} route matched the observed training grammar near step ${partialSteps.length}; strongest next step is ${top}.`;
}

function extractJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    const match = /\{[\s\S]*\}/.exec(text);
    if (!match) return null;
    try {
      return JSON.parse(match[0]);
    } catch {
      return null;
    }
  }
}

function json(payload, status = 200) {
  return new Response(JSON.stringify(payload), { status, headers: JSON_HEADERS });
}

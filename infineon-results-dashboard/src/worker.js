import { FAB_DATA } from "./data/fab-data.js";

const CORS = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET,POST,OPTIONS",
  "access-control-allow-headers": "content-type,authorization",
};
const JSON_HEADERS = { "content-type": "application/json; charset=utf-8", ...CORS };
const STEP_TO_ID = new Map(FAB_DATA.vocab.map((step, id) => [step, id]));
const ID_TO_STEP = FAB_DATA.vocab;
const MAX_NGRAM = 12;

const RULE_IDS = [
  "RULE_DEP_NO_CLEAN",
  "RULE_METAL_ETCH_NO_LITHO",
  "RULE_ETCH_NO_MASK",
  "RULE_LITHO_LEVEL_SKIP",
  "RULE_IMPLANT_NO_MASK",
  "RULE_CMP_NO_DEP",
  "RULE_PAD_OPEN_BEFORE_DEP",
  "RULE_TEST_BEFORE_PASSIVATION",
  "RULE_SHIP_BEFORE_TEST",
  "RULE_BACKSIDE_BEFORE_PASSIVATION",
];

const DEPOSITION_STEPS = set([
  "THERMAL OXIDATION",
  "GATE OXIDE GROWTH",
  "DEPOSIT PAD OXIDE",
  "EPITAXIAL DEPOSITION",
  "DEPOSIT POLYSILICON",
  "DEPOSIT SPACER DIELECTRIC",
  "DEPOSIT FIELD OXIDE",
  "DEPOSIT GATE OXIDE OR DIELECTRIC",
  "DEPOSIT INTERLAYER DIELECTRIC",
  "DEPOSIT INTERLEVEL DIELECTRIC",
  "DEPOSIT BARRIER METAL",
  "DEPOSIT METAL SEED",
  "DEPOSIT METAL 1",
  "DEPOSIT TOP METAL",
  "DEPOSIT BACKSIDE METAL",
  "DEPOSIT TUNGSTEN SEED",
  "DEPOSIT PASSIVATION",
  "DEPOSIT PASSIVATION LAYER",
  "DEPOSIT BACKSIDE PROTECTION",
]);

const CLEAN_STEPS = set([
  "PRE CLEAN WAFER",
  "WAFER CLEAN PRE PROCESS",
  "WAFER SURFACE CLEAN",
  "RCA CLEAN 1",
  "RCA CLEAN 2",
  "WET CLEAN RCA1",
  "WET CLEAN RCA2",
  "HF DIP",
  "OXIDE STRIP",
  "SURFACE PREP FOR DEPOSITION",
  "FRONTSIDE CLEAN",
  "BACKSIDE CLEAN",
  "FRONTSIDE CLEAN FINAL",
  "BACKSIDE CLEAN FINAL",
  "WAFER CLEAN PRE-GRIND",
  "DRY WAFER",
  "DRY WAFER BACKSIDE",
  "CLEAN AFTER ETCH",
  "CLEAN AFTER OXIDE ETCH",
  "CLEAN AFTER POLY ETCH",
  "CLEAN AFTER VIA ETCH",
  "CLEAN AFTER METAL ETCH",
  "CLEAN AFTER WINDOW ETCH",
  "CLEAN AFTER FIELD ETCH",
  "CLEAN PAD OPENING",
  "BACKSIDE ETCH CLEAN",
  "BACKSIDE RINSE",
  "THERMAL OXIDATION",
  "GATE OXIDE PREP",
  "RAPID THERMAL ANNEAL",
  "EPITAXY ANNEAL",
  "ANNEAL OXIDE",
]);

const ETCH_STEPS = set([
  "OXIDE ETCH",
  "OXIDE ETCH DRY",
  "POLYSILICON ETCH",
  "POLYSILICON ETCH DRY",
  "ETCH SILICON OR OXIDE WINDOW",
  "FIELD OXIDE ETCH",
  "VIA ETCH",
  "VIA ETCH THROUGH DIELECTRIC",
  "DIELECTRIC ETCH VIA",
  "METAL ETCH",
  "METAL ETCH DRY",
  "PASSIVATION ETCH PAD OPENING",
  "PASSIVATION ETCH",
]);

const METAL_ETCH_STEPS = set(["METAL ETCH", "METAL ETCH DRY"]);
const IMPLANT_STEPS = set([
  "IMPLANT WELL",
  "IMPLANT SOURCE DRAIN",
  "IMPLANT SOURCE REGION",
  "IMPLANT LDD",
  "IMPLANT P BODY",
  "IMPLANT N BUFFER",
  "IMPLANT CHANNEL STOP",
  "IMPLANT DRAIN / CATHODE REGION",
  "IMPLANT N-TYPE",
]);
const IMPLANT_OPENER_STEPS = set([
  "OXIDE ETCH",
  "OXIDE ETCH DRY",
  "ETCH SILICON OR OXIDE WINDOW",
  "DEVELOP PHOTORESIST",
]);
const CMP_STEPS = set(["CMP DIELECTRIC", "CMP INTERLAYER DIELECTRIC", "CMP METAL", "CMP VIA FILL"]);
const FILL_STEPS = new Set([...DEPOSITION_STEPS, "FILL VIA METAL", "FILL VIA TUNGSTEN"]);
const PAD_WINDOW_STEPS = set([
  "OPEN PAD WINDOW",
  "OPEN BOND PAD WINDOW",
  "PAD WINDOW LITHO",
  "OPEN PAD WINDOW LITHO",
]);
const ELECTRICAL_TEST_STEPS = set([
  "PARAMETRIC TEST",
  "ELECTRICAL PARAMETRIC TEST",
  "THRESHOLD VOLTAGE TEST",
  "BREAKDOWN VOLTAGE TEST",
  "LEAKAGE TEST",
  "SWITCHING TEST",
]);
const BACKSIDE_METAL_STEPS = set(["DEPOSIT BACKSIDE METAL"]);

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

async function completeSequence(family, partialSteps, env, maxSteps) {
  const originalLength = partialSteps.length;
  const current = [...partialSteps];
  const exact = trainingCompletion(family, current);

  if (exact) {
    const full = [...current, ...exact];
    return {
      family,
      predicted_sequence: exact,
      full_sequence: full,
      completionMode: exact.mode,
      anomaly: checkAnomaly(family, full),
    };
  }

  const cap = Math.min(Math.max(maxSteps, 1), 120);
  for (let i = 0; i < cap; i += 1) {
    if (current.at(-1) === "SHIP LOT") break;
    const { ids, unknownSteps } = encodeSteps(current);
    if (unknownSteps.length) break;
    const next = rankCandidates(family, ids, 1)[0]?.step;
    if (!next) break;
    current.push(next);
  }

  return {
    family,
    predicted_sequence: current.slice(originalLength),
    full_sequence: current,
    completionMode: "iterative_ngram",
    anomaly: checkAnomaly(family, current),
  };
}

function checkAnomaly(family, steps) {
  const { unknownSteps } = encodeSteps(steps);
  if (unknownSteps.length) {
    return {
      family,
      is_valid: false,
      score: 0.01,
      predicted_rule: "UNKNOWN_STEP",
      violations: unknownSteps.map((step) => ({ rule: "UNKNOWN_STEP", step_name: step })),
    };
  }

  const violations = validateSequence(steps);
  return {
    family,
    is_valid: violations.length === 0,
    score: violations.length ? 0.05 : 0.97,
    predicted_rule: violations[0]?.rule || null,
    violations,
  };
}

function rankCandidates(family, partialIds, limit = 5) {
  const sequences = FAB_DATA.sequences[family] || [];
  const maxN = Math.min(MAX_NGRAM, partialIds.length);

  for (let n = maxN; n >= 1; n -= 1) {
    const suffix = partialIds.slice(-n);
    const counts = new Map();
    for (const seq of sequences) {
      for (let i = n; i < seq.length; i += 1) {
        if (matchesAt(seq, i - n, suffix)) counts.set(seq[i], (counts.get(seq[i]) || 0) + 1);
      }
    }
    if (counts.size) return countsToPredictions(counts, limit, `${n}-gram_${family}`);
  }

  const position = partialIds.length;
  const counts = new Map();
  for (const seq of sequences) {
    if (position < seq.length) counts.set(seq[position], (counts.get(seq[position]) || 0) + 1);
  }
  return countsToPredictions(counts, limit, `position_${family}`);
}

function countsToPredictions(counts, limit, source) {
  const total = [...counts.values()].reduce((a, b) => a + b, 0) || 1;
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || ID_TO_STEP[a[0]].localeCompare(ID_TO_STEP[b[0]]))
    .slice(0, limit)
    .map(([id, count], rank) => ({
      rank: rank + 1,
      step: ID_TO_STEP[id],
      confidence: Math.round((count / total) * 1000) / 1000,
      support: count,
      source,
    }));
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

function trainingCompletion(family, partialSteps) {
  const { ids, unknownSteps } = encodeSteps(partialSteps);
  if (unknownSteps.length || !ids.length) return null;
  const sequences = FAB_DATA.sequences[family] || [];

  for (const seq of sequences) {
    if (ids.length < seq.length && matchesAt(seq, 0, ids)) {
      const steps = seq.slice(ids.length).map((id) => ID_TO_STEP[id]);
      steps.mode = "exact_training_prefix";
      return steps;
    }
  }

  const maxSuffix = Math.min(18, ids.length);
  for (let n = maxSuffix; n >= 3; n -= 1) {
    const suffix = ids.slice(-n);
    let best = null;
    for (const seq of sequences) {
      for (let start = 0; start <= seq.length - n; start += 1) {
        if (!matchesAt(seq, start, suffix)) continue;
        const continuationStart = start + n;
        if (continuationStart >= seq.length) continue;
        const positionPenalty = Math.abs(continuationStart - ids.length);
        if (!best || n > best.n || (n === best.n && positionPenalty < best.positionPenalty)) {
          best = { seq, continuationStart, n, positionPenalty };
        }
      }
    }
    if (best) {
      const steps = best.seq.slice(best.continuationStart).map((id) => ID_TO_STEP[id]);
      steps.mode = `suffix_match_${best.n}`;
      return steps;
    }
  }
  return null;
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

function validateSequence(steps) {
  const violations = [];
  const window = (i, size) => steps.slice(Math.max(0, i - size), i);
  const anyInWindow = (i, size, targets) => window(i, size).some((s) => targets.has(s));

  steps.forEach((step, i) => {
    if (DEPOSITION_STEPS.has(step) && !anyInWindow(i, 12, CLEAN_STEPS)) {
      violations.push(v("RULE_DEP_NO_CLEAN", i, step, "Deposition requires a prior clean step."));
    }
    if (METAL_ETCH_STEPS.has(step)) {
      const w = window(i, 15);
      const hasExpose = w.some((s) => s.startsWith("EXPOSE LITHO LEVEL"));
      const hasDevelop = w.includes("DEVELOP PHOTORESIST") || w.includes("DEVELOP PAD WINDOW");
      if (!hasExpose || !hasDevelop) {
        violations.push(v("RULE_METAL_ETCH_NO_LITHO", i, step, "Metal etch requires expose and develop."));
      }
    }
    if (ETCH_STEPS.has(step)) {
      const w = window(i, 12);
      const hasDevelop = w.includes("DEVELOP PHOTORESIST") || w.includes("DEVELOP PAD WINDOW");
      if (!hasDevelop) violations.push(v("RULE_ETCH_NO_MASK", i, step, "Etch requires a patterned mask."));
    }
    if (IMPLANT_STEPS.has(step) && !anyInWindow(i, 15, IMPLANT_OPENER_STEPS)) {
      violations.push(v("RULE_IMPLANT_NO_MASK", i, step, "Implant requires an open implant window."));
    }
    if (CMP_STEPS.has(step) && !anyInWindow(i, 6, FILL_STEPS)) {
      violations.push(v("RULE_CMP_NO_DEP", i, step, "CMP requires prior deposition or fill."));
    }
  });

  const aligns = [];
  steps.forEach((step, i) => {
    const m = /^ALIGN MASK LEVEL (\d+)$/.exec(step);
    if (m) aligns.push([i, Number(m[1])]);
  });
  for (let i = 1; i < aligns.length; i += 1) {
    const [prevI, prevLevel] = aligns[i - 1];
    const [currI, currLevel] = aligns[i];
    if (currLevel > prevLevel + 1 || currLevel < prevLevel) {
      violations.push(
        v("RULE_LITHO_LEVEL_SKIP", currI, steps[currI], `Litho level moved from ${prevLevel} at ${prevI} to ${currLevel}.`),
      );
    }
  }

  let passivationDep = -1;
  let cure = -1;
  steps.forEach((step, i) => {
    if (step === "DEPOSIT PASSIVATION" || step === "DEPOSIT PASSIVATION LAYER") passivationDep = i;
    if (step === "CURE PASSIVATION") cure = i;
    if (PAD_WINDOW_STEPS.has(step) && (passivationDep < 0 || cure < 0 || i < passivationDep || i < cure)) {
      violations.push(v("RULE_PAD_OPEN_BEFORE_DEP", i, step, "Pad window must follow deposited and cured passivation."));
    }
  });

  steps.forEach((step, i) => {
    if (ELECTRICAL_TEST_STEPS.has(step) && (cure < 0 || i < cure)) {
      violations.push(v("RULE_TEST_BEFORE_PASSIVATION", i, step, "Electrical test must follow passivation cure."));
    }
    if (BACKSIDE_METAL_STEPS.has(step) && (cure < 0 || i < cure)) {
      violations.push(v("RULE_BACKSIDE_BEFORE_PASSIVATION", i, step, "Backside metal must follow passivation cure."));
    }
  });

  const ship = steps.indexOf("SHIP LOT");
  const sort = steps.indexOf("WAFER SORT TEST");
  if (ship >= 0 && (sort < 0 || ship < sort)) {
    violations.push(v("RULE_SHIP_BEFORE_TEST", ship, "SHIP LOT", "Lot shipment must follow wafer sort test."));
  }

  return violations;
}

function v(rule, step_index, step_name, description) {
  return { rule, step_index, step_name, description };
}

function parseSequence(input) {
  if (Array.isArray(input)) return input.map(cleanStep).filter(Boolean);
  return String(input || "")
    .split(/\||\n/)
    .map(cleanStep)
    .filter(Boolean);
}

function cleanStep(step) {
  return String(step || "").trim().replace(/\s+/g, " ").toUpperCase();
}

function encodeSteps(steps) {
  const ids = [];
  const unknownSteps = [];
  for (const step of steps) {
    const id = STEP_TO_ID.get(step);
    if (id === undefined) unknownSteps.push(step);
    else ids.push(id);
  }
  return { ids, unknownSteps };
}

function normalizeFamily(family) {
  const out = String(family || "").trim().toUpperCase();
  if (!FAB_DATA.families.includes(out)) throw new Error(`Unknown product line: ${family}`);
  return out;
}

function matchesAt(seq, start, ids) {
  if (start < 0 || start + ids.length > seq.length) return false;
  for (let i = 0; i < ids.length; i += 1) {
    if (seq[start + i] !== ids[i]) return false;
  }
  return true;
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

function set(items) {
  return new Set(items);
}

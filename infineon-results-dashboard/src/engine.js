import { FAB_DATA } from "./data/fab-data.js";

export { FAB_DATA };
export const STEP_TO_ID = new Map(FAB_DATA.vocab.map((step, id) => [step, id]));
export const ID_TO_STEP = FAB_DATA.vocab;
export const MAX_NGRAM = 12;

export const RULE_IDS = [
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

export const DEPOSITION_STEPS = set([
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

export const CLEAN_STEPS = set([
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

export const ETCH_STEPS = set([
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

export const METAL_ETCH_STEPS = set(["METAL ETCH", "METAL ETCH DRY"]);
export const IMPLANT_STEPS = set([
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
export const IMPLANT_OPENER_STEPS = set([
  "OXIDE ETCH",
  "OXIDE ETCH DRY",
  "ETCH SILICON OR OXIDE WINDOW",
  "DEVELOP PHOTORESIST",
]);
export const CMP_STEPS = set(["CMP DIELECTRIC", "CMP INTERLAYER DIELECTRIC", "CMP METAL", "CMP VIA FILL"]);
export const FILL_STEPS = new Set([...DEPOSITION_STEPS, "FILL VIA METAL", "FILL VIA TUNGSTEN"]);
export const PAD_WINDOW_STEPS = set([
  "OPEN PAD WINDOW",
  "OPEN BOND PAD WINDOW",
  "PAD WINDOW LITHO",
  "OPEN PAD WINDOW LITHO",
]);
export const ELECTRICAL_TEST_STEPS = set([
  "PARAMETRIC TEST",
  "ELECTRICAL PARAMETRIC TEST",
  "THRESHOLD VOLTAGE TEST",
  "BREAKDOWN VOLTAGE TEST",
  "LEAKAGE TEST",
  "SWITCHING TEST",
]);
export const BACKSIDE_METAL_STEPS = set(["DEPOSIT BACKSIDE METAL"]);

export function checkAnomaly(family, steps) {
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

export function rankCandidates(family, partialIds, limit = 5, sequences = FAB_DATA.sequences[family]) {
  sequences = sequences || [];
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

export function countsToPredictions(counts, limit, source) {
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

export function trainingCompletion(family, partialSteps, sequences = FAB_DATA.sequences[family]) {
  const { ids, unknownSteps } = encodeSteps(partialSteps);
  if (unknownSteps.length || !ids.length) return null;
  sequences = sequences || [];

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

export function validateSequence(steps) {
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

export function v(rule, step_index, step_name, description) {
  return { rule, step_index, step_name, description };
}

export function parseSequence(input) {
  if (Array.isArray(input)) return input.map(cleanStep).filter(Boolean);
  return String(input || "")
    .split(/\||\n/)
    .map(cleanStep)
    .filter(Boolean);
}

export function cleanStep(step) {
  return String(step || "").trim().replace(/\s+/g, " ").toUpperCase();
}

export function encodeSteps(steps) {
  const ids = [];
  const unknownSteps = [];
  for (const step of steps) {
    const id = STEP_TO_ID.get(step);
    if (id === undefined) unknownSteps.push(step);
    else ids.push(id);
  }
  return { ids, unknownSteps };
}

export function normalizeFamily(family) {
  const out = String(family || "").trim().toUpperCase();
  if (!FAB_DATA.families.includes(out)) throw new Error(`Unknown product line: ${family}`);
  return out;
}

export function matchesAt(seq, start, ids) {
  if (start < 0 || start + ids.length > seq.length) return false;
  for (let i = 0; i < ids.length; i += 1) {
    if (seq[start + i] !== ids[i]) return false;
  }
  return true;
}

export function set(items) {
  return new Set(items);
}

// Pure baseline next-step prediction: up to k ranked step names from the n-gram engine.
export function predictNextBaseline(family, partialSteps, k = 5, sequences) {
  const { ids } = encodeSteps(partialSteps);
  return rankCandidates(family, ids, k, sequences).map((p) => p.step);
}

// Pure baseline completion: training-prefix exact match then iterative n-gram, no env/model.
// predicted_sequence = steps added AFTER the input (after the cut).
export function completeRouteBaseline(family, partialSteps, maxSteps = 120, sequences) {
  const originalLength = partialSteps.length;
  const current = [...partialSteps];
  const exact = trainingCompletion(family, current, sequences);

  if (exact) {
    const full = [...current, ...exact];
    return {
      predicted_sequence: [...exact],
      full_sequence: full,
      completionMode: exact.mode,
    };
  }

  const cap = Math.min(Math.max(maxSteps, 1), 120);
  for (let i = 0; i < cap; i += 1) {
    if (current.at(-1) === "SHIP LOT") break;
    const { ids, unknownSteps } = encodeSteps(current);
    if (unknownSteps.length) break;
    const next = rankCandidates(family, ids, 1, sequences)[0]?.step;
    if (!next) break;
    current.push(next);
  }

  return {
    predicted_sequence: current.slice(originalLength),
    full_sequence: current,
    completionMode: "iterative_ngram",
  };
}

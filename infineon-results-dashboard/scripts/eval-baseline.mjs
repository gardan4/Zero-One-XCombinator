// Offline batch tool for the Infineon fab-route challenge.
//
//   MODE A (submission):  node scripts/eval-baseline.mjs --input <eval_input_valid.csv> [--outdir DIR]
//     Writes byte-exact nextstep.csv (Task 1) and completion.csv (Task 2).
//
//   MODE B (benchmark):   node scripts/eval-baseline.mjs --benchmark [--outdir DIR]
//     Deterministic train/val split of FAB_DATA, scores the baseline against gold
//     using the track_metrics formulas, writes metrics.json + per-family breakdown.
//
// Predictions come ONLY from the n-gram/retrieval baseline in src/engine.js. No network.

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  FAB_DATA,
  ID_TO_STEP,
  predictNextBaseline,
  completeRouteBaseline,
} from "../src/engine.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const DEFAULT_OUTDIR = resolve(__dirname, "..", "extras", "results");
const STEP_SEP = "|";

// --------------------------------------------------------------------------- CSV I/O

// Read organizer eval input: strip a UTF-8 BOM, strip surrounding spaces on header keys,
// split steps on "|" dropping empties. Mirrors submission.py _norm_rows / _split.
function readEvalInput(text) {
  const lines = String(text).replace(/^﻿/, "").split(/\r\n|\r|\n/);
  while (lines.length && lines[lines.length - 1] === "") lines.pop();
  if (!lines.length) return [];

  const header = parseCsvLine(lines[0]).map((k) => k.replace(/^﻿/, "").trim().replace(/^"|"$/g, "").trim());
  const rows = [];
  for (let i = 1; i < lines.length; i += 1) {
    if (lines[i] === "") continue;
    const cells = parseCsvLine(lines[i]);
    const row = {};
    header.forEach((key, j) => {
      row[key] = (cells[j] ?? "").trim();
    });
    rows.push({
      example_id: row.EXAMPLE_ID ?? "",
      family: row.FAMILY ?? "",
      completion_fraction: row.COMPLETION_FRACTION ? Number(row.COMPLETION_FRACTION) : 0.0,
      partial_sequence: splitSteps(row.PARTIAL_SEQUENCE ?? ""),
    });
  }
  return rows;
}

function splitSteps(s) {
  return String(s || "")
    .split(STEP_SEP)
    .map((x) => x.trim())
    .filter((x) => x);
}

// Minimal RFC-4180-ish line parser (handles quoted fields with embedded commas/quotes).
function parseCsvLine(line) {
  const out = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"') {
        if (line[i + 1] === '"') {
          cur += '"';
          i += 1;
        } else {
          inQuotes = false;
        }
      } else {
        cur += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      out.push(cur);
      cur = "";
    } else {
      cur += ch;
    }
  }
  out.push(cur);
  return out;
}

// Python csv.writer defaults: QUOTE_MINIMAL, lineterminator "\r\n".
// A field is quoted only if it contains a comma, double-quote, CR or LF;
// embedded double-quotes are doubled.
function csvField(value) {
  const s = String(value ?? "");
  if (/[",\r\n]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
  return s;
}

function writeCsv(path, rows) {
  const body = rows.map((row) => row.map(csvField).join(",")).join("\r\n");
  // Python csv.writer terminates EVERY row (including the last) with the lineterminator.
  writeFileSync(path, body + "\r\n");
}

// --------------------------------------------------------------------------- metrics (track_metrics.py)

function levenshtein(a, b) {
  if (a.length === b.length && a.every((x, i) => x === b[i])) return 0;
  if (!a.length) return b.length;
  if (!b.length) return a.length;
  let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i += 1) {
    const cur = [i];
    for (let j = 1; j <= b.length; j += 1) {
      cur.push(Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] !== b[j - 1] ? 1 : 0)));
    }
    prev = cur;
  }
  return prev[b.length];
}

const safeDiv = (a, b) => (b ? a / b : 0.0);

function scoreNextstep(preds, gold) {
  const ids = Object.keys(gold);
  const n = ids.length;
  let top1 = 0;
  let top3 = 0;
  let top5 = 0;
  let rr = 0.0;
  for (const ex of ids) {
    const ranks = preds[ex] || [];
    const idx = ranks.indexOf(gold[ex]);
    const pos = idx >= 0 ? idx + 1 : 0;
    if (pos === 1) top1 += 1;
    if (pos >= 1 && pos <= 3) top3 += 1;
    if (pos >= 1 && pos <= 5) top5 += 1;
    rr += pos ? 1.0 / pos : 0.0;
  }
  return { n, top1: safeDiv(top1, n), top3: safeDiv(top3, n), top5: safeDiv(top5, n), mrr: safeDiv(rr, n) };
}

function scoreCompletion(preds, gold, block = 5) {
  const ids = Object.keys(gold);
  const n = ids.length;
  let em = 0;
  let nedSum = 0.0;
  let tokSum = 0.0;
  let blkSum = 0.0;
  for (const ex of ids) {
    const g = gold[ex];
    const p = preds[ex] || [];
    if (p.length === g.length && p.every((x, i) => x === g[i])) em += 1;
    const denom = Math.max(p.length, g.length) || 1;
    nedSum += levenshtein(p, g) / denom;
    let tokOk = 0;
    const lim = Math.min(p.length, g.length);
    for (let i = 0; i < lim; i += 1) if (p[i] === g[i]) tokOk += 1;
    tokSum += safeDiv(tokOk, g.length || 1);
    const nblocks = Math.floor((g.length + block - 1) / block) || 1;
    let ok = 0;
    for (let s = 0; s < g.length; s += block) {
      const pg = p.slice(s, s + block);
      const gg = g.slice(s, s + block);
      if (pg.length === gg.length && pg.every((x, i) => x === gg[i])) ok += 1;
    }
    blkSum += ok / nblocks;
  }
  return {
    n,
    exact_match: safeDiv(em, n),
    norm_edit_dist: safeDiv(nedSum, n),
    token_acc: safeDiv(tokSum, n),
    block_acc: safeDiv(blkSum, n),
  };
}

// --------------------------------------------------------------------------- arg parsing

function parseArgs(argv) {
  const args = { benchmark: false, input: null, outdir: null };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === "--benchmark") args.benchmark = true;
    else if (a === "--input") args.input = argv[++i];
    else if (a === "--outdir") args.outdir = argv[++i];
  }
  return args;
}

// --------------------------------------------------------------------------- MODE A: submission

function runSubmission(inputPath, outdir) {
  const rows = readEvalInput(readFileSync(inputPath, "utf-8"));
  const nextRows = [["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"]];
  const complRows = [["EXAMPLE_ID", "PREDICTED_SEQUENCE"]];

  for (const r of rows) {
    const ranks = predictNextBaseline(r.family, r.partial_sequence, 5);
    const padded = ranks.slice(0, 5);
    while (padded.length < 5) padded.push("");
    nextRows.push([r.example_id, ...padded]);

    const completion = completeRouteBaseline(r.family, r.partial_sequence).predicted_sequence;
    complRows.push([r.example_id, completion.join(STEP_SEP)]);
  }

  mkdirSync(outdir, { recursive: true });
  const nextPath = resolve(outdir, "nextstep.csv");
  const complPath = resolve(outdir, "completion.csv");
  writeCsv(nextPath, nextRows);
  writeCsv(complPath, complRows);
  console.log(`Wrote ${rows.length} predictions:`);
  console.log(`  ${nextPath}`);
  console.log(`  ${complPath}`);
}

// --------------------------------------------------------------------------- MODE B: benchmark

function runBenchmark(outdir) {
  const fractions = [0.6, 0.8];
  const nsPreds = {};
  const nsGold = {};
  const cpPreds = {};
  const cpGold = {};
  const familyOf = {};

  for (const family of FAB_DATA.families) {
    const all = FAB_DATA.sequences[family];
    const train = [];
    const val = [];
    all.forEach((seq, i) => {
      if (i % 10 === 0) val.push(seq);
      else train.push(seq);
    });

    let counter = 0;
    for (const idSeq of val) {
      const steps = idSeq.map((id) => ID_TO_STEP[id]);
      for (const frac of fractions) {
        const cut = Math.max(1, Math.floor(steps.length * frac));
        if (cut >= steps.length) continue;
        const ex = `${family}_${counter}_${frac}`;
        counter += 1;
        const partial = steps.slice(0, cut);
        nsGold[ex] = steps[cut];
        cpGold[ex] = steps.slice(cut);
        familyOf[ex] = family;
        nsPreds[ex] = predictNextBaseline(family, partial, 5, train);
        cpPreds[ex] = completeRouteBaseline(family, partial, 120, train).predicted_sequence;
      }
    }
  }

  const families = FAB_DATA.families;
  const nextstep = {
    overall: scoreNextstep(nsPreds, nsGold),
    perFamily: families.map((family) => {
      const g = filterByFamily(nsGold, familyOf, family);
      const p = filterKeys(nsPreds, g);
      return { family, ...scoreNextstep(p, g) };
    }),
  };
  const completion = {
    overall: scoreCompletion(cpPreds, cpGold),
    perFamily: families.map((family) => {
      const g = filterByFamily(cpGold, familyOf, family);
      const p = filterKeys(cpPreds, g);
      return { family, ...scoreCompletion(p, g) };
    }),
  };

  const metrics = {
    nextstep,
    completion,
    n_eval_examples: Object.keys(nsGold).length,
    val_frac: 0.1,
  };

  mkdirSync(outdir, { recursive: true });
  const metricsPath = resolve(outdir, "metrics.json");
  writeFileSync(metricsPath, JSON.stringify(metrics, null, 2) + "\n");

  const ns = nextstep.overall;
  const cp = completion.overall;
  console.log(`Benchmark over ${metrics.n_eval_examples} val examples (val_frac=0.1):`);
  console.log(
    `  Task1 next-step: top1=${ns.top1.toFixed(3)} top3=${ns.top3.toFixed(3)} top5=${ns.top5.toFixed(3)} mrr=${ns.mrr.toFixed(3)}`,
  );
  console.log(
    `  Task2 completion: exact=${cp.exact_match.toFixed(3)} ned=${cp.norm_edit_dist.toFixed(3)} tok=${cp.token_acc.toFixed(3)} block=${cp.block_acc.toFixed(3)}`,
  );
  console.log(`  ${metricsPath}`);
}

function filterByFamily(gold, familyOf, family) {
  const out = {};
  for (const ex of Object.keys(gold)) if (familyOf[ex] === family) out[ex] = gold[ex];
  return out;
}

function filterKeys(preds, gold) {
  const out = {};
  for (const ex of Object.keys(gold)) if (ex in preds) out[ex] = preds[ex];
  return out;
}

// --------------------------------------------------------------------------- main

const args = parseArgs(process.argv.slice(2));
const outdir = args.outdir ? resolve(args.outdir) : DEFAULT_OUTDIR;

if (args.benchmark) {
  runBenchmark(outdir);
} else if (args.input) {
  runSubmission(resolve(args.input), outdir);
} else {
  console.error("Usage: eval-baseline.mjs (--input <csv> | --benchmark) [--outdir DIR]");
  process.exit(1);
}

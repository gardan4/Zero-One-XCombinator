/**
 * Generate src/data/results.json from promoted eval artifacts (extras/results/INDEX.json).
 *
 * Feeds the copilot's Benchmarks view (base-vs-best). Mirrors the metrics_report.json
 * parsing in infineon-results-dashboard/scripts/build-results.mjs, but selects entries by
 * role tags and emits the shape src/lib/benchmarks.ts expects.
 *
 * Usage (from xcombinator-copilot/):
 *   node scripts/build-benchmarks.mjs        (or: npm run build:benchmarks)
 *
 * Selection rules from INDEX.json tags:
 *   base   = role:baseline, preferring baseline:zeroshot, else predictor:ngram.
 *   best   = role:finetuned with split:id; null if none promoted yet.
 *   oracle = role:oracle.
 *   idOod  = split:id finetuned (all families) paired against each split:ood
 *            finetuned (per held-out family:X). Empty array if not promoted.
 *   scaling = every entry tagged data-size:<N> (accuracy vs #training sequences),
 *            sorted ascending by N. Empty array if none promoted.
 */
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const appRoot = resolve(here, "..");
const repoRoot = process.env.ZERO_ONE_REPO || resolve(appRoot, "..");
const indexPath = resolve(repoRoot, "extras/results/INDEX.json");
const outPath = resolve(appRoot, "src/data/results.json");

const FAMILIES = ["MOSFET", "IGBT", "IC"];

function loadJson(p) {
  return JSON.parse(readFileSync(p, "utf8"));
}

function tagsOf(entry) {
  return entry?.tags || [];
}

function hasTag(entry, tag) {
  return tagsOf(entry).includes(tag);
}

/** Pull a `family:X` tag, if present. */
function familyTag(entry) {
  const t = tagsOf(entry).find((x) => x.startsWith("family:"));
  return t ? t.slice("family:".length).toUpperCase() : null;
}

/** Pull the N from a `data-size:<N>` tag, if present. */
function dataSizeTag(entry) {
  const t = tagsOf(entry).find((x) => x.startsWith("data-size:"));
  if (!t) return null;
  const n = Number(t.slice("data-size:".length));
  return Number.isFinite(n) ? n : null;
}

function pickOverall(taskBlock) {
  if (!taskBlock?.by_family?.overall) return null;
  return taskBlock.by_family.overall;
}

/** Overall block: `by_family.overall`, else the single family present (else null). */
function pickOverallOrSingle(taskBlock) {
  const byFamily = taskBlock?.by_family;
  if (!byFamily) return null;
  if (byFamily.overall) return byFamily.overall;
  const fams = Object.keys(byFamily);
  return fams.length === 1 ? byFamily[fams[0]] : null;
}

function pickFamily(taskBlock, family) {
  return taskBlock?.by_family?.[family] ?? null;
}

function mapNextstep(o) {
  if (!o) return null;
  return {
    top1: o.top1 ?? 0,
    top3: o.top3 ?? 0,
    top5: o.top5 ?? 0,
    mrr: o.mrr ?? 0,
  };
}

function mapCompletion(o) {
  if (!o) return null;
  return {
    exactMatch: o.exact_match ?? 0,
    normEditDist: o.norm_edit_dist ?? 0,
    tokenAcc: o.token_acc ?? 0,
    blockAcc: o.block_acc ?? 0,
  };
}

/** confusion may be an object {tp,fp,tn,fn} OR a string like "tp80/fp0/tn100/fn20". */
function mapConfusion(raw) {
  const out = { tp: 0, fp: 0, tn: 0, fn: 0 };
  if (!raw) return out;
  if (typeof raw === "string") {
    for (const m of raw.matchAll(/(tp|fp|tn|fn)\s*[:=]?\s*(\d+)/gi)) {
      out[m[1].toLowerCase()] = Number(m[2]);
    }
    return out;
  }
  out.tp = raw.tp ?? 0;
  out.fp = raw.fp ?? 0;
  out.tn = raw.tn ?? 0;
  out.fn = raw.fn ?? 0;
  return out;
}

function mapAnomaly(o) {
  if (!o) return null;
  return {
    binAcc: o.binary_acc ?? 0,
    precision: o.precision ?? 0,
    recall: o.recall ?? 0,
    f1: o.f1 ?? 0,
    rocAuc: o.roc_auc ?? 0,
    ruleAttrAcc: o.rule_attribution_acc ?? 0,
    confusion: mapConfusion(o.confusion),
  };
}

/** Full model block (all three tasks) from a report's overall by_family. */
function buildModel(report, manifest, label) {
  const tasks = report.tasks || {};
  return {
    label,
    modelRef: manifest?.model_ref || report.model_ref || report.predictor || label,
    nextstep: mapNextstep(pickOverall(tasks.nextstep)),
    completion: mapCompletion(pickOverall(tasks.completion)),
    anomaly: mapAnomaly(pickOverall(tasks.anomaly)),
  };
}

/** nextstep + completion for one family (used by the ID->OOD generalization view). */
function buildFamilyTasks(report, family) {
  const tasks = report.tasks || {};
  return {
    nextstep: mapNextstep(pickFamily(tasks.nextstep, family)),
    completion: mapCompletion(pickFamily(tasks.completion, family)),
  };
}

function loadSlug(index, slug) {
  const entry = index[slug];
  if (!entry) throw new Error(`Unknown slug: ${slug}`);
  const dir = resolve(repoRoot, entry.path);
  const reportPath = resolve(dir, "metrics_report.json");
  const manifestPath = resolve(dir, "manifest.json");
  if (!existsSync(reportPath)) throw new Error(`Missing metrics_report.json in ${dir}`);
  return {
    entry,
    report: loadJson(reportPath),
    manifest: existsSync(manifestPath) ? loadJson(manifestPath) : {},
  };
}

/**
 * Data-scaling series: every entry tagged `data-size:<N>`, sorted ascending by N.
 * Each point = { size, nextstepTop1, completionBlockAcc } from the report's overall
 * (by_family.overall, else the single family present).
 */
function buildScaling(index, slugs) {
  const points = [];
  for (const slug of slugs) {
    const size = dataSizeTag(index[slug]);
    if (size == null) continue;
    const { report } = loadSlug(index, slug);
    const tasks = report.tasks || {};
    const ns = pickOverallOrSingle(tasks.nextstep);
    const comp = pickOverallOrSingle(tasks.completion);
    points.push({
      size,
      nextstepTop1: ns?.top1 ?? null,
      completionBlockAcc: comp?.block_acc ?? null,
    });
  }
  points.sort((a, b) => a.size - b.size);
  return points;
}

function main() {
  if (!existsSync(indexPath)) {
    // No promoted results yet — still emit a valid empty payload so the build never breaks.
    writeEmpty(`No index at ${indexPath}`);
    return;
  }

  const index = loadJson(indexPath);
  const slugs = Object.keys(index);

  // ---- base: role:baseline, prefer baseline:zeroshot, else predictor:ngram ----
  const baselineSlugs = slugs.filter((s) => hasTag(index[s], "role:baseline"));
  let baseSlug =
    baselineSlugs.find((s) => hasTag(index[s], "baseline:zeroshot")) ||
    baselineSlugs.find((s) => hasTag(index[s], "predictor:ngram")) ||
    baselineSlugs[0] ||
    null;

  // ---- best: role:finetuned + split:id, EXCLUDING data-size scaling points ----
  // (scaling-study runs are also role:finetuned+split:id but tagged data-size:<N>; the headline
  // "best" is the full-data canonical model, never a scaling point.)
  const bestSlug =
    slugs.find(
      (s) =>
        hasTag(index[s], "role:finetuned") &&
        hasTag(index[s], "split:id") &&
        dataSizeTag(index[s]) == null,
    ) || null;

  // ---- oracle: role:oracle ----
  const oracleSlug = slugs.find((s) => hasTag(index[s], "role:oracle")) || null;

  const base = baseSlug ? buildModel(...modelArgs(loadSlug(index, baseSlug)), "Base · " + labelFor(index[baseSlug])) : null;
  const best = bestSlug ? buildModel(...modelArgs(loadSlug(index, bestSlug)), "Best · " + labelFor(index[bestSlug])) : null;

  // oracle: anomaly ceiling only
  let oracle = null;
  if (oracleSlug) {
    const o = loadSlug(index, oracleSlug);
    oracle = { anomaly: mapAnomaly(pickOverall((o.report.tasks || {}).anomaly)) };
  }

  // ---- per-family: base vs best across MOSFET/IGBT/IC ----
  const baseReport = baseSlug ? loadSlug(index, baseSlug).report : null;
  const bestReport = bestSlug ? loadSlug(index, bestSlug).report : null;
  const perFamily = FAMILIES.map((family) => {
    const b = baseReport ? buildFamilyTasks(baseReport, family) : null;
    const f = bestReport ? buildFamilyTasks(bestReport, family) : null;
    const hasBase = b && (b.nextstep || b.completion);
    const hasBest = f && (f.nextstep || f.completion);
    return {
      family,
      base: hasBase ? b : null,
      best: hasBest ? f : null,
    };
  }).filter((row) => row.base || row.best);

  // ---- ID -> OOD: split:id finetuned (all families) vs each split:ood finetuned (per held-out family) ----
  const idSlug = bestSlug; // the all-family in-distribution finetuned run
  const oodSlugs = slugs.filter((s) => hasTag(index[s], "role:finetuned") && hasTag(index[s], "split:ood"));
  const idOod = [];
  if (idSlug && oodSlugs.length) {
    const idReport = loadSlug(index, idSlug).report;
    for (const slug of oodSlugs) {
      const oodLoaded = loadSlug(index, slug);
      const fam = familyTag(oodLoaded.entry) || familyTag(index[slug]);
      if (!fam) continue;
      idOod.push({
        family: fam,
        // "told the rules" — the all-family run measured on this family's slice
        id: buildFamilyTasks(idReport, fam),
        // "learned the rules" — the leave-one-family-out run, held out on this family
        ood: {
          nextstep: mapNextstep(pickOverall((oodLoaded.report.tasks || {}).nextstep)),
          completion: mapCompletion(pickOverall((oodLoaded.report.tasks || {}).completion)),
        },
      });
    }
  }

  // ---- data scaling: accuracy vs number of training sequences (data-size:<N>) ----
  const scaling = buildScaling(index, slugs);

  const payload = {
    base,
    best,
    oracle,
    perFamily,
    idOod,
    scaling,
    generatedAt: new Date().toISOString(),
    _source: {
      index: "extras/results/INDEX.json",
      baseSlug,
      bestSlug,
      oracleSlug,
      oodSlugs,
    },
  };

  writeFileSync(outPath, JSON.stringify(payload, null, 2) + "\n");
  console.log(
    `Wrote ${outPath}\n  base=${baseSlug || "none"}  best=${bestSlug || "none (awaiting finetuned)"}  oracle=${oracleSlug || "none"}  idOod=${idOod.length} pair(s)  scaling=${scaling.length} point(s)`,
  );
}

/** Build a human label from an INDEX entry (version, else predictor). */
function labelFor(entry) {
  return entry?.version || entry?.predictor || entry?.model_ref || "run";
}

/** Spread helper so buildModel(report, manifest, label) reads cleanly above. */
function modelArgs(loaded) {
  return [loaded.report, loaded.manifest];
}

function writeEmpty(reason) {
  const payload = {
    base: null,
    best: null,
    oracle: null,
    perFamily: [],
    idOod: [],
    scaling: [],
    generatedAt: new Date().toISOString(),
    _source: { note: reason },
  };
  writeFileSync(outPath, JSON.stringify(payload, null, 2) + "\n");
  console.warn(`${reason} — wrote empty ${outPath}`);
}

main();

/**
 * Generate public/results.js from promoted eval artifacts (extras/results/INDEX.json).
 *
 * Usage (from repo root):
 *   node infineon-results-dashboard/scripts/build-results.mjs
 *   node infineon-results-dashboard/scripts/build-results.mjs --baseline <slug> --finetuned <slug>
 */
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(here, "..");
const repoRoot = process.env.ZERO_ONE_REPO || resolve(projectRoot, "..");
const indexPath = resolve(repoRoot, "extras/results/INDEX.json");
const outPath = resolve(projectRoot, "public/results.js");

function parseArgs() {
  const args = process.argv.slice(2);
  const out = { baseline: null, finetuned: null };
  for (let i = 0; i < args.length; i += 1) {
    if (args[i] === "--baseline") out.baseline = args[++i];
    if (args[i] === "--finetuned") out.finetuned = args[++i];
  }
  return out;
}

function loadJson(p) {
  return JSON.parse(readFileSync(p, "utf8"));
}

function pickOverall(taskBlock) {
  if (!taskBlock?.by_family?.overall) return {};
  return taskBlock.by_family.overall;
}

// --- tag helpers for the storyline series (data-scaling + model-size) ---
function tagVal(entry, prefix) {
  const t = (entry.tags || []).find((x) => x.startsWith(prefix));
  return t ? t.slice(prefix.length) : null;
}
function dataSizeOf(entry) {
  const v = tagVal(entry, "data-size:");
  return v == null ? null : Number(v);
}
function modelSizeOf(entry) {
  const v = tagVal(entry, "model-size:"); // e.g. "1.5b"
  if (v == null) return null;
  const m = String(v).toLowerCase().match(/([\d.]+)\s*b/);
  return { label: v, params: m ? Number(m[1]) : null };
}

function mapNextstep(overall) {
  return {
    top1: overall.top1 ?? 0,
    top3: overall.top3 ?? 0,
    top5: overall.top5 ?? 0,
    mrr: overall.mrr ?? 0,
  };
}

function mapCompletion(overall) {
  return {
    exactMatch: overall.exact_match ?? 0,
    normEditDistance: overall.norm_edit_dist ?? 0,
    tokenAcc: overall.token_acc ?? 0,
    blockAcc: overall.block_acc ?? 0,
  };
}

function mapAnomaly(overall) {
  const conf = overall.confusion || {};
  return {
    binAcc: overall.binary_acc ?? 0,
    precision: overall.precision ?? 0,
    recall: overall.recall ?? 0,
    f1: overall.f1 ?? 0,
    rocAuc: overall.roc_auc ?? 0,
    ruleAttrAcc: overall.rule_attribution_acc ?? 0,
    confusion: {
      tp: conf.tp ?? 0,
      fp: conf.fp ?? 0,
      tn: conf.tn ?? 0,
      fn: conf.fn ?? 0,
    },
  };
}

function perFamilyNextstep(report) {
  const fam = report?.tasks?.nextstep?.by_family || {};
  return ["IC", "IGBT", "MOSFET"]
    .filter((f) => fam[f])
    .map((family) => ({
      family,
      baselineTop1: 0,
      baselineTop5: 0,
      finetunedTop1: fam[family].top1 ?? 0,
      finetunedTop5: fam[family].top5 ?? 0,
    }));
}

function buildFromReport(report, manifest, roleLabel) {
  const tasks = report.tasks || {};
  const ns = pickOverall(tasks.nextstep);
  const cp = pickOverall(tasks.completion);
  const an = pickOverall(tasks.anomaly);
  return {
    modelName: manifest?.model_ref || report.model_ref || report.predictor || roleLabel,
    nextstep: mapNextstep(ns),
    completion: mapCompletion(cp),
    anomaly: mapAnomaly(an),
    perFamily: perFamilyNextstep(report),
  };
}

function main() {
  const { baseline: baselineSlug, finetuned: finetunedSlug } = parseArgs();
  if (!existsSync(indexPath)) {
    console.error(`No index at ${indexPath} — promote eval results first (just promote).`);
    process.exit(1);
  }
  const index = loadJson(indexPath);
  const slugs = Object.keys(index);
  const isFinetuned = (s) => (index[s].tags || []).some((t) => t.includes("role:finetuned") || t === "finetuned");
  // Headline "best" = canonical full-data 1.5B, NEVER a scaling point (data-size:N) — those are the
  // data-scaling study, not the best model.
  const finSlug =
    finetunedSlug ||
    slugs.find((s) => isFinetuned(s) && modelSizeOf(index[s])?.params === 1.5 && dataSizeOf(index[s]) == null) ||
    slugs.find((s) => isFinetuned(s) && dataSizeOf(index[s]) == null) ||
    slugs.find(isFinetuned) ||
    slugs[0];
  const baseSlug =
    baselineSlug || slugs.find((s) => s !== finSlug && (index[s].predictor === "ngram" || (index[s].tags || []).includes("role:baseline")));

  function loadSlug(slug) {
    const entry = index[slug];
    if (!entry) throw new Error(`Unknown slug: ${slug}`);
    const dir = resolve(repoRoot, entry.path);
    const reportPath = resolve(dir, "metrics_report.json");
    const manifestPath = resolve(dir, "manifest.json");
    if (!existsSync(reportPath)) throw new Error(`Missing metrics_report.json in ${dir}`);
    return {
      report: loadJson(reportPath),
      manifest: existsSync(manifestPath) ? loadJson(manifestPath) : {},
    };
  }

  const fin = loadSlug(finSlug);
  const finBuilt = buildFromReport(fin.report, fin.manifest, "finetuned");
  let baseBuilt = { nextstep: { top1: 0, top3: 0, top5: 0, mrr: 0 }, completion: { exactMatch: 0, normEditDistance: 1, tokenAcc: 0, blockAcc: 0 }, anomaly: { binAcc: 0, precision: 0, recall: 0, f1: 0, rocAuc: 0, ruleAttrAcc: 0, confusion: { tp: 0, fp: 0, tn: 0, fn: 0 } }, perFamily: [] };
  if (baseSlug) {
    const base = loadSlug(baseSlug);
    baseBuilt = buildFromReport(base.report, base.manifest, "baseline");
  }

  const perFamily = finBuilt.perFamily.map((f) => {
    const b = baseBuilt.perFamily.find((x) => x.family === f.family);
    return {
      family: f.family,
      baselineTop1: b?.finetunedTop1 ?? 0,
      baselineTop5: b?.finetunedTop5 ?? 0,
      finetunedTop1: f.finetunedTop1,
      finetunedTop5: f.finetunedTop5,
    };
  });

  // --- storyline series: data-scaling (data-size:N) + model-size (model-size:Xb) ---
  function seriesMetrics(slug) {
    const { report } = loadSlug(slug);
    const t = report.tasks || {};
    return {
      slug,
      nextstepTop1: pickOverall(t.nextstep).top1 ?? null,
      completionBlockAcc: pickOverall(t.completion).block_acc ?? null,
      anomalyF1: pickOverall(t.anomaly).f1 ?? null,
    };
  }
  const scaling = slugs
    .filter((s) => dataSizeOf(index[s]) != null)
    .map((s) => ({ size: dataSizeOf(index[s]), ...seriesMetrics(s) }))
    .sort((a, b) => a.size - b.size);
  const modelSize = slugs
    .filter((s) => modelSizeOf(index[s]) != null && isFinetuned(s)) // fine-tuned sweep only, not baselines
    .map((s) => ({ ...modelSizeOf(index[s]), ...seriesMetrics(s) }))
    .sort((a, b) => (a.params ?? 0) - (b.params ?? 0));

  const payload = {
    copy: {
      heroEyebrow: "Infineon Industrial-AI · generated from extras/results",
      heroTitlePre: "Process-logic eval —",
      heroTitleEmph: finBuilt.modelName,
      heroTitlePost: "vs baseline",
      heroLede: `Built from promoted results (${finSlug}${baseSlug ? ` vs ${baseSlug}` : ""}). Not placeholder data.`,
      modelName: finBuilt.modelName,
      modelBlurb: fin.manifest?.notes || fin.report.predictor || "",
      takeaways: [
        "Metrics sourced from metrics_report.json in extras/results.",
        "Regenerate with: node infineon-results-dashboard/scripts/build-results.mjs",
        "Use the Next.js /compare dashboard for multi-model study.",
      ],
      sections: {
        nextstep: "Top-1/3/5 and MRR from labeled eval.",
        completion: "Exact match, NED, token and block accuracy.",
        anomaly: "Binary metrics, ROC-AUC, rule attribution.",
        training: "Training curves: use the run registry dashboard.",
        ood: "Compare split:id vs split:ood runs in /compare.",
      },
    },
    families: ["IC", "IGBT", "MOSFET"],
    nextstep: {
      overall: { baseline: baseBuilt.nextstep, finetuned: finBuilt.nextstep },
      perFamily,
    },
    completion: {
      overall: { baseline: baseBuilt.completion, finetuned: finBuilt.completion },
    },
    anomaly: {
      overall: { baseline: baseBuilt.anomaly, finetuned: finBuilt.anomaly },
      confusion: finBuilt.anomaly.confusion,
      perRule: [],
    },
    scaling, // [{size, slug, nextstepTop1, completionBlockAcc, anomalyF1}] — data-scaling study
    modelSize, // [{label, params, slug, nextstepTop1, completionBlockAcc, anomalyF1}] — model-size sweep
    training: { params: "—", epochs: 0, finalLoss: 0, finalValLoss: 0, steps: [] },
    ood: { heldOutFamily: "—", idF1: null, oodF1: null, idTop1: null, oodTop1: null },
    _generated: {
      at: new Date().toISOString(),
      finetuned_slug: finSlug,
      baseline_slug: baseSlug,
      source: indexPath,
    },
  };

  writeFileSync(
    outPath,
    `// Generated by scripts/build-results.mjs — do not edit by hand.\nexport const RESULTS = ${JSON.stringify(payload, null, 2)};\n`,
  );
  console.log(`Wrote ${outPath} (finetuned=${finSlug}, baseline=${baseSlug || "none"})`);
}

main();

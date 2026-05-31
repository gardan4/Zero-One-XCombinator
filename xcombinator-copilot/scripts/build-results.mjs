/**
 * Generate src/data/results.json from promoted eval artifacts (../extras/results/INDEX.json).
 * The copilot's Results view imports this. Run: npm run build:results (from xcombinator-copilot/).
 */
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(here, ".."); // xcombinator-copilot/
const repoRoot = process.env.ZERO_ONE_REPO || resolve(projectRoot, ".."); // repo root
const indexPath = resolve(repoRoot, "extras/results/INDEX.json");
const outPath = resolve(projectRoot, "src/data/results.json");

const WANDB_ENTITY = process.env.WANDB_ENTITY || "XCombinator";
const WANDB_PROJECT = process.env.WANDB_PROJECT || "XCombinator";

const loadJson = (p) => JSON.parse(readFileSync(p, "utf8"));
const pickOverall = (b) => (b?.by_family?.overall ? b.by_family.overall : {});
const tagVal = (e, p) => {
  const t = (e.tags || []).find((x) => x.startsWith(p));
  return t ? t.slice(p.length) : null;
};
const dataSizeOf = (e) => {
  const v = tagVal(e, "data-size:");
  return v == null ? null : Number(v);
};
const modelSizeOf = (e) => {
  const v = tagVal(e, "model-size:");
  if (v == null) return null;
  const m = String(v).toLowerCase().match(/([\d.]+)\s*b/);
  return { label: v, params: m ? Number(m[1]) : null };
};
const wandbUrl = (runId) =>
  runId ? `https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}/runs/${runId}` : null;

function main() {
  if (!existsSync(indexPath)) {
    console.error(`No index at ${indexPath} — promote eval results first.`);
    process.exit(1);
  }
  const index = loadJson(indexPath);
  const slugs = Object.keys(index);
  const isFinetuned = (s) =>
    (index[s].tags || []).some((t) => t.includes("role:finetuned") || t === "finetuned");
  const isLlmBaseline = (s) =>
    (index[s].tags || []).some(
      (t) => t.includes("llm-zeroshot") || t.includes("baseline:zeroshot") || t === "role:baseline",
    ) && (index[s].predictor === "llm-zeroshot" || s.includes("deepseek") || s.includes("zeroshot"));

  const finSlug =
    (slugs.includes("sft-instruct-all-local") && "sft-instruct-all-local") ||
    slugs.find((s) => isFinetuned(s) && modelSizeOf(index[s])?.params === 1.5 && dataSizeOf(index[s]) == null) ||
    slugs.find((s) => isFinetuned(s) && dataSizeOf(index[s]) == null) ||
    slugs.find(isFinetuned) ||
    slugs[0];
  const baseSlug = slugs.find(
    (s) => s !== finSlug && (index[s].predictor === "ngram" || (index[s].tags || []).includes("role:baseline")),
  );

  const loadSlug = (slug) => {
    const dir = resolve(repoRoot, index[slug].path);
    return loadJson(resolve(dir, "metrics_report.json"));
  };
  const metrics = (slug) => {
    const t = loadSlug(slug).tasks || {};
    const ns = pickOverall(t.nextstep);
    const cp = pickOverall(t.completion);
    const an = pickOverall(t.anomaly);
    const entry = index[slug];
    return {
      slug,
      label: entry.version || slug,
      runId: entry.run_id || null,
      wandbUrl: wandbUrl(entry.run_id),
      nNextstep: ns.n ?? null,
      nAnomaly: an.n ?? null,
      nextstepTop1: ns.top1 ?? null,
      completionBlockAcc: cp.block_acc ?? null,
      completionTokenAcc: cp.token_acc ?? null,
      anomalyF1: an.f1 ?? null,
      anomalyRecall: an.recall ?? null,
      notes: entry.notes || null,
    };
  };

  const finetunedSlugs = slugs.filter(isFinetuned);
  const bestOver = (task, key) => {
    let m = null;
    for (const s of finetunedSlugs) {
      const v = pickOverall((loadSlug(s).tasks || {})[task])[key];
      if (typeof v === "number") m = m == null ? v : Math.max(m, v);
    }
    return m;
  };
  const baseT = baseSlug ? loadSlug(baseSlug).tasks || {} : {};

  const llmBaselineSlugs = [
    "qwen-zeroshot-local200",
    "deepseek-zeroshot-mosfet40",
  ].filter((s) => slugs.includes(s));
  const llmBaselineMeta = {
    "qwen-zeroshot-local200": {
      name: "Qwen2.5-1.5B Instruct (zero-shot)",
      evalNote: "eval_local gold · n=200 · unified JSON",
    },
    "deepseek-zeroshot-mosfet40": {
      name: "DeepSeek-V4-Flash (zero-shot)",
      evalNote: "mosfet40 · n=80/task",
    },
    "baseline-zeroshot-1_5b": {
      name: "Qwen2.5-1.5B (hf baseline, legacy)",
      evalNote: "eval_local gold · n=200 · wrong -p hf",
    },
  };
  const llmBaselines = llmBaselineSlugs.map((s) => ({
    ...metrics(s),
    name: llmBaselineMeta[s]?.name ?? s,
    evalNote: llmBaselineMeta[s]?.evalNote ?? "eval_local",
  }));

  const finetuned = {
    name: "SFT instruct-all · Qwen2.5-1.5B",
    evalNote: "eval_local gold · n=200/task",
    ...metrics(finSlug),
  };

  const deepseekSlug = "deepseek-zeroshot-mosfet40";
  const deepseek = slugs.includes(deepseekSlug)
    ? {
        name: "DeepSeek-V4-Flash (zero-shot)",
        evalNote: "mosfet40 · n=80/task",
        ...metrics(deepseekSlug),
      }
    : null;

  const ngramSlug = slugs.find((s) => index[s].predictor === "ngram") || "baseline-ngram";
  const ngram =
    slugs.includes(ngramSlug)
      ? {
          name: "n-gram baseline",
          evalNote: "eval_local gold · n=200/task",
          ...metrics(ngramSlug),
        }
      : null;

  const scale2000Slug = slugs.find((s) => s === "hf-sft-scale-2000" || dataSizeOf(index[s]) === 2000);
  const featuredModels = [];
  if (scale2000Slug && scale2000Slug !== finSlug) {
    featuredModels.push({
      name: "SFT · 2000 training routes (best completion)",
      evalNote: "eval_local gold · n=200/task · data-size:2000",
      highlight: "completion",
      ...metrics(scale2000Slug),
    });
  }

  const bestByTask = {
    nextstep: null,
    completion: null,
    anomaly: null,
  };
  for (const task of ["nextstep", "completion", "anomaly"]) {
    const key = task === "nextstep" ? "top1" : task === "completion" ? "block_acc" : "f1";
    let bestSlug = null;
    let bestVal = null;
    for (const s of finetunedSlugs) {
      const v = pickOverall((loadSlug(s).tasks || {})[task])[key];
      if (typeof v === "number" && (bestVal == null || v > bestVal)) {
        bestVal = v;
        bestSlug = s;
      }
    }
    if (bestSlug) {
      const names = {
        "hf-sft-scale-2000": "SFT · 2000 routes",
        "hf-sft-instruct-all": "SFT instruct-all · 1.5B",
        "sft-instruct-all-local": "SFT instruct-all · 1.5B",
      };
      bestByTask[task] = {
        task,
        slug: bestSlug,
        name: names[bestSlug] || index[bestSlug].version || bestSlug,
        value: bestVal,
        ...metrics(bestSlug),
      };
    }
  }

  const payload = {
    baselineName: baseSlug || "baseline",
    bestName: "best fine-tuned (per task)",
    finetuned,
    deepseek,
    ngram,
    featuredModels,
    bestByTask,
    headline: deepseek
      ? {
          baselineName: deepseek.name,
          finetunedName: finetuned.name,
          baselineEvalNote: deepseek.evalNote,
          finetunedEvalNote: finetuned.evalNote,
          nextstep: { baseline: deepseek.nextstepTop1 ?? 0, finetuned: finetuned.nextstepTop1 ?? 0 },
          completion: {
            baseline: deepseek.completionBlockAcc ?? 0,
            finetuned: finetuned.completionBlockAcc ?? 0,
          },
          anomaly: { baseline: deepseek.anomalyF1 ?? 0, finetuned: finetuned.anomalyF1 ?? 0 },
        }
      : null,
    llmBaselines,
    nextstep: { baseline: pickOverall(baseT.nextstep).top1 ?? 0, finetuned: bestOver("nextstep", "top1") ?? 0 },
    completion: {
      baseline: pickOverall(baseT.completion).block_acc ?? 0,
      finetuned: bestOver("completion", "block_acc") ?? 0,
    },
    anomaly: { baseline: pickOverall(baseT.anomaly).f1 ?? 0, finetuned: bestOver("anomaly", "f1") ?? 0 },
    scaling: slugs
      .filter((s) => dataSizeOf(index[s]) != null)
      .map((s) => ({ size: dataSizeOf(index[s]), ...metrics(s) }))
      .sort((a, b) => a.size - b.size),
    modelSize: slugs
      .filter((s) => modelSizeOf(index[s]) != null && isFinetuned(s))
      .map((s) => ({ ...modelSizeOf(index[s]), ...metrics(s) }))
      .sort((a, b) => (a.params ?? 0) - (b.params ?? 0)),
    perFamily: ["MOSFET", "IGBT", "IC"]
      .map((fam) => {
        const slug = fam === "MOSFET" ? finSlug : `${finSlug.replace(/-local$/, "")}-${fam}`;
        const alt = fam === "MOSFET" ? "hf-sft-instruct-all" : `hf-sft-instruct-all-${fam}`;
        const pick = index[slug] ? slug : index[alt] ? alt : null;
        return pick ? { family: fam, ...metrics(pick) } : null;
      })
      .filter(Boolean),
    wandb: {
      entity: WANDB_ENTITY,
      project: WANDB_PROJECT,
      runs: [finetuned, ...featuredModels, ...llmBaselines]
        .filter((r) => r?.runId)
        .map((r) => ({ label: r.name || r.slug, runId: r.runId, url: r.wandbUrl })),
    },
    generatedAt: new Date().toISOString(),
  };

  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, JSON.stringify(payload, null, 2) + "\n");
  console.log(`Wrote ${outPath} (finetuned=${finSlug}, baseline=${baseSlug || "none"}, llm=${llmBaselineSlugs.join(",")})`);
}

main();

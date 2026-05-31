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

function main() {
  if (!existsSync(indexPath)) {
    console.error(`No index at ${indexPath} — promote eval results first.`);
    process.exit(1);
  }
  const index = loadJson(indexPath);
  const slugs = Object.keys(index);
  const isFinetuned = (s) => (index[s].tags || []).some((t) => t.includes("role:finetuned") || t === "finetuned");
  const finSlug =
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
    return {
      slug,
      nextstepTop1: pickOverall(t.nextstep).top1 ?? null,
      completionBlockAcc: pickOverall(t.completion).block_acc ?? null,
      anomalyF1: pickOverall(t.anomaly).f1 ?? null,
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

  const payload = {
    baselineName: baseSlug || "baseline",
    bestName: "best fine-tuned (per task)",
    nextstep: { baseline: pickOverall(baseT.nextstep).top1 ?? 0, finetuned: bestOver("nextstep", "top1") ?? 0 },
    completion: { baseline: pickOverall(baseT.completion).block_acc ?? 0, finetuned: bestOver("completion", "block_acc") ?? 0 },
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
        const slug = fam === "MOSFET" ? finSlug : `${finSlug}-${fam}`;
        return index[slug] ? { family: fam, ...metrics(slug) } : null;
      })
      .filter(Boolean),
    generatedAt: new Date().toISOString(),
  };

  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, JSON.stringify(payload, null, 2) + "\n");
  console.log(`Wrote ${outPath} (best=${finSlug}, baseline=${baseSlug || "none"})`);
}

main();

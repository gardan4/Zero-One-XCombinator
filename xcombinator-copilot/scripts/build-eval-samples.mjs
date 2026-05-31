/**
 * Sample random sequences from the track eval-input CSVs into src/data/eval_samples.json, so the
 * Live-compare page can fan a real eval sequence out to every model. Reads from the repo root
 * (eval_input_valid.csv + eval_input_anomaly.csv); if a CSV is missing, keeps that bucket empty.
 * Run: npm run build:samples (from xcombinator-copilot/).
 *
 * Each sample keeps its FULL sequence so the Live-compare page can walk the route from the start,
 * predicting each next step and comparing every model's guess against the real next step.
 *   valid   — PARTIAL_SEQUENCE rows (real prefixes at a completion fraction): clean ground-truth routes.
 *   anomaly — full SEQUENCE rows (may contain one rule violation): the route the models must follow.
 */
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(here, ".."); // xcombinator-copilot/
const repoRoot = process.env.ZERO_ONE_REPO || resolve(projectRoot, "..");
const outPath = resolve(projectRoot, "src/data/eval_samples.json");

const PER_BUCKET = 80; // full routes are ~125 steps; 80 each keeps the bundled JSON reasonable

// Minimal CSV parse: split on newlines, comma-split the first 3 fields; the sequence field (last) may
// itself contain no commas (it's pipe-delimited), so a naive split on the leading commas is safe.
function rows(file) {
  const text = readFileSync(file, "utf8").trim();
  const lines = text.split(/\r?\n/);
  const header = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const out = {};
    // header has N columns; the LAST column holds the sequence and never contains a comma here.
    const parts = line.split(",");
    header.forEach((h, i) => {
      out[h] = i === header.length - 1 ? parts.slice(i).join(",") : parts[i];
    });
    return out;
  });
}

function sample(arr, n) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a.slice(0, n);
}

function buildValid() {
  const f = resolve(repoRoot, "eval_input_valid.csv");
  if (!existsSync(f)) return [];
  return sample(rows(f), PER_BUCKET)
    .map((r) => ({
      id: r.EXAMPLE_ID,
      family: r.FAMILY,
      steps: (r.PARTIAL_SEQUENCE || "").split("|").map((s) => s.trim()).filter(Boolean),
    }))
    .filter((r) => r.steps.length >= 4);
}

function buildAnomaly() {
  const f = resolve(repoRoot, "eval_input_anomaly.csv");
  if (!existsSync(f)) return [];
  return sample(rows(f), PER_BUCKET)
    .map((r) => ({
      id: r.EXAMPLE_ID,
      family: r.FAMILY,
      steps: (r.SEQUENCE || "").split("|").map((s) => s.trim()).filter(Boolean),
    }))
    .filter((r) => r.steps.length >= 8);
}

function main() {
  const payload = { valid: buildValid(), anomaly: buildAnomaly(), generatedAt: new Date().toISOString() };
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, JSON.stringify(payload) + "\n");
  console.log(`Wrote ${outPath} (valid=${payload.valid.length}, anomaly=${payload.anomaly.length})`);
}

main();

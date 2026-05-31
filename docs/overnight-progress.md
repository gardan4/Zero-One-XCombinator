# Overnight autonomous run — progress blackboard

Updated each ralph-loop iteration. Goal: best eval values + comparison story + working live demo.
Deadline 10:00. Gotchas in memory `zero-one-instruct-redo.md`. **After any corpus regen: `bash scripts/ship_corpora.sh`.**

## CRITICAL cluster lessons (cost ~2 rounds of failed jobs)
1. **Corpus delivery**: `submit` prep ONLY ships code (sync_code_to_cluster), NOT data/. Run
   `bash scripts/ship_corpora.sh` after any regen.
2. **GPU env / uv sync race**: the cluster Python moved 3.11.2→3.11.13, so ANY `uv sync` wants to
   recreate the SHARED `.venv` → wipes torch/trl. Concurrent jobs syncing raced + corrupted it
   mid-run. FIX (done): patched train.sbatch.j2 + infer.sbatch.j2 to `uv run --no-sync` (verify
   read-only, never sync). Pre-stage once on login node: `cd $R && uv sync --extra gpu`.
3. **Always submit with `--no-prep`** (the prep's `uv sync --extra gpu` also prunes the env).
   For changed configs/code, pscp directly or accept one careful prep.

## Training matrix (SLURM jobs) — round 2 (race-free --no-sync template)
| model | config | job | status | checkpoint slug after eval |
|---|---|---|---|---|
| best 1.5B canonical | leonardo_sft_fab_instruct.yaml | 43146516 | submitted | hf-sft-instruct-all |
| scale 100 (1.5B) | leonardo_sft_scale_instruct_100.yaml | 43146520 | submitted | hf-sft-scale-100 |
| scale 300 (1.5B) | leonardo_sft_scale_instruct_300.yaml | 43146530 | submitted | hf-sft-scale-300 |
| scale 800 (1.5B) | leonardo_sft_scale_instruct_800.yaml | 43146539 | submitted | hf-sft-scale-800 |
| scale 2000 (1.5B) | leonardo_sft_scale_instruct_2000.yaml | 43146546 | RUNNING | hf-sft-scale-2000 |
| size 0.5B | leonardo_sft_fab_instruct_0_5b.yaml | 43147368 | RUNNING | hf-sft-0_5b (tag model-size:0.5b) |
| size 3B | leonardo_sft_fab_instruct_3b.yaml | 43147369 | RUNNING (watch OOM) | hf-sft-3b (tag model-size:3b) |
| scale 100 (DONE,evaled) | — | 43146520 | EVALED: ns0.365 cp0.345 an-f1:0 (underfit) | hf-sft-scale-100 |

**Best-model waiter armed (43146516).** When it fires → batch-eval everything COMPLETED (best + scale 300/800/2000 + 0.5B, maybe 3B). Eval cmd per model: `uv run zo-cluster judge-eval --local --no-prep --model <ckpt> --predictor hf --version <slug>-v2 --eval-dir extras/eval_local/MOSFET --tags real-run,reportable,role:finetuned,split:id,family:MOSFET[,scale,data-size:N | ,model-size:X] --train-run <rid> --promote <slug>` then `pull_promote_scale.sh`-style pull. Tag the BEST eval with model-size:1.5b (so it anchors the size panel too). Until it fires: BUILD THE DASHBOARD (infineon-results-dashboard build-results.mjs + results.js → add scaling[] + modelSize[] panels + storyline).
| size 0.5B | (todo: leonardo_sft_fab_instruct_0_5b.yaml) | - | needs base staged | hf-sft-0_5b |
| size 3B | (todo: leonardo_sft_fab_instruct_3b.yaml) | - | needs base staged | hf-sft-3b |
| size 7B | (todo, if feasible) | - | needs base staged | hf-sft-7b |

Baselines (no training): n-gram (have: extras/results/baseline-ngram), zero-shot base (todo: judge-eval the frozen base).

## Pipeline state
- [x] Merged main JSON refactor; kept sft.py fix; corpora regenerated (JSON) + shipped to cluster.
- [x] Fixed corpus delivery (ship_corpora.sh), GPU env (uv sync --extra gpu), sbatch race (--no-sync templates).
- [x] Best + 4 scaling RUNNING cleanly on correct JSON corpora (round-2 jobs in table). scale-100 ~5min.
- [ ] **NEXT: scale-100 (43146520) done → validate JSON eval** then `pull_promote_scale.sh 100`. Confirm scorer gives sane top1/block_acc AND anomaly f1 > 0 (balanced+CoT corpus should fix the old f1=0).
- [x] Stage base 0.5B/3B — DONE (in hf-local). Size configs created: leonardo_sft_fab_instruct_{0_5b,3b}.yaml. TODO: submit them --no-prep (after scale-100 eval validates). For the size panel, eval --tags must include model-size:0.5b / model-size:1.5b / model-size:3b (tag the canonical best eval with model-size:1.5b too).
- [ ] Eval every checkpoint (judge-eval --no-prep) + promote → extras/results/INDEX.json.
- [x] Dashboard DATA layer DONE: infineon-results-dashboard/scripts/build-results.mjs now emits
  `scaling[]` (data-size:N) + `modelSize[]` (model-size:Xb); headline 'best' excludes scaling points.
  Stale raw entries dropped from INDEX. Regenerate anytime: `node infineon-results-dashboard/scripts/build-results.mjs`.
- [x] Dashboard FRONTEND DONE: built self-contained **infineon-results-dashboard/public/story.html**
  (hero + base-vs-best cards + data-scaling chart + model-size chart + narrative), renders from
  results.js, VERIFIED via preview (looks great). Auto-populates as evals land — just re-run
  build-results.mjs. To view: `python3 -m http.server -d infineon-results-dashboard/public` → /story.html.
  (The existing index.html dashboard also renders base-vs-best but its hero shows the raw checkpoint
  path as the model name — cosmetic; story.html is the clean storyline.)
- [ ] (optional) zero-shot base-model baseline: eval FROZEN Qwen2.5-1.5B-Instruct with rules-in-context
  (ZO_RULES_IN_CONTEXT / zero-shot predictor) → richer "base model vs best model" story. n-gram baseline already covers "baseline".
- [ ] BATCH-EVAL when best-waiter fires: ls completed checkpoints in zo-experiments, judge-eval each:
  best→slug hf-sft-instruct-all tags(...,model-size:1.5b); scale 300/800/2000→hf-sft-scale-N tags(...,scale,data-size:N);
  0.5b/3b→hf-sft-{0_5b,3b} tags(...,model-size:{0.5b,3b}). Then pull+promote, rebuild build-results, build frontend panels.
- [x] serve_copilot_mac.py JSON framing DONE + **E2E VALIDATED** (started server, curled copilot-format
  request → reframed → base model generated JSON → parsed steps[0] → 'PREPARE WAFERS' + conf 0.77).
  Live demo path works. REMAINING: when best lands, pscp ckpt to ~/zo-models/sft-instruct-all, run
  `ZO_COPILOT_MODELS='{"base-qwen":"Qwen/Qwen2.5-1.5B-Instruct","sft-best":"~/zo-models/sft-instruct-all"}' uv run --no-sync python scripts/serve_copilot_mac.py`,
  start copilot (cd xcombinator-copilot; set .env.local VITE_MODEL_BASE_URL=http://localhost:8001/v1 + VITE_MODEL_NAME=sft-best; npm run dev), verify base-vs-best in UI.

## Active background waiters
- scale-100 completion waiter (fires when 43146520 terminal → run the eval validation above).
- base-model staging on cluster (/tmp/stage_models.log).

## Eval results collected (fill as evals complete)
- **JSON eval pipeline VALIDATED** (43146973): scorer parses correctly (nextstep→list, completion→list,
  anomaly→{is_valid,rule}). NOT an eval bug.
- **hf-sft-scale-100** (100 seqs, 1 epoch): nextstep top1 **0.365**, completion block_acc 0.345
  (norm_edit 0.995 — model emits only 1 step, didn't learn full suffix), anomaly f1 **0** (always valid).
  → heavily UNDERFIT. NOTE: JSON+reasoning format is HARDER than the old pipe format (old 100/1ep got
  0.69), so 1-epoch scaling underfits. WATCH: does the best (18k rows × 3 ep) emit full completions +
  real anomaly? If the 1-epoch scaling curve is flat/underfit, bump scaling epochs to 2–3 and rerun.

## Scaling trend (1 epoch) — data helps, anomaly needs more epochs
| size | nextstep top1 | completion block_acc | anomaly f1 |
|---|---|---|---|
| 100 | 0.365 | 0.345 | 0.0 |
| 300 | 0.435 | 0.50 | 0.0 |
| (n-gram baseline) | 0.69 | 0.637 | 0.89 |
→ TREND up for nextstep+completion (more data helps — the scaling story holds). Anomaly stuck at f1=0
for 1-epoch models (always "valid"). **The 3-epoch BEST is the linchpin**: must beat baseline on
completion (LLM's strength) + ideally learn anomaly. If best ALSO has anomaly f1=0, anomaly may need
more epochs / different LR — but the n-gram already aces anomaly, so lead the story with completion
coherence + the live demo, and present n-gram as a strong classical baseline (honest).

## Discipline
- DO NOT resubmit a job already in this table as RUNNING/COMPLETED. Check sacct first.
- Eval is cheap + fixable; training is the long pole. Validate eval on first completion, then fan out evals.
- Once matrix trained+evaled+dashboard+demo done → stop submitting; polish only.

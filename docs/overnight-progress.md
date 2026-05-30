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
| scale 2000 (1.5B) | leonardo_sft_scale_instruct_2000.yaml | 43146546 | submitted | hf-sft-scale-2000 |
| size 0.5B | (todo: leonardo_sft_fab_instruct_0_5b.yaml) | - | needs base staged | hf-sft-0_5b |
| size 3B | (todo: leonardo_sft_fab_instruct_3b.yaml) | - | needs base staged | hf-sft-3b |
| size 7B | (todo, if feasible) | - | needs base staged | hf-sft-7b |

Baselines (no training): n-gram (have: extras/results/baseline-ngram), zero-shot base (todo: judge-eval the frozen base).

## Pipeline state
- [x] Merged main JSON refactor; kept sft.py fix; corpora regenerated (JSON) + shipped to cluster.
- [x] Fixed corpus delivery (ship_corpora.sh), GPU env (uv sync --extra gpu), sbatch race (--no-sync templates).
- [x] Best + 4 scaling RUNNING cleanly on correct JSON corpora (round-2 jobs in table). scale-100 ~5min.
- [ ] **NEXT: scale-100 (43146520) done → validate JSON eval** then `pull_promote_scale.sh 100`. Confirm scorer gives sane top1/block_acc AND anomaly f1 > 0 (balanced+CoT corpus should fix the old f1=0).
- [ ] Stage base 0.5B/3B (/7B) — restarted, /tmp/stage_models.log → STAGING_DONE. Then size configs (copy instruct config, swap model path), submit --no-prep.
- [ ] Eval every checkpoint (judge-eval --no-prep) + promote → extras/results/INDEX.json.
- [ ] Dashboard: infineon-results-dashboard build-results.mjs reads INDEX.json → base-vs-best; EXTEND for scaling + size panels (the storyline).
- [ ] Update scripts/serve_copilot_mac.py to JSON framing (build_messages + parse steps[0]); pscp best ckpt to ~/zo-models; verify live demo from Mac.

## Active background waiters
- scale-100 completion waiter (fires when 43146520 terminal → run the eval validation above).
- base-model staging on cluster (/tmp/stage_models.log).

## Eval results collected (fill as evals complete)
(none yet)

## Discipline
- DO NOT resubmit a job already in this table as RUNNING/COMPLETED. Check sacct first.
- Eval is cheap + fixable; training is the long pole. Validate eval on first completion, then fan out evals.
- Once matrix trained+evaled+dashboard+demo done → stop submitting; polish only.

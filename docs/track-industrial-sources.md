# Industrial AI track — sources, gaps, and where our artifacts live

This doc is the team’s **source of truth** for what the public Lumos track repo includes vs what organizers
hand out at kickoff, and where **our** checkpoints and training logs live (not in git).

## What we have in this repository

| Source | Location | Use |
|--------|----------|-----|
| Track brief (EN) | `data/industrial-infineon/Track_industrial_en.md` | Same text as `docs/Track One Assignment.txt` |
| Track brief (DE) | `data/industrial-infineon/Track_industrial.md` | German briefing |
| Data + grammar + generator | `data/industrial-infineon/training_data/` | 3×1000 sequences, `generation_rules.md`, `generate_sequences.py` |
| Eval protocol (metrics + CSV formats) | `generation_rules.md` §5 | Authoritative **formats**; metric *definitions* for NED/token/block may differ slightly from our stand-in scorer until kickoff |
| Submission checklist | `docs/submission/SUBMISSION.md`, `REPORT_TEMPLATE.md` | Tally + repo deliverables |
| Local stand-in scorer | `packages/eval/zo_eval/track_metrics.py` | All documented task metrics + per-family (+ per-cut) breakdown |
| Local eval proxy | `extras/eval_local/` (regenerate: `just local-eval MOSFET`) | Organizer-format CSVs + `gold.json` from held-out data |

## What we do **not** have (and should not wait on to build)

| Missing item | Referenced by | What we do instead |
|--------------|---------------|-------------------|
| **`eval_metrics.py`** | Track README, `SUBMISSION.md`, `generation_rules.md` | `zo_eval/track_metrics.py` + `zo-track predict` (reconcile numbers when the official script lands) |
| **Kickoff eval inputs** | `eval_input_valid.csv` (600 rows), `eval_input_anomaly.csv` (987 rows) | `extras/eval_local/` until organizers distribute the fixed set; swap paths in `zo-track` / `just judge-eval` |
| **`judging/rubrics.md`** | `docs/submission/SUBMISSION.md` (link only) | **Not published upstream.** Judge using: general rubric in `SUBMISSION.md` (“What we judge”), track brief §7, and `generation_rules.md` §5. No per-track numeric rubric weights in repo. |

When kickoff files arrive, copy them to e.g. `extras/eval_kickoff/` (gitignored or committed per team choice), set
`ZO_JUDGE_EVAL_DIR`, and re-run inference with tag `eval-set:kickoff`.

## Official metrics (all three scored tasks)

Documented in `generation_rules.md` §5.2. Our registry / `metrics_report.md` use the same names (flat keys).

**Task 1 — next-step:** `top1`, `top3`, `top5`, `mrr` (+ per-family `_MOSFET` etc., per-cut `top1_frac60` …)

**Task 2 — completion:** `em`, `ned`, `token_acc`, `block_acc` (lead with **ned** / **block_acc** in the report; EM is often ~0)

**Task 3 — anomaly:** `anomaly_acc`, `anomaly_p`, `anomaly_r`, `anomaly_f1`, `anomaly_auc`, `rule_attr_acc`, `cm_tp/fp/tn/fn`

**Task 4 (organizers only):** ID→OOD performance drop on a hidden 4th family — no submission file; report our **LOFO** proxy from tagged runs (`split:ood,family:<held-out>`).

## Trained artifacts (not in git)

| Artifact | Location | Notes |
|----------|----------|--------|
| **Model checkpoints** | Hugging Face org **`XCombinator`** | e.g. `XCombinator/sft-fab-all`, LOFO variants, smoke LoRA repos. Upload via `scripts/leonardo_upload_artifact.sh` / training `extra.hub_model_id`. |
| **Training logs & curves** | Weights & Biases **`XCombinator/XCombinator`** | Set `WANDB_ENTITY` / `WANDB_PROJECT` in `.env`. Leonardo compute: `WANDB_MODE=offline` then `wandb sync` on login. |
| **Run registry + eval CSVs** | `experiments/<run_id>/` or `$ZO_EXPERIMENTS_DIR` | `meta.json`, `metrics.jsonl`, `results/{nextstep,completion,anomaly}.csv`, `metrics_report.json` |

Reproduce eval on a checkpoint:

```bash
# Local / login node (HF hub, no vLLM):
uv run zo-track predict -p hf --model XCombinator/sft-fab-lofo-mosfet \
  --version sft-lofo-mosfet-v1 --model-ref XCombinator/sft-fab-lofo-mosfet \
  --valid extras/eval_local/eval_input_valid.csv \
  --anomaly extras/eval_local/eval_input_anomaly.csv \
  --gold extras/eval_local/gold.json \
  --tags split:ood,family:MOSFET,eval-set:local

# Baseline (no GPU):
uv run zo-track predict -p ngram --train-families IGBT,IC \
  --version ngram-lofo-v1 --tags split:ood,family:MOSFET,predictor:ngram,eval-set:local \
  --valid ... --gold ...
```

See `docs/leonardo-eval.md` and `packages/training/configs/README.md` for cluster batch paths.

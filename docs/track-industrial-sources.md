# Industrial AI track — sources, scoring, and artifacts

> **Entire file is XCombinator-authored** (not from the Lumos/upstream hackathon pack). For inline notes
> inserted into vendored briefs, search `XCombinator-TEAM-START` or see
> [XCOMBINATOR-TEAM-ADDITIONS.md](XCOMBINATOR-TEAM-ADDITIONS.md).

Team **source of truth** for what is in the repo, how judging works, and where checkpoints/logs live.

## Judging criteria (no separate `rubrics.md`)

`docs/submission/SUBMISSION.md` links to `judging/rubrics.md`, but **that file is not published** and we do not receive it. Everything we need is already in the repo:

| Topic | Where |
|-------|--------|
| General jury bar (working artifact, honest eval, infra, no wrappers) | `docs/submission/SUBMISSION.md` — “What we judge” |
| Track problem, tasks, metrics, stretch goals | `data/industrial-infineon/Track_industrial_en.md` (= `docs/Track One Assignment.txt`) |
| CSV formats + metric names | `data/industrial-infineon/training_data/generation_rules.md` §5 |
| Data + eval overview | `data/industrial-infineon/README.md`, `training_data/README.md` |
| Repo deliverables checklist | `docs/submission/REPORT_TEMPLATE.md` |

## Scoring: what we submit vs what we run locally

**Kickoff eval inputs** (600 + 987 rows) and the official **`eval_metrics.py`** are in
[`data/industrial-infineon/eval/`](../data/industrial-infineon/eval/). Organizers hold **ground-truth
labels** for those IDs and score our three output CSVs after submission:

- `nextstep.csv` — `EXAMPLE_ID,RANK_1..RANK_5`
- `completion.csv` — `EXAMPLE_ID,PREDICTED_SEQUENCE`
- `anomaly.csv` — `EXAMPLE_ID,IS_VALID,SCORE,PREDICTED_RULE`

**Self-evaluation** options — full guide: **[eval-and-artifacts.md](eval-and-artifacts.md)**

| Scenario | Command |
|----------|---------|
| Baseline vs model matrix (labeled) | `just eval-suite packages/eval/eval_suites/local_compare.yaml --model XCombinator/...` |
| Kickoff submission + auto-promote | `just eval-suite packages/eval/eval_suites/kickoff_submit.yaml --model XCombinator/...` |
| Single run + promote | `just track "-p hf --model … -V final --eval-set kickoff --promote kickoff-final"` |
| Re-score CSVs only | `just rescore --results $ZO_EXPERIMENTS_DIR/<run_id>/results --gold extras/eval_local/gold.json --self-check` |
| Promote after the fact | `just promote kickoff-final <run_id>` or `uv run zo-track promote-wandb <wandb_run_id> --slug kickoff-final` |
| HF training params on model repo | `just hub-manifest <run_id> --hub-model-id XCombinator/...` |
| Compare index | `extras/results/INDEX.json` |

Wrong CSV columns still score zero on their side.

```bash
# Kickoff predict (defaults to data/industrial-infineon/eval/*.csv)
uv run zo-track predict -p ngram -V ngram-v1 --eval-set kickoff --tags split:id

# Local labeled proxy
uv run zo-track predict -p ngram -V ngram-v1 \
  --valid extras/eval_local/eval_input_valid.csv \
  --anomaly extras/eval_local/eval_input_anomaly.csv \
  --gold extras/eval_local/gold.json \
  --tags split:id,eval-set:local
```

## Documented metrics (Tasks 1–3)

From `generation_rules.md` §5.2 — implemented in `track_metrics.py` (aligned with vendored `eval_metrics.py`):

| Task | Metrics |
|------|---------|
| Next-step | Top-1/3/5 accuracy, MRR |
| Completion | Exact match, normalized edit distance, token accuracy, block-level accuracy |
| Anomaly | Binary accuracy, precision, recall, F1, confusion matrix, ROC-AUC, rule attribution accuracy |

Registry keys and per-family / per-cut (`_frac60`, `_frac80`) breakdowns: see `.claude/knowledge/eval.md`.

**Task 4 (OOD):** organizers only — hidden 4th family, ID→OOD drop. We report a **LOFO proxy** (`split:ood,family:<held-out>`).

## What is in the repo

| Item | Location |
|------|----------|
| Training data + grammar | `data/industrial-infineon/training_data/` |
| Kickoff eval inputs + official scorer | `data/industrial-infineon/eval/` |
| Local labeled proxy (regenerate) | `extras/eval_local/` via `just local-eval <FAMILY>` |
| Self-eval implementation | `packages/eval/zo_eval/track_metrics.py` |
| **`judging/rubrics.md`** | **Not published** — criteria in table above |

## Trained artifacts (not in git)

| Artifact | Location |
|----------|----------|
| Checkpoints | Hugging Face **`XCombinator`** |
| Training + eval metrics, eval CSV artifacts | W&B **`XCombinator/XCombinator`** |
| Local scratch (disposable) | `~/.cache/zo-experiments/` or `$SCRATCH/zo-experiments` |
| Promoted pitch CSVs | `extras/results/` |

**Production playbook:** [eval-and-artifacts.md](eval-and-artifacts.md) (train → HF → eval → W&B → promote).

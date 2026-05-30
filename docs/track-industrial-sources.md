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

**We do not receive `eval_metrics.py`.** The organizers score our three output CSVs with **their own** script after submission. Our job is to produce correctly formatted files:

- `nextstep.csv` — `EXAMPLE_ID,RANK_1..RANK_5`
- `completion.csv` — `EXAMPLE_ID,PREDICTED_SEQUENCE`
- `anomaly.csv` — `EXAMPLE_ID,IS_VALID,SCORE,PREDICTED_RULE`

**Self-evaluation** on data we hold (training hold-outs or a local proxy set) uses our implementation of the **same documented metrics** in `packages/eval/zo_eval/track_metrics.py`, driven by `zo-track predict` → `metrics_report.json` / `metrics_report.md`. Wrong CSV columns still score zero on their side; our stand-in does not change that.

```bash
uv run zo-track predict -p ngram -V ngram-v1 \
  --valid extras/eval_local/eval_input_valid.csv \
  --anomaly extras/eval_local/eval_input_anomaly.csv \
  --gold extras/eval_local/gold.json \
  --tags split:id,eval-set:local
# → experiments/<run>/results/metrics_report.md
```

Regenerate local organizer-format inputs: `just local-eval MOSFET` → `extras/eval_local/`.

If organizers distribute a **fixed kickoff eval set** (`eval_input_valid.csv`, `eval_input_anomaly.csv`), use those paths for final predictions; we still self-score only where we have labels (`gold.json` from hold-out synthesis or labels they provide).

## Documented metrics (Tasks 1–3)

From `generation_rules.md` §5.2 — implemented in `track_metrics.py`:

| Task | Metrics |
|------|---------|
| Next-step | Top-1/3/5 accuracy, MRR |
| Completion | Exact match, normalized edit distance, token accuracy, block-level accuracy |
| Anomaly | Binary accuracy, precision, recall, F1, confusion matrix, ROC-AUC, rule attribution accuracy |

Registry keys and per-family / per-cut (`_frac60`, `_frac80`) breakdowns: see `.claude/knowledge/eval.md`.

**Task 4 (OOD):** organizers only — hidden 4th family, ID→OOD drop. We report a **LOFO proxy** (`split:ood,family:<held-out>`).

## What is in the repo vs optional at kickoff

| Item | In public repo? | Our approach |
|------|-----------------|--------------|
| Training data + grammar | Yes — `data/industrial-infineon/` | |
| Eval protocol (formats + metric names) | Yes — `generation_rules.md` §5 | |
| **`eval_metrics.py`** | **No — not given to teams** | `track_metrics.py` for self-eval |
| **`judging/rubrics.md`** | **No** | Criteria in table above |
| Fixed kickoff `eval_input_*.csv` | Often distributed at event | `extras/eval_local/` until then |

## Trained artifacts (not in git)

| Artifact | Location |
|----------|----------|
| Checkpoints | Hugging Face **`XCombinator`** |
| Training logs / loss curves | W&B **`XCombinator/XCombinator`** |
| Eval CSVs + reports | `experiments/<run_id>/results/` or `extras/results/` |

See `docs/leonardo-eval.md`, `packages/training/configs/README.md`, `.env.example`.

# Eval

Code: `packages/eval/zo_eval/` (`tasks.py`, `harness.py`, `cli.py`). Tasks: `packages/eval/tasks/*.yaml`.

> **Track reality ([track-industrial-ai.md](track-industrial-ai.md)):** the scored eval is **3 process
> tasks** (next-step / completion / anomaly). **`eval_metrics.py` and kickoff CSVs are not in the public
> repo** — use `track_metrics.py` + `extras/eval_local/` until kickoff. **`judging/rubrics.md` is also
> unpublished**; see [docs/track-industrial-sources.md](../../docs/track-industrial-sources.md).
> Checkpoints: HF **`XCombinator`**; logs: W&B **`XCombinator/XCombinator`**. Not the OpenAI harness below.

## How it works
- A **task** is a YAML: an id, a list of items (`prompt` + expected answer), and a `metric`.
- `metric` ∈ `exact_match | contains | numeric | regex` (see `score()` in `tasks.py`).
- `run_eval(task, model, base_url, run_id, limit, temperature)` iterates items, calls the model via
  the OpenAI-compatible client (`zo_common.llm.chat`), scores each, `append_metric`s per item, and
  writes a final `accuracy`. Everything lands in the run registry like a training run.

## Running
- `just eval <task> <model>` → `uv run zo-eval run --task <task.yaml> --model <model>`.
- Flags: `--task/-t`, `--model/-m`, `--base-url` (defaults to `ZO_MODEL_BASE_URL`,
  `http://localhost:8001/v1`), `--limit`.
- The model must be served at an OpenAI-compatible endpoint. Start one with `just serve <model>`
  (vLLM) — that's how you eval your **fine-tuned** checkpoint: serve it, then point eval at it.

## Adding a task
Drop a new YAML in `packages/eval/tasks/`. Example: `tasks/example.yaml` = `arithmetic-tiny`
(toy task where every answer is `42`). Copy it, change items + metric.

## Append below as you learn

### Track submission I/O + local scorer (2026-05-30) — `zo_eval/submission.py` + `zo_eval/track_metrics.py`
- **`submission.py`** = the EXACT organizer formats (format is what's scored — wrong columns = 0):
  `read_valid_inputs` / `read_anomaly_inputs` (kickoff CSVs), `write_nextstep` / `write_completion` /
  `write_anomaly` (the 3 submission files → `extras/results/`), and `make_local_eval_set(valid_seqs,
  negatives, out)` which synthesizes organizer-format eval inputs + `gold.json` from held-out data so
  the full input→predict→write→score path runs **before kickoff** (e.g. on a LOFO test family).
- **`track_metrics.py`** = stand-in scorer for every documented metric (Top-1/3/5+MRR; ExactMatch/
  NormEditDist/TokenAcc/BlockAcc; BinAcc/P/R/F1/confusion/ROC-AUC/RuleAttribution). Anomaly positive
  class = INVALID; ROC-AUC uses SCORE=P(valid). `per_family(score_fn, preds, gold, family_of)` gives
  the required per-family breakdown. **Authoritative scorer = organizers' `eval_metrics.py` at
  kickoff**; NormEditDist/TokenAcc/BlockAcc are sensible stand-in definitions to reconcile then.
- Verified end-to-end: perfect preds → top1/EM/F1/AUC = 1.0 (`python -m zo_eval.submission`).

### Inference/predict core (2026-05-30) — `predict.py` + `baselines.py` + `track.py`/`track_cli.py`
The shared, model-agnostic path every stream uses (there was NO inference code before this).
- **`predict.py`** — `Predictor` Protocol (`next_step`/`complete`/`anomaly`) + the **output
  normalizer**: `snap(text, vocab, strict)` (free text → exact step; `strict=False` passes novel OOD
  tokens through), `parse_pipe_list`, `extract_answer` (strip `<think>`), `parse_anomaly`. Owns the
  separator translation: prompts use `" | "` (`datagen.SEP`), CSVs use `"|"` (`submission.STEP_SEP`).
- **`baselines.py`** — `NGramPredictor` (back-off; restrict to LOFO train families for OOD),
  `OraclePredictor` (`validate_sequence` — the **submitted** anomaly path, ~100%), `FreqPredictor`.
- **`track.py` / `track_cli.py`** (`zo-track`, `just track` / `just local-eval`) — run a predictor →
  write the 3 CSVs (namespaced `experiments/<run>/results/`) → score+`per_family` → flat tagged
  scalars in the registry. Driver mirrors the organizers' `eval_metrics.py` (drop-in at kickoff).
- **Metric/tag convention (dashboard contract):** flat scalars `top1/top3/top5/mrr`,
  `em/ned/token_acc/block_acc`, `anomaly_acc/anomaly_p/anomaly_r/anomaly_f1/anomaly_auc/rule_attr_acc`,
  `cm_tp/fp/tn/fn`; per-family adds `_MOSFET|_IGBT|_IC`; per-cut adds `_frac60` / `_frac80` (60%/80%).
  **ID vs OOD is a run TAG** (`split:id|ood`), not a metric name. Required repro tags:
  `version:<label>`, `model-ref:<hf-repo-or-path>`, `eval-set:local|kickoff`, plus
  `predictor:ngram|oracle|freq|hf|llm|featherless|likelihood-ngram|classifier`.
- Each eval run writes `results/metrics_report.json` + `metrics_report.md` (paste into REPORT.md).
- Predictors for reproducible matrix: `ngram`/`freq` (baselines), `oracle` (submitted anomaly),
  `hf`/`llm`/`featherless` (finetuned), `likelihood-ngram` (learned anomaly science), `classifier` (LLM verdict).
- `zo-eval` now depends on `zo-train` (declared) so it can import grammar/fab/datagen.

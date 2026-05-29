# Eval

Code: `packages/eval/zo_eval/` (`tasks.py`, `harness.py`, `cli.py`). Tasks: `packages/eval/tasks/*.yaml`.

> **Track reality ([track-industrial-ai.md](track-industrial-ai.md)):** the scored eval is **3 process
> tasks** (next-step / completion / anomaly) graded by the organizer-provided `eval_metrics.py`
> against held-out inputs — **not** the OpenAI `exact_match` harness below. Plan to wrap
> `eval_metrics.py` as zo-eval tasks (and use `validate_sequence()` as a rule-based anomaly baseline).
> The harness pattern below (iterate items → score → `append_metric` → final number) is still the
> right shape; just swap the scoring + drop the served-model requirement for a from-scratch seq-model.

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

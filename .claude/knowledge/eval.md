# Eval

Code: `packages/eval/zo_eval/` (`tasks.py`, `harness.py`, `cli.py`). Tasks: `packages/eval/tasks/*.yaml`.

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
- (real tasks we built and their baselines: TBD)

---
description: Evaluate a served model on a task and report accuracy vs baseline
argument-hint: <task.yaml> <model> [--base-url URL]
allowed-tools: Bash(just eval:*), Bash(uv run zo-eval:*), Bash(uv run zo-runs:*), Read
---
Run an eval: $ARGUMENTS

1. Confirm a model is served at an OpenAI-compatible endpoint (`ZO_MODEL_BASE_URL`, default
   `http://localhost:8001/v1`). If nothing is serving, tell the user to `just serve <model>` first.
2. Run `just eval <task> <model>` (append `--base-url <url>` if provided).
3. Report the final `accuracy` and the run id. Compare against any prior baseline recorded in
   `.claude/knowledge/eval.md`; if there's none, note this as the baseline. Point out where the model
   failed if the per-item results make it obvious.

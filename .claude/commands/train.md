---
description: Run a training config locally — dry-run first, then for real
argument-hint: <config-path> [--real]
allowed-tools: Bash(just train:*), Bash(just grpo:*), Bash(uv run zo-train:*), Bash(uv run zo-runs:*), Read
---
Run training for: $ARGUMENTS

1. Read the config to determine its `kind` (sft or grpo).
2. Unless `--real` is passed, do a **dry run first**: `just train <cfg> --dry-run`
   (or `just grpo <cfg> --dry-run`). Then `uv run zo-runs ls` to confirm metrics were written.
3. If the dry run looks right AND this machine has the `[gpu]` extra synced, run for real
   (same command without `--dry-run`). If there's no local GPU, do NOT attempt a real run — tell the
   user to use `/cluster <cfg>` to submit to Leonardo instead.
4. Report the run id and how to view it: the dashboard (`just dev`) or `uv run zo-runs show <id>`.

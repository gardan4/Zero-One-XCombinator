---
description: Scaffold a new training experiment (config YAML, optionally a worktree)
argument-hint: <sft|grpo> <name> [--worktree]
allowed-tools: Read, Write, Edit, Bash(just wt:*), Bash(./scripts/wt.sh:*), Bash(ls:*)
---
Scaffold a new experiment for: $ARGUMENTS

1. Parse a `kind` (`sft` or `grpo`) and a short `name` from the arguments.
2. Copy the closest existing config in `packages/training/configs/` as a starting point —
   `sft_smoke.yaml` for sft, `grpo_example.yaml` for grpo — to `packages/training/configs/<name>.yaml`.
3. Edit the new config: set `name`, `kind`, and a small `model` for fast iteration
   (e.g. `Qwen/Qwen2.5-0.5B-Instruct`). If the dataset/reward isn't obvious from context, ask.
4. If `--worktree` is in the args, run `just wt <name>` to get an isolated checkout + branch.
5. Tell the user the exact next step and stop — do NOT start training:
   `just train packages/training/configs/<name>.yaml --dry-run` (or `just grpo ...`). Once the dry
   run looks right, drop `--dry-run` for a local GPU run or use `/cluster` to submit to Leonardo.

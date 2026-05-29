---
description: Submit a training config to the Leonardo SLURM cluster and watch it
argument-hint: <config-path>
allowed-tools: Bash(just submit:*), Bash(just cluster-watch), Bash(uv run zo-cluster:*), Read
---
Submit to the cluster: $ARGUMENTS

1. Read `.claude/knowledge/cluster.md` for current known-good host/account/partition/QOS.
2. Check `.env` has real (non-placeholder) values for `ZO_CLUSTER_USER` and `ZO_SLURM_ACCOUNT`
   (and `ZO_SLURM_QOS` if the cluster needs it). If any are blank or still guesses, **STOP and ask
   the user** — do not submit with placeholder settings.
3. Submit: `just submit <cfg>`. Capture the run id and the returned SLURM job id.
4. `just cluster-watch` to show the queue.
5. Remind the user: the dashboard only sees this run if `ZO_EXPERIMENTS_DIR` points at shared scratch
   (see cluster.md). This spends shared, budgeted GPU time — submit deliberately, one clean job.

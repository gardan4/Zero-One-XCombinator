# experiments/

The **run registry**. One directory per run; written by the train/eval/agent CLIs, read by the
backend + dashboard. Contents are gitignored (this README and `.gitkeep` are the only tracked files).

```
experiments/
  <YYYYMMDD_HHMMSS>_<kind>_<slug>/
    meta.json       # RunMeta: id, name, kind, status, git info, slurm_job_id, config, metrics summary
    metrics.jsonl   # one JSON object per logged step (append-only)
    config.yaml     # the exact config this run used
    logs/           # free-form logs
    artifacts/      # checkpoints, samples, etc. (also gitignored)
```

Created via `zo_common.registry.new_run(...)`; metrics appended with `append_metric(run_id, step, **m)`.
Override the location with `ZO_EXPERIMENTS_DIR`. **Default scratch is now
`~/.cache/zo-experiments`** (outside the repo). Set `ZO_EXPERIMENTS_DIR=./experiments` in `.env`
to use this legacy in-repo folder. Durable metrics live in **W&B**; weights on **Hugging Face**;
pitch artifacts in **`extras/results/`** (via `zo-track promote-wandb`).

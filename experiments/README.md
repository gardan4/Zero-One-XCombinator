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
Override the location with `ZO_EXPERIMENTS_DIR` (point at shared scratch on the cluster so the
dashboard sees cluster runs). Inspect with `just runs` / `uv run zo-runs show <id>`.

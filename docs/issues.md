# Known issues (main branch review)

Review date: 2026-05-30. Focus: obvious oversights that can cause wrong runs, failed jobs, or misleading dashboard state during the hackathon. Not a full audit.

---

## Critical

### 1. Cluster submit ignores YAML `kind` — GRPO configs run as SFT

`just submit` / `zo-cluster submit` always use the CLI `--kind` flag (default `sft`). It never reads `kind:` from the experiment YAML.

**Impact:** `just submit packages/training/configs/grpo_fab.yaml` renders `zo-train sft …` instead of `zo-train grpo …`. Wrong trainer, wrong dataset loader, wrong `RunMeta.kind`.

**Workaround:** pass `--kind grpo` manually until fixed.

**Fix:** read `cfg.kind` from `ExperimentConfig` in `packages/training/zo_train/cluster/submit.py` (and/or teach `just submit` to pass it).

---

### 2. Cluster jobs are HF-offline but fab configs use hub model IDs

`train.sbatch.j2` sets `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1`. Configs like `sft_fab.yaml` and `grpo_fab.yaml` still use `model: Qwen/Qwen2.5-1.5B-Instruct`.

**Impact:** cluster training fails on model load unless the checkpoint is pre-staged to a local path (Leonardo smoke configs do this; main fab configs do not).

**Workaround:** pre-download to scratch and point `model` at the local directory with `extra.local_files_only: true`, or use the Leonardo-specific configs.

---

### 3. Generated training data may be missing on the cluster

Fab configs depend on `data/generated/*_sft_lm.jsonl` etc. Those files are gitignored (`data/generated/*`; only `splits.json` / `manifest.json` are tracked).

**Impact:** cluster jobs fail with `FileNotFoundError` if `uv run python -m zo_train.datagen --build` was not run locally before sync/submit.

**Workaround:** run datagen locally, confirm JSONL files exist, then sync the repo to the cluster.

---

### 4. `sft_cot.yaml` will fail on a real GPU run

The config expects `text_field: text`, but `datagen` writes CoT rows as `{prompt, completion, …}` only — no `text` column.

**Impact:** real SFT run errors or trains on empty/missing text. `--dry-run` still passes, so this is easy to miss.

**Workaround:** add a `text = prompt + completion` field in datagen, or change `sft.py` to support prompt/completion columns (see comments in `sft_cot.yaml`).

---

### 5. No checkpoints until the very end

Both SFT and GRPO use `save_strategy="no"` and only call `save_model()` at the end of training.

**Impact:** SLURM timeout, preemption, or late OOM loses all weights. `metrics.jsonl` may exist but there is no usable checkpoint.

**Workaround:** keep runs short (`max_steps` smoke) until checkpointing is enabled; don't assume you can resume.

---

### 6. `judge-eval` creates a run, but the SLURM job creates a second one

`judge_eval()` registers a run and writes `infer.sbatch`, but `infer.sbatch.j2` calls `zo-track predict` without `--run-id`. `run_track()` then creates a new run for metrics/status.

**Impact:** the pre-created run stays `queued` forever; CSVs may land in its `results/` dir while metrics attach to a different run. Dashboard looks broken for cluster judge evals.

**Fix:** thread `--run-id` through `infer.sbatch.j2`, `track_cli predict`, and `run_track()`.

---

## High

### 7. Local vs cluster experiment dirs diverge

Submit writes `meta.json` locally under `experiments/`, while the GPU job uses `ZO_CLUSTER_EXPERIMENTS_DIR` on scratch.

**Impact:** local dashboard shows runs stuck at `queued` after cluster jobs finish, unless both sides share the same `ZO_EXPERIMENTS_DIR`.

**Workaround:** set `ZO_EXPERIMENTS_DIR` to the same scratch path everywhere (laptop `.env`, backend, cluster sbatch).

---

### 8. Backend does not load `ZO_EXPERIMENTS_DIR` from `.env`

Training CLIs call `load_dotenv()`, but the backend only reads `ZO_API_*` via Pydantic settings. `ZO_EXPERIMENTS_DIR` in `.env` is not applied to `experiments_dir()` unless exported in the shell.

**Impact:** `just dev` dashboard reads `./experiments` even when `.env` points at cluster scratch.

**Fix:** call `zo_common.env.load_dotenv()` in the backend startup, or add `experiments_dir` to backend settings.

---

### 9. `--dry-run` validates almost nothing

`simulate_training()` always writes 20 fake metric steps. It never loads the dataset, checks the model path, or reads YAML hyperparameters.

**Impact:** `just train … --dry-run` passing does **not** mean a real GPU run will work.

**Workaround:** treat dry-run as pipeline smoke only; do one short real run (`max_steps: 50` is already in fab configs).

---

### 10. `grpo_fab.yaml` still points at the base model

The config has an explicit TODO: `model` is still `Qwen/Qwen2.5-1.5B-Instruct`, not the SFT checkpoint output directory.

**Impact:** GRPO starts from the wrong weights — won't crash, but wastes GPU time and won't match the intended pipeline.

**Workaround:** replace `model` with `experiments/<sft_run_id>/artifacts` after Stream-1 SFT completes.

---

## Suggested fix order (before burning GPU time)

1. Wire submit `kind` from YAML (or always pass `--kind grpo` for GRPO configs).
2. Run datagen and confirm JSONL paths exist before cluster submit.
3. Pre-stage models for offline cluster jobs.
4. Align `ZO_EXPERIMENTS_DIR` between backend and cluster.
5. Fix judge-eval `--run-id` if cluster batch eval must show up in the dashboard.
6. Don't treat `--dry-run` as config validation.

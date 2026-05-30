# Known issues (main branch review)

Review date: 2026-05-30. Branch **`fix/known-issues`** addresses the items below.

Legend: **Verified** = confirmed in code on main before the fix branch. **Fixed** = change landed on `fix/known-issues`.

---

## Critical

### 1. Cluster submit ignores YAML `kind` — GRPO configs run as SFT

| | |
|---|---|
| **Verified** | Yes — `submit_run()` used CLI `kind` (default `"sft"`) and never read `cfg.kind`. |
| **Fixed** | `submit_run()` now uses `_effective_kind(cfg, kind)` — YAML `kind` wins; CLI `--kind` is an optional override. Added test `test_submit_uses_kind_from_yaml`. |
| **Files** | `packages/training/zo_train/cluster/submit.py` |

---

### 2. Cluster jobs are HF-offline but fab configs use hub model IDs

| | |
|---|---|
| **Verified** | Yes — `train.sbatch.j2` sets `TRANSFORMERS_OFFLINE=1` / `HF_HUB_OFFLINE=1`; fab configs used bare hub ids. |
| **Fixed** | (a) `sft_fab.yaml` / `grpo_fab.yaml` now use `${ZO_BASE_MODEL_DIR}` / `${ZO_SFT_CHECKPOINT_DIR}` with docs in `.env.example`. (b) `ExperimentConfig.from_yaml()` expands `${VAR}` via `zo_common.env.expand_env_refs`. (c) `validate_experiment(..., cluster=True)` rejects HF hub ids at submit time with an actionable error. |
| **Files** | `packages/common/zo_common/config.py`, `packages/common/zo_common/env.py`, `packages/training/zo_train/preflight.py`, `packages/training/configs/sft_fab.yaml`, `packages/training/configs/grpo_fab.yaml`, `.env.example` |

---

### 3. Generated training data may be missing on the cluster

| | |
|---|---|
| **Verified** | Yes — `data/generated/*` is gitignored; fab configs reference local JSONL paths. |
| **Fixed** | `validate_experiment()` checks that local dataset paths resolve before cluster submit (and on dry-run). Error message points to `zo_train.datagen --build`. Repo sync already tars `data/` when files exist locally — no code change needed there. |
| **Files** | `packages/training/zo_train/preflight.py`, `packages/training/zo_train/cluster/submit.py` |

---

### 4. `sft_cot.yaml` will fail on a real GPU run

| | |
|---|---|
| **Verified** | Yes — `anomaly_example()` returned `{prompt, completion, …}` only; config expects `text_field: text`. |
| **Fixed** | CoT rows now include `text = prompt + completion`. Test `test_cot_anomaly_rows_include_text`. Re-run datagen to regenerate JSONL if you already built the corpus. |
| **Files** | `packages/training/zo_train/datagen.py` |

---

### 5. No checkpoints until the very end

| | |
|---|---|
| **Verified** | Yes — both SFT and GRPO used `save_strategy="no"`. |
| **Fixed** | `checkpoint_kwargs()` defaults to `save_strategy=steps`, `save_steps=50`, `save_total_limit=3`. Disable with `extra.save_steps: 0`. |
| **Files** | `packages/training/zo_train/preflight.py`, `packages/training/zo_train/sft.py`, `packages/training/zo_train/rl.py` |

---

### 6. `judge-eval` creates a run, but the SLURM job creates a second one

| | |
|---|---|
| **Verified** | Yes — `infer.sbatch.j2` called `zo-track predict` without `--run-id`; `track_cli` had no `--run-id` flag. |
| **Fixed** | Threaded `run_id` through `judge_eval()` → `infer.sbatch.j2` → `zo-track predict --run-id`. Updated `test_render_infer_sbatch`. |
| **Files** | `packages/training/zo_train/cluster/judge.py`, `packages/training/zo_train/cluster/slurm/infer.sbatch.j2`, `packages/eval/zo_eval/track_cli.py` |

---

## High

### 7. Local vs cluster experiment dirs diverge

| | |
|---|---|
| **Verified** | Yes — submit writes local `experiments/`; GPU job uses `ZO_CLUSTER_EXPERIMENTS_DIR` on scratch. Fully unifying paths from a laptop is not possible. |
| **Fixed (partial)** | (a) Submit stores the cluster run dir in `meta.notes`. (b) New `zo-cluster pull-run <id>` / `just cluster-pull-run` SCPs `meta.json` + `metrics.jsonl` from cluster scratch to the local registry. (c) Dry-run status is now `created` (not misleading `queued`). |
| **Files** | `packages/training/zo_train/cluster/submit.py`, `packages/training/zo_train/cluster/_remote.py` (`scp_download`), `Justfile` |

---

### 8. Backend does not load `ZO_EXPERIMENTS_DIR` from `.env`

| | |
|---|---|
| **Verified** | Yes — backend never called `load_dotenv()`; Pydantic settings only mapped `ZO_API_*`. |
| **Fixed** | `load_dotenv()` at top of `apps/backend/zo_backend/main.py` before router import. |
| **Files** | `apps/backend/zo_backend/main.py` |

---

### 9. `--dry-run` validates almost nothing

| | |
|---|---|
| **Verified** | Yes — `simulate_training()` always wrote 20 fake steps with no config checks. |
| **Fixed** | Dry-run now calls `validate_experiment(cfg, cluster=False)` (dataset paths, unexpanded env refs). Still does not load torch/model weights — by design. Respects `extra.max_steps` for simulated step count. |
| **Files** | `packages/training/zo_train/sim.py`, `packages/training/zo_train/sft.py`, `packages/training/zo_train/rl.py`, `packages/training/zo_train/preflight.py` |

---

### 10. `grpo_fab.yaml` still points at the base model

| | |
|---|---|
| **Verified** | Yes — intentional placeholder/TODO, not a code bug, but easy to miss. |
| **Fixed** | Config uses `${ZO_SFT_CHECKPOINT_DIR}`; `.env.example` documents it. Preflight fails if unset. For dry-run before SFT exists, temporarily set the var to the base instruct id. |
| **Files** | `packages/training/configs/grpo_fab.yaml`, `.env.example` |

---

## Suggested fix order (before burning GPU time)

1. ~~Wire submit `kind` from YAML~~ ✓
2. ~~Run datagen and confirm JSONL paths exist before cluster submit~~ ✓ (preflight enforces)
3. ~~Pre-stage models for offline cluster jobs~~ ✓ (env vars + preflight)
4. ~~Align `ZO_EXPERIMENTS_DIR` between backend and cluster~~ ✓ (backend dotenv; use `cluster-pull-run` for meta)
5. ~~Fix judge-eval `--run-id`~~ ✓
6. ~~Don't treat `--dry-run` as full validation~~ ✓ (partial validation added; still not a GPU substitute)

---

## New commands

```bash
just cluster-pull-run <run_id>   # sync meta + metrics from cluster scratch
uv run zo-cluster pull-run <run_id>
```

## Tests added

- `tests/test_preflight.py` — dataset / hub-id / CoT `text` checks
- `tests/test_cluster_training.py::test_submit_uses_kind_from_yaml`
- `tests/test_cluster_judge.py` — `--run-id` in infer sbatch

Run: `just test` (54 tests, excluding live Featherless integration).

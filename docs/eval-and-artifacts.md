# Eval, tagging, and Hugging Face artifacts

Team guide for **running eval**, **tracking models with tags**, and **reading training params on HF**.

Related: [track-industrial-sources.md](track-industrial-sources.md) · [leonardo-eval.md](leonardo-eval.md) ·
[data/industrial-infineon/eval/README.md](../data/industrial-infineon/eval/README.md) ·
[packages/training/configs/README.md](../packages/training/configs/README.md)

---

## Where results live

| Layer | What goes here | When to use |
|-------|----------------|-------------|
| **W&B** | Training metrics, eval metrics, eval CSV artifacts | Internal compare, loss curves, all experiments |
| **Hugging Face** | Model weights, `training_manifest.json`, model card | Checkpoint hosting and provenance |
| **Local scratch** | `meta.json`, `metrics.jsonl`, result CSVs (cache) | Offline dev, tests; disposable |
| **`extras/results/`** | Promoted final CSVs + reports | Pitch / REPORT / git commit |

Default scratch: **`~/.cache/zo-experiments/`** (override with `ZO_EXPERIMENTS_DIR` in `.env`).
See [experiments/README.md](../experiments/README.md).

Dashboard backend source (`ZO_RESULTS_SOURCE` in `.env`):

| Mode | Reads from |
|------|------------|
| `local` | Local scratch registry (default) |
| `wandb` | W&B project (server-side API; needs `WANDB_API_KEY` on backend) |
| `repo` | `extras/results/INDEX.json` (pitch mode) |

Infineon dashboard Compare tab has the same **Source** toggle (Local / W&B / GitHub final).

---

## Production run playbook

Follow this for any run you care about. Smoke tests can skip `real-run` / `reportable`.

### 0. One-time `.env`

```bash
cp .env.example .env
```

Set at minimum:

| Variable | Purpose |
|----------|---------|
| `WANDB_API_KEY` | Log train + eval to W&B (leave blank = local-only) |
| `WANDB_ENTITY` / `WANDB_PROJECT` | Default `XCombinator` / `XCombinator` |
| `HF_TOKEN` | Upload / download private `XCombinator/*` models |
| `ZO_EXPERIMENTS_DIR` | Optional; default is `~/.cache/zo-experiments` |
| `ZO_RESULTS_SOURCE` | Backend dashboard: `local`, `wandb`, or `repo` |

On Leonardo compute nodes use `WANDB_MODE=offline`; sync from the login node after the job:

```bash
WANDB_MODE=online uv run wandb sync $ZO_EXPERIMENTS_DIR/<run_id>/wandb/offline-run-*
```

### 1. Training config — tags + HF identity

Add to your YAML (`packages/training/configs/…`):

```yaml
extra:
  hub_model_id: XCombinator/sft-fab-lofo-mosfet
  push_to_hub: true          # or upload later with leonardo_upload_artifact.sh
  hub_private: true
  tags: [real-run, reportable, version:sft-lofo-mosfet-v1, leonardo, sft]
  hub_tags: [real-run, reportable]   # copied to HF model card
  hub_notes: "LOFO MOSFET; 2 epochs"
```

Training logs to W&B when `WANDB_API_KEY` is set (run id = registry `run_id`).

### 2. Train

```bash
# Laptop dry-run (no GPU)
just train packages/training/configs/sft_smoke.yaml --dry-run

# Leonardo
just submit packages/training/configs/sft_fab_lofo_mosfet.yaml
# note the run_id from output
```

### 3. Upload weights to Hugging Face

If not using `push_to_hub: true` during training, on a Leonardo **login node**:

```bash
bash scripts/leonardo_upload_artifact.sh <train_run_id> XCombinator/sft-fab-lofo-mosfet
```

This writes `training_manifest.json` + `README.md`, uploads weights, and logs HF repo/revision back to the training W&B run when credentials are set.

### 4. Eval (labeled local proxy)

Link eval to training with `--train-run` and matching tags:

```bash
just local-eval MOSFET

just track "-p hf \
  --model XCombinator/sft-fab-lofo-mosfet \
  --model-ref XCombinator/sft-fab-lofo-mosfet \
  --version sft-lofo-mosfet-v1 \
  --train-run <train_run_id> \
  --valid extras/eval_local/eval_input_valid.csv \
  --anomaly extras/eval_local/eval_input_anomaly.csv \
  --gold extras/eval_local/gold.json \
  --self-check \
  --tags real-run,reportable,split:ood,family:MOSFET"
```

Writes three CSVs + `metrics_report.md`, logs metrics + `eval-results` artifact to W&B (unless `--no-wandb`).

### 5. Eval on Leonardo (GPU batch)

```bash
just judge-setup    # once
just judge-stage    # download HF weights to scratch

just judge-eval --local \
  --model XCombinator/sft-fab-lofo-mosfet \
  --train-run <train_run_id> \
  --tags real-run,reportable,version:sft-lofo-mosfet-v1,split:ood,family:MOSFET
```

Kickoff eval (no public labels — proxy metrics only, tagged `proxy-only`):

```bash
just judge-eval --local \
  --model XCombinator/sft-fab-lofo-mosfet \
  --eval-set kickoff \
  --train-run <train_run_id> \
  --tags real-run,reportable,version:sft-lofo-mosfet-v1,split:id \
  --promote kickoff-final
```

### 6. Verify

```bash
just runs                              # local scratch
just runs -- --tag real-run            # filter
uv run zo-runs show <run_id>           # full meta + config
```

- **W&B:** project `XCombinator/XCombinator` — train run + eval run, eval artifact with CSVs.
- **HF:** `https://huggingface.co/XCombinator/<model>` — README + `training_manifest.json`.
- **Dashboard:** set `ZO_RESULTS_SOURCE=wandb`, restart backend, Compare → Source = **W&B results**.

### 7. Promote cherry-picked finals (pitch only)

```bash
# From local scratch
just promote kickoff-final <eval_run_id>

# From W&B artifact
uv run zo-track promote-wandb <wandb_run_id> --slug kickoff-final
```

Commit `extras/results/` for the pitch. Do **not** commit scratch runs or W&B exports.

---

## Tagging reference

**Production runs** should include:

| Tag | Meaning |
|-----|---------|
| `real-run` | Real experiment (not smoke/test) |
| `reportable` | Show in default dashboard / W&B filters |
| `version:<id>` | Repro label (matches `--version`) |
| `model-ref:<org--name>` | HF repo (`/` → `--` in tag) |
| `train-run:<run_id>` | Set via `--train-run` on eval |
| `eval-set:local\|kickoff` | Which eval inputs |
| `split:id\|ood` | In-distribution vs held-out family |
| `family:MOSFET` etc. | Per-family eval |
| `role:baseline\|finetuned` | Compare table role |

**Auto-tags** (you do not set these manually):

| Tag | When |
|-----|------|
| `test` | pytest |
| `smoke` / `dry-run` | smoke configs |
| `debug` | dashboard ad-hoc inference |
| `proxy-only` | kickoff / no-gold eval (not accuracy) |

Default dashboard filters **hide** `test`, `smoke`, `debug`, and `proxy-only`.

Extra free-form tags (`attempt-3`, `lr-1e-5`, …) are fine — add them after the required ones.

---

## Eval workflow (three tasks → three CSVs)

Organizers grade three submission files (`nextstep.csv`, `completion.csv`, `anomaly.csv`).
Formats: `data/industrial-infineon/training_data/generation_rules.md` §5.

| Eval set | Inputs | Labels | Use for |
|----------|--------|--------|---------|
| **Kickoff** | `data/industrial-infineon/eval/` | Organizers only | Final submission CSVs |
| **Local proxy** | `extras/eval_local/` (+ `gold.json`) | We hold labels | REPORT numbers, baselines |

### Comparison suites (YAML)

```bash
just eval-suite packages/eval/eval_suites/local_compare.yaml --model XCombinator/sft-fab-all
just eval-suite packages/eval/eval_suites/kickoff_submit.yaml --model XCombinator/sft-fab-all
```

### Dashboard inference

From Next.js **`/inference`** or Infineon dashboard inference panel. Ad-hoc runs get `debug` unless you pass `real-run` in tags. For production eval, use `zo-track predict` or `judge-eval` instead.

### Validate / rescore

```bash
just validate-completion <scratch>/<run_id>/results/completion.csv
just rescore --results <scratch>/<run_id>/results --gold extras/eval_local/gold.json --self-check
```

---

## Hugging Face artifacts

Every HF upload writes into the checkpoint folder before push:

| File | Contents |
|------|----------|
| `training_manifest.json` | Run id, git SHA, SLURM job, tags, full training YAML, W&B URL when available |
| `README.md` | HF model card (searchable tags + hyperparameter table) |

Regenerate locally: `just hub-manifest <run_id> --hub-model-id XCombinator/...`

---

## Reporting API

| Endpoint | Purpose |
|----------|---------|
| `GET /api/compare/report?source=wandb` | Normalized compare rows from W&B |
| `GET /api/compare/report?source=repo` | Promoted pitch results |
| `GET /api/compare/report?source=local` | Local scratch (default) |
| `POST /api/compare/refresh?source=wandb` | Bust W&B cache |
| `GET /api/compare/examples?run_a=&run_b=` | Side-by-side examples (needs local `examples.jsonl`) |

Query filters: `only_reportable`, `include_tests`, `include_proxy`, `split`, `family`, `role`, …

---

## Command reference

| Command | What it does |
|---------|--------------|
| `just submit <config>` | SLURM training job |
| `bash scripts/leonardo_upload_artifact.sh <run_id> <hf_repo>` | HF upload + W&B HF metadata |
| `just judge-eval --local …` | GPU batch eval on Leonardo |
| `just track "…"` | `zo-track predict` (local or kickoff) |
| `just local-eval FAMILY` | Build `extras/eval_local/` + `gold.json` |
| `just eval-suite <yaml>` | Tagged comparison matrix |
| `just rescore …` | Score CSVs without re-inference |
| `just promote <slug> <run_id>` | Copy scratch results → `extras/results/` |
| `uv run zo-track promote-wandb <id> --slug …` | W&B artifact → `extras/results/` |
| `just runs [-- --tag X]` | List/filter local scratch runs |
| `just hub-manifest <run_id>` | Write HF README + manifest locally |

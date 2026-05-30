# Eval, tagging, and Hugging Face artifacts

Team guide for **running eval**, **tracking models with tags**, and **reading training params on HF**.

Related: [track-industrial-sources.md](track-industrial-sources.md) · [leonardo-eval.md](leonardo-eval.md) ·
[data/industrial-infineon/eval/README.md](../data/industrial-infineon/eval/README.md) ·
[packages/training/configs/README.md](../packages/training/configs/README.md)

---

## 1. Eval workflow (three tasks → three CSVs)

Organizers grade three submission files (`nextstep.csv`, `completion.csv`, `anomaly.csv`).
Formats are defined in `data/industrial-infineon/training_data/generation_rules.md` §5.

| Eval set | Inputs | Labels | Use for |
|----------|--------|--------|---------|
| **Kickoff** | `data/industrial-infineon/eval/` | Organizers only | Final submission CSVs |
| **Local proxy** | `extras/eval_local/` (+ `gold.json`) | We hold labels | REPORT numbers, baselines |

### One-off predict

```bash
# Kickoff (no public labels — writes CSVs + proxy_report.json)
just kickoff-predict "-p hf --model XCombinator/sft-fab-all -V final-v1 \
  --train-run <training_run_id> --tags split:id,report:final \
  --promote kickoff-final"

# Local labeled eval (scores + metrics_report.md)
just local-eval MOSFET
just track "-p hf --model XCombinator/sft-fab-lofo-mosfet -V v1 \
  --valid extras/eval_local/eval_input_valid.csv \
  --anomaly extras/eval_local/eval_input_anomaly.csv \
  --gold extras/eval_local/gold.json \
  --self-check --tags split:ood,family:MOSFET"
```

### Comparison suites (YAML)

```bash
# Baselines + optional finetuned model on local gold
just eval-suite packages/eval/eval_suites/local_compare.yaml --model XCombinator/sft-fab-all

# Kickoff submission + auto-promote to extras/results/kickoff-final/
just eval-suite packages/eval/eval_suites/kickoff_submit.yaml --model XCombinator/sft-fab-all
```

Suite summary: `extras/results/suite_<name>/suite_summary.json`

### Validate without organizer labels

```bash
# Grammar-check kickoff completions (validate_sequence proxy)
just validate-completion experiments/<run>/results/completion.csv

# Re-score existing CSVs against gold + official eval_metrics.py
just rescore --results experiments/<run>/results \
  --gold extras/eval_local/gold.json --self-check
```

### Submission folder

Promoted runs land in `extras/results/<slug>/` with an index at `extras/results/INDEX.json`:

```bash
just promote kickoff-final <eval_run_id>
```

Each promoted folder includes: the three CSVs, `metrics_report.md` (if gold was used),
`proxy_report.json` (kickoff), `manifest.json`, and optionally `official_scores.txt`.

---

## 2. Tagging (training + eval)

Tags are **free-form strings** — use whatever helps you find runs later. No fixed schema.

### Training (YAML config)

```yaml
extra:
  tags: [leonardo, sft, full-ft, lofo:mosfet, lr-1e-5, attempt-2]
  hub_tags: [report-final, best-so-far]   # extra tags on Hugging Face only
  hub_notes: "2ep full-FT; held-out MOSFET family"
```

Tags from config are stored on the registry run (`experiments/<run_id>/meta.json`).
Cluster submits now merge config tags with `cluster`.

### Eval (`zo-track predict`)

```bash
--tags split:id,family:MOSFET,role:baseline,anything-you-want
--train-run <training_run_id>    # links eval → training run; adds train-run:… tag
--note "eval after W&B run xyz"  # free text in meta.json + manifest
```

Auto-tags added: `version:…`, `predictor:…`, `model-ref:…`, `eval-set:…`

### Look up runs

```bash
just runs                        # tags column
just runs -- --tag lofo-mosfet   # filter
just runs-tag report-final
uv run zo-runs show <run_id>     # full meta + training config
```

Dashboard `/compare` groups eval runs by tags like `split:`, `family:`, `predictor:`.

### Reporting API (normalized comparisons)

| Endpoint | Purpose |
|----------|---------|
| `GET /api/compare/report` | Normalized rows: model identity, all headline metrics, artifact paths, deltas vs baseline |
| `GET /api/compare/examples?run_a=&run_b=&task=&mode=` | Per-example side-by-side (requires `examples.jsonl`) |
| `GET /api/runs/{id}/examples?task=&outcome=` | Single-run example traces (correct/wrong filter) |
| `GET /api/runs/{id}/artifacts` | Artifact file paths for a run |

Each eval run can write `experiments/<run_id>/results/examples.jsonl` (enabled by default in `zo-track predict`) with predictions, gold, correctness, and optional LLM `reasoning` / `raw_response` traces.

Regenerate the static Infineon dashboard from promoted results:

```bash
node infineon-results-dashboard/scripts/build-results.mjs --finetuned kickoff-final --baseline ngram-baseline
```

---

## 3. Hugging Face artifacts (training params on the model repo)

Every HF upload writes two files **into the checkpoint folder** before push:

| File | Contents |
|------|----------|
| `training_manifest.json` | Run id, git SHA, SLURM job, tags, full training YAML, notes |
| `README.md` | HF model card (searchable YAML tags + hyperparameter table) |

**Where to look on Hugging Face:** open `https://huggingface.co/XCombinator/<model>` → **README**
or download **`training_manifest.json`** from the Files tab.

### Upload paths

1. **During training** — set in config:
   ```yaml
   extra:
     hub_model_id: XCombinator/sft-fab-lofo-mosfet
     push_to_hub: true
     hub_private: true
   ```
2. **After Leonardo job** — login node:
   ```bash
   bash scripts/leonardo_upload_artifact.sh <run_id> XCombinator/sft-fab-lofo-mosfet
   ```

### Backfill metadata for an existing run

```bash
just hub-manifest <run_id> --hub-model-id XCombinator/your-model --note "optional note"
# then upload (script above) if not already on HF
```

---

## 4. End-to-end checklist (final model)

1. **Train** with tags + `hub_model_id` in config → checkpoint on scratch / HF
2. **Eval locally** with `--gold` + `--self-check` → paste `metrics_report.md` into REPORT
3. **Eval kickoff** with `--eval-set kickoff --promote kickoff-final` → submission CSVs in `extras/results/`
4. **Link eval to train** with `--train-run` so manifests trace back to training params
5. **Compare** baselines via `eval-suite` or dashboard `/compare`

Organizers score kickoff CSVs with their held-out labels (Task 4 OOD is post-submission only).

---

## 5. Command reference

| Command | What it does |
|---------|----------------|
| `just local-eval FAMILY` | Build `extras/eval_local/` + `gold.json` |
| `just track "…"` | `zo-track predict` |
| `just kickoff-predict "…"` | Predict on organizer inputs |
| `just eval-suite <yaml>` | Tagged comparison matrix |
| `just rescore …` | Score CSVs without re-inference |
| `just promote <slug> <run_id>` | Copy to `extras/results/` |
| `just self-check …` | Run official `eval_metrics.py` locally |
| `just validate-completion …` | Grammar proxy on completions |
| `just runs [-- --tag X]` | List/filter registry runs |
| `just hub-manifest <run_id>` | Write HF README + manifest locally |

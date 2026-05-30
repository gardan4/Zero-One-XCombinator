# Training configs

YAML experiment configs (`zo_common.config.ExperimentConfig`). Run with
`just train <config>` (SFT) / `just grpo <config>` (GRPO); add `--dry-run` to validate the
config + the registry->backend->frontend path with **no torch** (writes a simulated loss curve).

## Stream 1 — SFT spine (fab step-sequences)

Full fine-tune of `Qwen/Qwen2.5-1.5B-Instruct` on the generated corpus. **`lora: false`** on
purpose: a full fine-tune saves a complete checkpoint, so `just serve <ckpt_dir>` works with
**no LoRA-merge step**. `text_field: text` trains on the `*_sft_lm.jsonl` rows
(`{"text": "Product family: <F>\nProcess sequence: <step | step | ...>", "family": <F>}`,
one full sequence per row).

| Config | Trains on | Role |
|---|---|---|
| `sft_fab.yaml` | MOSFET + IGBT + IC | all-family baseline (leaky-ID upper bound) |
| `sft_fab_lofo_mosfet.yaml` | IC + IGBT | OOD model, MOSFET held out |
| `sft_fab_lofo_igbt.yaml` | IC + MOSFET | OOD model, IGBT held out |
| `sft_fab_lofo_ic.yaml` | IGBT + MOSFET | OOD model, IC held out |

The LOFO mapping (which 2 families each fold trains on) mirrors `data/generated/splits.json`
-> `lofo.<FAMILY>.train_families`. Each LOFO config expresses its fold simply by joining the
two train families' `*_sft_lm.jsonl` paths with a comma — `load_sft_dataset` (in
`zo_train/data.py`) splits a comma-separated `dataset` (or a glob) into
`load_dataset("json", data_files=[...], split="train")`.

**Prereq:** the corpus must exist. Build it once with `uv run python -m zo_train.datagen`
(writes `data/generated/{MOSFET,IGBT,IC}_sft_lm.jsonl` + `splits.json`). `--dry-run` does
**not** need the corpus (it never loads the dataset); a real run does.

`extra.max_steps: 50` caps each config at a 1-GPU smoke. Delete it (or set `-1`) for a full run.

### ID-vs-OOD: the leaky-ID caveat

`sft_fab.yaml` trains on **all** families, including whatever family the organizers eval — so its
score on any one family is **leaky in-distribution (ID)**, an upper bound, not generalization. The
three `sft_fab_lofo_*` configs are the **honest OOD comparison**: each is tested on the family it
never saw. Report both as the headline ID->OOD table. Caveat for the writeup: LOFO over the 3 real
families is our only OOD proxy — the hidden 4th family (Task 4) can't be rehearsed locally.

## Where trained weights and logs live

| Artifact | Location |
|----------|----------|
| Published checkpoints | Hugging Face org **`XCombinator`** (`extra.hub_model_id` / `leonardo_upload_artifact.sh`) |
| Training metrics | W&B project **`XCombinator/XCombinator`** (`WANDB_ENTITY` / `WANDB_PROJECT`) |
| SLURM-local checkpoint | `experiments/<run_id>/artifacts/` before upload |

## Checkpoint -> submissions (the path)

Training is **not** the submission step. After upload, point eval at the HF repo id. The **shared
driver** (`zo-track`) produces the three submission CSVs + `metrics_report.md`:

```bash
# 1. serve the full-FT checkpoint (vLLM, OpenAI-compatible; no merge needed)
just serve experiments/<run_id>/artifacts

# 2. generate submissions; self-score with gold.json (organizers score CSVs with their script)
zo-track predict -p hf --model XCombinator/sft-fab-lofo-mosfet \
  --version sft-lofo-mosfet-v1 --model-ref XCombinator/sft-fab-lofo-mosfet \
  --valid   eval_input_valid.csv \
  --anomaly eval_input_anomaly.csv \
  --gold    gold.json \
  --tags    split:ood,family:MOSFET,eval-set:local   # LOFO held-out family

# Baseline on same inputs (restrict train families for honest OOD):
zo-track predict -p ngram --train-families IGBT,IC \
  --version ngram-baseline-v1 --tags split:ood,family:MOSFET,eval-set:local \
  --valid ... --gold ...
```

This writes `nextstep.csv` / `completion.csv` / `anomaly.csv` (to `extras/results/<run>/`) and
logs scored, tagged metrics to the registry -> dashboard. Tag the all-family baseline
`split:id` and each LOFO checkpoint `split:ood,family:<held-out>` so the dashboard can draw the
ID-vs-OOD bars per family. **Lead completion results with edit-distance / block-acc, not exact
match** (EM is ~0 for everyone — synonyms + optional steps mean many valid completions exist).

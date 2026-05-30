# XCombinator — Industrial AI (Infineon)

> **Status: draft / seeded.** Real numbers + the OOD table go in once the eval phase lands.
> Fill `{...}` placeholders and the `TODO:` markers; delete nothing structural.

---

## Team

- **{Name}** — training / cluster
- **{Name}** — eval / inference
- **{Name}** — frontend / dashboard
- **{Name}** — data / infra

**Track:** Industrial AI (Infineon) — learning & benchmarking process logic from fab step sequences.

---

## TL;DR

We fine-tuned a pretrained LLM (Qwen2.5-1.5B) on semiconductor fab process sequences and measure
whether it learns real **process logic** vs. memorizing step-order statistics — using a
leave-one-family-out (LOFO) setup as an in-distribution → out-of-distribution probe. We ship the 3
graded task outputs, an ID→OOD generalization study, and a live dashboard with an interactive
recipe-validation copilot. {TODO: one-line headline result.}

---

## Problem

The judged question: **does a model learn the *process logic* of semiconductor fabrication, or just
memorize step-order patterns?** We attack it three ways the organizers grade — next-step prediction,
sequence completion, anomaly detection — and probe generalization by training on 2 of 3 product
families and testing on the held-out one (a proxy for the hidden 4th family in Task 4).

---

## Approach

- **Substrate:** pretrained `Qwen/Qwen2.5-1.5B-Instruct`, **full fine-tune** (not LoRA) — serves
  directly, no merge step.
- **Data factory:** deterministic generator over the organizer grammar; `validate_sequence()` is a
  **free, perfect verifier** of the 10 process rules — we use it as oracle, RL reward, and the
  copilot engine.
- **OOD probe:** leave-one-family-out — a model trained on {2 families} is scored on the held-out
  family; the ID→OOD drop is the headline.
- **Two model flavors:** (a) pure-LM (next-token over valid routes), evaluated by perplexity /
  likelihood; (b) **instruction-tuned** for promptable per-task outputs + step reasoning. {TODO:
  finalize which we submit.}
- **Where it runs:** training + batch inference on **Leonardo (A100)**; dashboard + copilot local;
  experiment tracking in W&B (`XCombinator/XCombinator`).

---

## How to run it

See [`README.md`](../../README.md) and [`docs/STATUS.md`](../../docs/STATUS.md). In short:

```bash
just setup                                   # light deps
just dev                                     # dashboard at http://localhost:3000
# cluster (Leonardo):
uv run zo-cluster submit -c packages/training/configs/leonardo_sft_fab.yaml
uv run zo-cluster pull <run_id>              # bring metrics/checkpoint refs local
uv run zo-track predict -p hf --version <v> --model <ckpt> --valid ... --anomaly ... --gold ...
```

---

## Results

> Self-scored with `zo-track` (metrics per `generation_rules.md` §5); organizers score our submitted
> CSVs with their own tooling. Raw outputs in `extras/results/`.

**Training (done):** Qwen2.5-1.5B full-FT, bf16, 2 epochs, 1×A100. All 4 runs completed.

| Model | Trains on | Held-out (OOD) | Final train loss | Token-acc |
|---|---|---|---|---|
| all-families | MOSFET+IGBT+IC | — (submission model) | 0.119 | 0.97 |
| LOFO-MOSFET | IGBT+IC | MOSFET | 0.137 | 0.97 |
| LOFO-IGBT | MOSFET+IC | IGBT | 0.144 | 0.97 |
| LOFO-IC | MOSFET+IGBT | IC | 0.130 | 0.97 |

**Task metrics (ID vs OOD):** `TODO:` per-family + per-cut Top-1/3/5 + MRR (Task 1), EM / edit-dist /
block-acc (Task 2), P/R/F1/ROC-AUC + rule-attribution (Task 3); held-out **perplexity** and
**likelihood-AUC** as the continuous generalization signal.

**Baselines:** n-gram (memorization floor) + symbolic oracle (upper bound). `TODO:` table.

---

## What worked

- End-to-end **training on Leonardo** (prestage → A100 → checkpoint → live W&B via the compute proxy).
- The **symbolic verifier** as oracle + interactive copilot — instant, exact, GPU-free.
- {TODO: the ID→OOD result once measured.}

## What didn't work

- First generation-based eval **mismatched the pure-LM** model (instruction-style prompts → ~chance
  on the model's own family). Fix: loss/likelihood metrics + an instruction-tuned variant.
- {TODO: more.}

## What we'd do with another 36 hours

- Scaling study (100 / 1k / 5k sequences; 0.5B / 1.5B / 3B).
- GRPO with the `validate_sequence` reward for verifier-checked "reasoning".
- {TODO.}

---

## Track-specific deliverables (Industrial AI)

- [ ] `extras/results/{nextstep,completion,anomaly}.csv` (Task 1/2/3 formats)
- [x] Training artifacts: 4 checkpoints (Leonardo scratch), loss curves (W&B + dashboard)
- [ ] Self-scores (`metrics_report.md`) on all three tasks, per-family + per-cut
- [ ] Demo: baseline vs. trained on identical inputs

---

## Credits & dependencies

- **Libraries:** torch 2.7, transformers 4.57, trl 0.29, peft, datasets, vLLM; Next.js + recharts; FastAPI.
- **Pre-trained model:** `Qwen/Qwen2.5-1.5B-Instruct`.
- **Compute:** Leonardo (CINECA) A100. **Tracking:** Weights & Biases.
- **AI coding assistant:** Claude Code.

## A note on honesty

The submitted **anomaly** verdicts use our symbolic `validate_sequence` oracle (a verifier we built);
the *learned* detector's likelihood-AUC is reported separately as the science. {TODO: note any other
stubs.}

---

*Submitted by team XCombinator for Zero One Hack_01, 2026-05.*

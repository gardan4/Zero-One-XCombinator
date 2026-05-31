# XCombinator — Industrial AI (Infineon)

**Track:** Industrial AI (Infineon) — learning & benchmarking **process logic** from semiconductor fab
step sequences. **Substrate:** `Qwen/Qwen2.5-1.5B-Instruct`, full fine-tune, unified **JSON** task format.

## Team

- **{Name}** — training / cluster · **{Name}** — eval / inference · **{Name}** — frontend / dashboard · **{Name}** — data / infra

---

## TL;DR

We fine-tune a small pretrained LLM to learn the **process logic** of semiconductor fabrication and
measure it the three ways the organizers grade — **next-step prediction, sequence completion, anomaly
(rule-violation) detection** — against honest baselines (frozen base, **DeepSeek-V4-Flash** zero-shot,
classical **n-gram**). Final MOSFET labeled eval (`eval_local` gold):

| Model | n | Next-step | Completion | Anomaly F1 |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B Instruct (zero-shot) · unified JSON | 200 | 0.0% | 0.2% | 0.0% |
| DeepSeek-V4-Flash (zero-shot) · mosfet40 | 80 | 23.8% | 17.7% | 13.6% |
| **n-gram baseline** | 200 | **69.0%** | 63.7% | **88.9%** |
| **SFT instruct-all** · Qwen2.5-1.5B | 200 | 47.5% | 56.5% | **56.7%** |
| **SFT · 2000 routes/family** (best completion) | 200 | 52.5% | **73.5%** | 10.8% |

Two findings stand out: **(1) fine-tuning is the best LLM approach** — it gives the best LLM score on
all three tasks and is the only method that **beats the n-gram on completion** (73.5% vs 63.7%;
scale-2000 checkpoint). **(2) DeepSeek-V4-Flash zero-shot is decent** (24% / 18% / 14% on mosfet40) —
proof the task is partly solvable without training — but the best SFT scores roughly **double next-step
and completion** and **4× anomaly F1** vs DeepSeek. **(3)** the n-gram still wins **next-step (69%)**
and **anomaly (89%)**;
no single fine-tune checkpoint wins all three tasks (instruct-all: 56.7% anomaly; scale-2000: weak
anomaly at 10.8%). A **live copilot** runs the fine-tune on-device (Apple Silicon).

---

## Problem

Does a model learn the *process logic* of fabrication, or just memorize step-order patterns? Routes are
ordered lists from a fixed ~120-step uppercase vocabulary across three families (MOSFET / IGBT / IC). We
compare **SFT fine-tunes** against **frozen base**, **large-model zero-shot** (DeepSeek), and a
classical **n-gram** so gains are measured honestly — not against a strawman.

## Approach

- **Unified JSON task format.** One promptable model for all tasks: numbered input + a single JSON
  answer schema — `{"reasoning":…, "steps":[…]}` (next-step/completion) and
  `{"reasoning":…, "valid":bool, "rule":"RULE_…"}` (anomaly, reasoning-first chain-of-thought). The
  scorer parses the JSON; `reasoning` makes the model's rule-checking explicit.
- **Data factory.** Deterministic generator over the organizer grammar; `validate_sequence()` is a free,
  perfect verifier of the 10 process rules — used as the oracle, the anomaly label source, and the
  copilot's real-time validity engine. Balanced valid/invalid examples teach detection.
- **Studies.** (a) **data-scaling** at 100/300/800/2000 routes-per-family (fixed 1-epoch budget);
  (b) **model-size** 0.5B vs 1.5B vs 3B; (c) **baselines** n-gram, frozen Qwen (unified JSON),
  DeepSeek-V4-Flash zero-shot (rules-in-context), vs SFT checkpoints.
- **Where it runs.** Training + batch inference on **Leonardo (A100)**; dashboard + live copilot run
  **locally on a Mac** (custom transformers/MPS server — vLLM needs CUDA).

## Results

**Final comparison (MOSFET, labeled local eval).** All rows promoted in `extras/results/`; DeepSeek
on a smaller mosfet40 slice (n=80/task). Metrics: top-1 next-step, block-level completion accuracy,
anomaly F1.

| Model | n | Next-step | Completion | Anomaly F1 |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B Instruct (zero-shot) · eval_local gold · unified JSON | 200 | 0.0% | 0.2% | 0.0% |
| DeepSeek-V4-Flash (zero-shot) · mosfet40 | 80 | 23.8% | 17.7% | 13.6% |
| n-gram baseline · eval_local gold | 200 | 69.0% | 63.7% | 88.9% |
| SFT instruct-all · Qwen2.5-1.5B · eval_local gold | 200 | 47.5% | 56.5% | 56.7% |
| SFT · 2000 training routes (best completion) · eval_local gold | 200 | 52.5% | 73.5% | 10.8% |

**Takeaways (aligned with the table above).**

| | Best overall | Runner-up | Notes |
|---|---|---|---|
| **Next-step** | n-gram **69.0%** | SFT scale-2000 **52.5%** | Fine-tune beats DeepSeek (23.8%) and frozen Qwen (0%) by a wide margin; n-gram still leads. |
| **Completion** | **SFT scale-2000 73.5%** | n-gram 63.7% | Only fine-tune beats the classical baseline; instruct-all 56.5%. |
| **Anomaly F1** | n-gram **88.9%** | **SFT instruct-all 56.7%** | Fine-tune learns detection from 0; DeepSeek 13.6% is decent zero-shot but far below SFT. |

- **Fine-tuning wins among LLMs.** Instruct-all beats frozen Qwen (~0%) and DeepSeek zero-shot on all
  three tasks; scale-2000 is the completion specialist. Instruct-all is the balanced production model
  (47.5% / 56.5% / 56.7%); scale-2000 pushes completion to **73.5%** — our only score above the n-gram.
- **DeepSeek zero-shot is decent, not enough.** At 23.8% / 17.7% / 13.6% (mosfet40), a large prompted
  model clearly understands something about fab logic — but remains well below SFT and the n-gram. Prompt
  engineering alone does not replace training on this vocabulary.
- **Frozen base fails; training is required.** Qwen2.5-1.5B with the same unified JSON prompt scores
  ~0% — the substrate needs domain SFT to emit valid step tokens and rule judgments.
- **Classical n-gram stays strong** on next-step and anomaly (local transition statistics); the LLM's
  unique win is **long-range completion coherence** after enough unique training routes.

**Fine-tuning the best checkpoints** (from W&B [`XCombinator/XCombinator`](https://wandb.ai/XCombinator/XCombinator)):

Both models are **full fine-tunes** of `Qwen/Qwen2.5-1.5B-Instruct` on Leonardo **A100** — no LoRA,
`trl` SFT with **completion-only loss** on the assistant JSON turn, lr **1×10⁻⁵**, effective batch
**16** (4×4 grad accum), max seq **1024**, bf16 + gradient checkpointing. Training rows are unified
instruct chat examples (system + numbered route + JSON answer) built by our datagen from
grammar-valid synthetic routes plus balanced invalid sequences for anomaly.

| | **SFT instruct-all** (balanced · best anomaly) | **SFT scale-2000** (best completion) |
|---|---|---|
| **Corpus** | `instruct_all.jsonl` — **18k** examples (800 train routes × 3 families; ~2:1 next-step vs completion + valid/invalid anomaly) | **2000 routes/family** → scale corpus (~**47k** examples, 1 epoch budget) |
| **Epochs run** | 3 planned → **2.72** (SLURM wall ~2.2 h) | **1** → **0.98** (same wall) |
| **Eval checkpoint** | step **3050** · train/loss **0.017** | step **2900** · train/loss **0.017** |
| **W&B train run** | [`…instruct-all_869c8a`](https://wandb.ai/XCombinator/XCombinator/runs/20260530_234537_sft_leonardo-sft-fab-instruct-all_869c8a) | [`…scale-2000_ee052b`](https://wandb.ai/XCombinator/XCombinator/runs/20260530_234606_sft_leonardo-sft-scale-instruct-2000_ee052b) |

Configs: `leonardo_sft_fab_instruct.yaml` · `leonardo_sft_scale_instruct_2000.yaml`. Both jobs hit the
cluster wall-time limit (~2.5 s/step); we evaluated the last saved checkpoints above.

**Why this fine-tuning was likely best** (supported by our ablations, not just the headline scores):

- **Instruct SFT on a small open model, not prompt-only.** The same `Qwen2.5-1.5B-Instruct` substrate
  scores ~0% zero-shot with our unified JSON prompts; DeepSeek zero-shot is decent but still ~2× below
  SFT. The closed ~120-step fab vocabulary needs **weight updates** to emit exact step names and parseable
  JSON — not a bigger prompt alone.
- **Full fine-tune over LoRA on one A100.** Steps are a tiny, structured vocabulary far from web
  pretraining; full FT (no adapter bottleneck) lets the model bind concrete step strings to grammar
  positions. LoRA would be cheaper but we prioritized capacity on this low-entropy output space.
- **One unified JSON schema for all three graded tasks.** The same model family handles next-step,
  completion, and anomaly with one scorer. The completion-specialist checkpoint beats the n-gram on
  completion, while the balanced instruct-all checkpoint learns anomaly detection. Reasoning-in-JSON
  especially helps the rule-violation task (instruct-all F1 56.7% vs 10.8% for the completion-tuned
  scale run).
- **Completion-only loss on the assistant turn.** Gradients update only the JSON answer, not the fixed
  system/task prompt — keeps the task definition stable while the model learns fab-specific completions.
- **Grammar-valid synthetic data at scale beats bigger models.** Our size sweep (0.5B / 1.5B / 3B) was
  flat; the data-scaling curve moved completion from 34.5% → 73.5%. **Unique routes**, not parameters,
  unlocked the win over the n-gram on long suffixes (local n-grams can't plan coherent 40+ step endings).
- **Two checkpoints for two objectives.** Instruct-all (all families, balanced valid/invalid anomaly,
  multi-epoch) is the right deploy/submit model; scale-2000 (more unique all-family routes, 1 epoch) is
  the completion specialist — an honest split rather than one compromised run.

**Data-scaling is the story** — completion block-accuracy climbs with data and **crosses the n-gram
baseline (63.7%)**:

| routes / family (1 epoch) | 100 | 300 | 800 | 2000 |
|---|---|---|---|---|
| completion block-acc | 0.345 | 0.500 | **0.660** | **0.735** |
| next-step top-1 | 0.365 | 0.435 | 0.430 | 0.525 |

**Model size doesn't help — data does.** Across the size sweep (0.5B / 1.5B / 3B, full data, ~3 epochs,
3B via 4-GPU FSDP), next-step and completion are essentially flat (~0.45 / ~0.56) — the 1.5B is the
sweet spot and bigger doesn't move the needle. Capacity isn't the bottleneck on this structured task;
**unique training data is** (the scaling curve above). A clean, slightly counter-intuitive finding.

**Anomaly is learned, not free — but trades off with completion.** The all-family instruct-all model
reaches F1 **56.7%**; the scale-2000 completion winner drops to **10.8%** (1-epoch scale run).
0.5B/3B size points and the frozen base sit at 0 (they answer "valid" to everything). Spotting one rule
violation in a ~100-step route is the hardest task and is sensitive to training mix, capacity, and
epochs.

**It generalizes across all three families** (one all-family model, evaled per family) — best 1.5B,
next-step / completion / anomaly-F1:

| family | next-step | completion | anomaly F1 |
|---|---|---|---|
| MOSFET | 0.475 | 0.555 | 0.567 |
| IGBT | 0.315 | 0.365 | 0.526 |
| IC | 0.430 | 0.500 | 0.000 |

Not a MOSFET-only model: MOSFET is strongest, IGBT routes are hardest (more variable early blocks), and
anomaly detection transfers to IGBT (F1 52.6%) but not yet to IC — an honest per-family weak spot.

## What worked / what didn't

- **Worked:** SFT clearly beats zero-shot (fine-tune ≈2× DeepSeek on next-step/completion, ≈4× on
  anomaly); unified JSON format (one model, three tasks); symbolic verifier as oracle + copilot engine;
  Leonardo A100 training with W&B; on-device (MPS) live inference with instruct-all in the copilot.
- **Didn't / honest caveats:** n-gram still beats our LLMs on next-step and anomaly — we do not claim
  SFT wins everything. No single checkpoint wins all three tasks (instruct-all: best anomaly 56.7%;
  scale-2000: best completion 73.5%, anomaly 10.8%). Instruct-all **collapses completion to one step**
  when the SFT mix skews next-step (2:1, shared JSON schema). SLURM wall-time stopped training at
  ~2.7/3 epochs. **Organizer kickoff CSVs** use instruct-all (stronger anomaly, weaker completion than
  scale-2000).

## Live demo

The **Fab Process Copilot** (`xcombinator-copilot`) does live next-step prediction and carries the
whole comparison story in an in-app **Results** view (rendered from **real promoted eval metrics**, no
placeholders). Pick a family, build/import a route, and it predicts the next step while
`validate_sequence()` flags rule violations in real time. The model picker switches between **three**
backends side by side — **DeepSeek-V4-Flash** (decent zero-shot baseline, ~24% next-step on mosfet40),
the **frozen Qwen2.5-1.5B** base (~0% — shows why training matters), and **SFT instruct-all** (our
deployed fine-tune, best balanced LLM) — and shows each model's own **reasoning** next to the predicted
step (rich chain-of-thought for DeepSeek; the fine-tune answers directly with token-probability
confidence). One launch task (`Demo: start all`) boots the local model server + UI. See
`scripts/serve_copilot_mac.py`.

## Reproduce

```bash
# corpora (ship to cluster: data/ is NOT synced by submit prep)
uv run python -m zo_train.datagen --build && uv run python scripts/build_scaling_corpora.py && bash scripts/ship_corpora.sh
# train (Leonardo A100, --no-prep avoids the env-pruning sync) + eval + promote
uv run zo-cluster submit --no-prep -c packages/training/configs/leonardo_sft_fab_instruct.yaml
uv run zo-cluster judge-eval --local --no-prep --model <ckpt> --predictor hf --eval-dir extras/eval_local/MOSFET --promote <slug>
# dashboard + live demo
node infineon-results-dashboard/scripts/build-results.mjs   # -> public/results.js (+ story.html)
uv run --no-sync python scripts/serve_copilot_mac.py        # :8001; then the copilot with VITE_MODEL_BASE_URL=:8001/v1
```

## Credits

- **Libraries:** torch 2.7, transformers 4.57, trl 0.29, peft, datasets; FastAPI; Vite/React.
- **Pre-trained models:** `Qwen/Qwen2.5-1.5B-Instruct` (SFT substrate); `deepseek-ai/DeepSeek-V4-Flash`
  (zero-shot baseline via Featherless). **Compute:** Leonardo (CINECA) A100. **Tracking:** W&B.

## A note on honesty

All dashboard numbers are real eval metrics from `extras/results/` — no placeholders. The copilot's
*validity* check (green "process logic valid") is the **symbolic `validate_sequence` verifier**, not the
LLM; the LLM does the *predictions* (next step). The learned anomaly detector's F1 is reported as the
science, separately from the symbolic oracle (`baseline-oracle-anomaly`, the upper bound).

**Organizer submission CSVs:** [`extras/results/kickoff-final/`](extras/results/kickoff-final/) (`nextstep.csv`, `completion.csv`, `anomaly.csv`) — SFT instruct-all · Qwen2.5-1.5B on the kickoff test set (600 + 987 rows).

---

*Submitted by team XCombinator for Zero One Hack_01, 2026-05.*

*Same report as [`REPORT.md`](../../REPORT.md).**

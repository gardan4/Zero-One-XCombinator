# XCombinator — Industrial AI (Infineon)

**Track:** Industrial AI (Infineon) — learning & benchmarking **process logic** from semiconductor fab
step sequences. **Substrate:** `Qwen/Qwen2.5-1.5B-Instruct`, full fine-tune, unified **JSON** task format.

## Team

- **{Name}** — training / cluster · **{Name}** — eval / inference · **{Name}** — frontend / dashboard · **{Name}** — data / infra

---

## TL;DR

We fine-tune a small pretrained LLM to learn the **process logic** of semiconductor fabrication and
measure it the three ways the organizers grade — **next-step prediction, sequence completion, anomaly
(rule-violation) detection** — against two honest baselines: a classical **n-gram** and the **frozen
base model**. Headline on the MOSFET labeled eval (n=200):

| task | frozen base | **n-gram baseline** | **fine-tuned (best of our models)** |
|---|---|---|---|
| next-step (top-1) | ~0.00 | **0.69** | 0.525 |
| sequence completion (block-acc) | ~0.00 | 0.637 | **0.745 ✅ beats baseline** |
| anomaly (F1) | 0.00 | **0.89** | 0.567 (learned from 0) |

Two findings stand out: **(1)** with enough data the LLM **beats the strong n-gram on completion**
(0.745 vs 0.637), and **(2)** fine-tuning **teaches rule-violation detection from scratch** (anomaly F1
**0 → 0.57**; the frozen base scores 0) — evidence the model learns *logic*, not just step statistics.
A **live copilot runs the model on-device (Apple Silicon)** and predicts next steps in real time.

---

## Problem

Does a model learn the *process logic* of fabrication, or just memorize step-order patterns? Routes are
ordered lists from a fixed ~120-step uppercase vocabulary across three families (MOSFET / IGBT / IC). We
attack the three graded tasks and benchmark against a strong classical baseline so the LLM's value is
measured honestly — not against a strawman.

## Approach

- **Unified JSON task format.** One promptable model for all tasks: numbered input + a single JSON
  answer schema — `{"reasoning":…, "steps":[…]}` (next-step/completion) and
  `{"reasoning":…, "valid":bool, "rule":"RULE_…"}` (anomaly, reasoning-first chain-of-thought). The
  scorer parses the JSON; `reasoning` makes the model's rule-checking explicit.
- **Data factory.** Deterministic generator over the organizer grammar; `validate_sequence()` is a free,
  perfect verifier of the 10 process rules — used as the oracle, the anomaly label source, and the
  copilot's real-time validity engine. Balanced valid/invalid examples teach detection.
- **Studies.** (a) **data-scaling** at 100/300/800/2000 routes-per-family (fixed 1-epoch budget);
  (b) **model-size** 0.5B vs 1.5B; (c) **baselines** n-gram (likelihood) + zero-shot frozen base.
- **Where it runs.** Training + batch inference on **Leonardo (A100)**; dashboard + live copilot run
  **locally on a Mac** (custom transformers/MPS server — vLLM needs CUDA).

## Results

**Data-scaling is the story** — completion block-accuracy climbs with data and **crosses the n-gram
baseline (0.637)**:

| routes / family (1 epoch) | 100 | 300 | 800 | 2000 |
|---|---|---|---|---|
| completion block-acc | 0.345 | 0.500 | **0.660** | **0.745** |
| next-step top-1 | 0.365 | 0.435 | 0.430 | 0.525 |

**Model size doesn't help — data does.** Across the size sweep (0.5B / 1.5B / 3B, full data, ~3 epochs,
3B via 4-GPU FSDP), next-step and completion are essentially flat (~0.45 / ~0.56) — the 1.5B is the
sweet spot and bigger doesn't move the needle. Capacity isn't the bottleneck on this structured task;
**unique training data is** (the scaling curve above). A clean, slightly counter-intuitive finding.

**Anomaly is learned, not free.** Only the 1.5B model reaches F1 0.567; the 0.5B/3B size points, all
1-epoch scaling models, and the frozen base sit at 0 (they answer "valid" to everything). Spotting one
rule violation in a ~100-step route is the hardest task and is sensitive to training dynamics — it needs
balanced data *and* the right capacity/epochs.

**It generalizes across all three families** (one all-family model, evaled per family) — best 1.5B,
next-step / completion / anomaly-F1:

| family | next-step | completion | anomaly F1 |
|---|---|---|---|
| MOSFET | 0.475 | 0.555 | 0.567 |
| IGBT | 0.315 | 0.365 | 0.526 |
| IC | 0.430 | 0.500 | 0.000 |

Not a MOSFET-only model: MOSFET is strongest, IGBT routes are hardest (more variable early blocks), and
anomaly detection transfers to IGBT (F1 0.53) but not yet to IC — an honest per-family weak spot.

**The n-gram is a genuinely strong baseline.** On next-step, local transition statistics are hard to
beat on these structured routes (0.69); on anomaly, an n-gram likelihood threshold is excellent (F1
0.89). The LLM's edge is **completion coherence**, being **one promptable model across all three
tasks**, and **explicit reasoning** — not raw next-step accuracy.

## What worked / what didn't

- **Worked:** the unified JSON format (one model, three tasks, parseable + reasoned); the symbolic
  verifier as oracle + live copilot engine; end-to-end training on Leonardo A100 with W&B via the
  compute proxy; on-device (MPS) live inference.
- **Didn't / honest caveats:** the single canonical full model **collapses completion to one step**
  (next-step examples outnumber completion 2:1 and share the answer schema) — so it trails on
  completion while the *data-scaled* models, trained on more unique routes, complete full suffixes and
  win. The headline reports the best each task achieves across our models. Training jobs hit the SLURM
  wall-time at the observed ~2.5 s/step; we evaluated the last saved checkpoint (~2.7 of 3 epochs).

## Live demo

The **Fab Process Copilot** (`xcombinator-copilot`) does live next-step prediction and carries the
whole comparison story in an in-app **Results** view (rendered from **real promoted eval metrics**, no
placeholders). Pick a family, build/import a route, and it predicts the next step while
`validate_sequence()` flags rule violations in real time. The model picker switches between **three**
backends side by side — **DeepSeek-V4-Flash** (hosted via Featherless, the zero-shot model we
evaluated), the **frozen Qwen2.5-1.5B** base, and **our best fine-tune** (both local on Apple Silicon /
MPS) — and shows each model's own **reasoning** next to the predicted step (rich chain-of-thought for
DeepSeek; the fine-tune answers directly with a real token-probability confidence). One launch task
(`Demo: start all`) boots the local model server + UI. See `scripts/serve_copilot_mac.py`.

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
- **Pre-trained model:** `Qwen/Qwen2.5-1.5B-Instruct`. **Compute:** Leonardo (CINECA) A100. **Tracking:** W&B.

## A note on honesty

All dashboard numbers are real eval metrics from `extras/results/` — no placeholders. The copilot's
*validity* check (green "process logic valid") is the **symbolic `validate_sequence` verifier**, not the
LLM; the LLM does the *predictions* (next step). The learned anomaly detector's F1 is reported as the
science, separately from the symbolic oracle (`baseline-oracle-anomaly`, the upper bound).

**Organizer submission CSVs:** [`extras/results/kickoff-final/`](extras/results/kickoff-final/) (`nextstep.csv`, `completion.csv`, `anomaly.csv`) — SFT instruct-all · Qwen2.5-1.5B on the kickoff test set (600 + 987 rows).

---

*Submitted by team XCombinator for Zero One Hack_01, 2026-05.*

*Same report as [`submissions/XCombinator/REPORT.md`](submissions/XCombinator/REPORT.md).*

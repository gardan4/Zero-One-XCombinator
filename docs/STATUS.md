# Project Status — Industrial AI (Infineon) track

**TL;DR:** We trained real models on Leonardo. A Qwen2.5‑1.5B was full‑fine‑tuned on fab process
sequences (all‑families + 3 leave‑one‑family‑out models) — all runs completed, loss 1.6 → 0.05,
token‑accuracy 0.97, checkpoints saved to scratch, metrics live in W&B. A dashboard (incl. an
interactive recipe‑validation copilot) is running locally. The **evaluation** is the main thing
still in flight: our first generation‑based eval mismatched the model type, so we're pivoting to
loss/likelihood metrics (cheaper + correct). Then: produce the 3 submission CSVs + the ID→OOD
generalization result.

---

## 1. What we did

**Data & task understanding**
- Vendored the organizer data + grammar (`data/industrial-infineon/`); built a **data factory**
  (`zo_train/datagen.py`) that generates LM / next‑step / completion / anomaly / CoT examples and
  leave‑one‑family‑out (LOFO) splits. `validate_sequence()` is a **free, perfect verifier** of the
  10 process‑logic rules — we use it everywhere (oracle, rewards, the copilot).
- Generated the training corpus: 3 families × 800 LM sequences (+ negatives, eval sets).

**Inference / eval core (Stream 0)** — `packages/eval/zo_eval/`
- A model‑agnostic `Predictor` interface + output normalizer, baselines (n‑gram / oracle / freq),
  served‑LLM + HF predictors, and a driver (`zo-track`) that writes the **exact organizer
  submission CSVs** and scores them. Submission format verified **byte‑for‑byte** vs the spec.

**Training (Stream 1) — REAL GPU runs on Leonardo ✅**
- `Qwen/Qwen2.5-1.5B-Instruct`, **full fine‑tune**, bf16 + gradient checkpointing, lr 1e‑5, 2 epochs,
  1× A100‑64GB. Four checkpoints, all `COMPLETED`:
  | model | trains on | held‑out (OOD) | final loss | token‑acc |
  |---|---|---|---|---|
  | `leonardo-sft-fab-all` | MOSFET+IGBT+IC | — (submission model) | 0.119 | 0.97 |
  | `…lofo-mosfet` | IGBT+IC | MOSFET | 0.137 | 0.97 |
  | `…lofo-igbt` | MOSFET+IC | IGBT | 0.144 | 0.97 |
  | `…lofo-ic` | MOSFET+IGBT | IC | 0.130 | 0.97 |
- Checkpoints live on `$SCRATCH/zo-experiments/<run_id>/artifacts/` (~5.8 GB each, full model).

**Leonardo pipeline** — works end‑to‑end from a laptop (no SSH key needed):
- Auth via PuTTY `plink -pw` (password from `.env`); prestage on a login node (`uv sync --extra gpu`
  + model download to scratch); offline `uv sync` on compute; live W&B via the compute proxy.
- Plus the merged **fix branch** (`fix/known-issues`): preflight config validation, GRPO `kind`
  threading, intermediate checkpoint saving, `judge-eval` run‑id threading, backend `.env` loading,
  and a new `zo-cluster pull <run_id>` to bring cluster results local. (10 issues + tests.)

**Dashboard (Stream 4 + new)** — `apps/frontend` (Next.js) + `apps/backend` (FastAPI),
`uv run python scripts/dev.py` (or optional `just dev`):
- **Runs** list + **Run detail** (loss / token‑accuracy curves, config, metrics).
- **Compare**: run‑comparison metric matrix (ID vs OOD across tasks), anomaly ID‑vs‑OOD bars,
  baseline‑vs‑trained, confusion matrix, loss overlay.
- **Copilot** (interactive demo): paste a fab recipe → live validation against all 10 rules,
  violation explanation + suggested repair, the rule reference. GPU‑free (symbolic verifier).

**Experiment tracking**: W&B wired (entity `XCombinator`). The 4 training runs logged to
`XCombinator/zero-one-philyr`; project for new runs switched to `XCombinator/XCombinator`.

---

## 2. What worked

- ✅ The whole **train‑on‑Leonardo** loop: prestage → submit → A100 → trl SFT → checkpoint→scratch →
  metrics→registry + **live W&B online via the compute‑node proxy**.
- ✅ All 4 models converged cleanly and identically (loss 1.6→0.05, acc 0.97) in ~7–10 min each.
- ✅ W&B API key + proxy validated on the cluster.
- ✅ Dashboard + the symbolic copilot demo (the verifier is exact and instant).
- ✅ Submission‑CSV format matches the grader spec byte‑for‑byte.

## 3. What didn't work / gotchas

- ⚠️ **First model eval gave nonsense numbers** (trained model at ~chance on its *own* family:
  `top1≈0.11`, `anomaly_auc≈0.5`). Root cause: our model is a **pure language model** (trained only
  on next‑token over valid routes), but the eval harness prompted it **instruction‑style** and
  parsed generated text — a task/model mismatch, not a bad model. It was also **slow** (~20 min:
  unbatched generation over 600 items). → we're switching the metrics (see §4).
- ⚠️ **`$HOME` path bug**: cluster paths using `$HOME` got expanded against the *local* machine →
  fixed by setting absolute Leonardo paths in `.env`.
- ⚠️ Login‑node 10‑min CPU limit — prestage stayed under it (mostly I/O), but watch it.
- ⚠️ The 4 W&B runs are in `…/zero-one-philyr`, not `…/XCombinator` (they initialized before the
  project switch) — drag them over in the W&B UI to consolidate.
- ⚠️ Committed `main` had a 1‑line backend import bug (now fixed locally, **not yet pushed**).
- ℹ️ We chose **full fine‑tune, not LoRA** (serving simplicity); LoRA remains a fallback.

## 4. What we still need to do (priority order)

1. **Eval, done right** (the headline science). For our pure‑LM model use loss/likelihood metrics —
   cheap (a forward pass, not 20 min) and correct:
   - **Held‑out perplexity / NLL** per family, **ID vs OOD** — the cleanest generalization signal.
   - **Teacher‑forced next‑step accuracy** (argmax from logits) — Tasks 1.
   - **Likelihood‑based anomaly AUC** (valid routes score higher than corrupted) — the ID→OOD
     collapse is the headline.
   - Run for all 4 checkpoints × 3 families.
2. **Submission CSVs** (gradeable, `extras/results/`): T1/T2 via **LM continuation** (continue the
   partial route), T3 via the **symbolic oracle** for `IS_VALID`/`PREDICTED_RULE`.
3. **Wire results into the dashboard** compare page (ID‑vs‑OOD bars per task, perplexity gap).
4. **Decide:** keep pure‑LM + likelihood story, or add an **instruction‑tuned SFT** so the
   generation‑style tasks (and the copilot's model‑repair) also work.
5. **Coordination/repo:** land `fix/known-issues` on `main` and push (incl. the backend import fix).
6. **Report + 2–3 slides** (template in `docs/submission/`).
7. *Optional spikes:* GRPO with the `validate_sequence` reward ("thinking"); agentic repair copilot
   wired to the model; data/model **scaling** comparison.

---

## How to see it / key locations

- **Dashboard:** `uv run python scripts/dev.py` → http://localhost:3000 (backend on :8000).
  Optional shortcut: `just dev`. Try `/copilot` and `/inference`.
- **W&B:** https://wandb.ai/XCombinator/zero-one-philyr (the 4 training runs).
- **Checkpoints (cluster):** `$SCRATCH/zo-experiments/<run_id>/artifacts/` on Leonardo.
- **Configs:** `packages/training/configs/leonardo_sft_fab*.yaml`.
- **Submit a run:** `uv run zo-cluster submit -c <config>` · **pull results:** `uv run zo-cluster pull <run_id>`.
- **Spec we grade against:** `data/industrial-infineon/training_data/generation_rules.md` §5.

## Git state (as of this writing)

- `origin/main` = `7dc0a23` (Copilot work; has the 1‑line import bug).
- `origin/fix/known-issues` = `8d5b7dd` (the 10 fixes + preflight + tests).
- Local `main` = `d00adab` = fixes **merged** + import bug **fixed**, **not pushed** (your call).

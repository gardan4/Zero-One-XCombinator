# Track — Industrial AI (Infineon): Learning & Benchmarking Process Logic

**This is the track we committed to.** Mentor on-site: **Simeon Harrison** (AI:AT). Track partner:
**Infineon**. Difficulty: Advanced→Expert; achievable in 36 h with synthetic data.

> **The question:** can a model learn the *process logic* of semiconductor fabrication — or does it
> just memorize step-order statistics? Real chip fab is a 100+ step sequence where **order is
> everything**. We train sequence models on fab step-sequences and benchmark next-step prediction,
> sequence completion, and anomaly detection — including generalization to a held-out product family.

This is **sequence modeling over a small vocabulary**, not chat-LLM eval. Each fab step string is one
token (~120-token vocab). That's a big divergence from the generic chat/vLLM scaffold — see
"How our scaffold maps" below.

## Where the data + spec live
**Vendored into our repo** (from MIT-licensed `github.com/Lumos-Data/zero_one_hack_01`, folders
`tracks/industrial-infineon/` + `submission/`):
- **`data/industrial-infineon/`** — the whole `training_data/` folder: the 3 `*_variants.csv`
  (1,000 seqs each, ~21 MB total), `synthetic_*.csv` references, `*_Longdescr.csv` +
  `*_longdescription_parameters.csv`, **`generation_rules.md`** (authoritative — read first),
  **`generate_sequences.py`** (generator + validator, stdlib only), and the data `README.md`.
- **`docs/submission/`** — `SUBMISSION.md` + `REPORT_TEMPLATE.md` (what/how to hand in).
- **`docs/Track One Assignment.txt`** — the EN briefing (byte-identical to the repo's
  `Track_industrial_en.md`, so not re-vendored).

Sanity-checked on load: MOSFET/IGBT/IC = 1,000 seqs each, lengths ~125/148/115, all start
`RECEIVE WAFER LOT` / end `SHIP LOT`, all pass `validate_sequence`. To refresh from upstream:
sparse-clone the repo and copy those two folders (see INDEX "Sources").

> **Not in the public repo (expect at kickoff):** **`eval_metrics.py`** and **`judging/rubrics.md`**
> are referenced by the READMEs/SUBMISSION but are **absent upstream** — the organizers distribute
> them (with the eval input CSVs) at the start. Don't assume we have the scorer until then; build to
> the documented submission formats (below) and wrap the scorer when it lands.

## The data (3 product families)
Long format, **one step per row**: `SEQUENCE_ID,STEP` (e.g. `MOSFET_0001,RECEIVE WAFER LOT`).
Every sequence starts with `RECEIVE WAFER LOT` and ends with `SHIP LOT`.

| Family | Ref steps | Pre-generated | ~Rows | Valid-sequence space |
|---|---|---|---|---|
| MOSFET | 126 | 1,000 | ~125k | ~51 billion |
| IGBT | 151 | 1,000 | ~148k | ~13 trillion |
| IC | 107 | 1,000 | ~115k | ~6 billion |

(The `docs/Track One Assignment.txt` brief says IGBT "~13 billion"; the repo README+rules say
"~13 trillion" in two places — trust the repo.) **3,000 validated sequences total** ship in
`training_data/`. Per family there are also: a single canonical reference (`synthetic_*.csv`), a
`*_Longdescr.csv` (`STEP` + text `DESCRIPTION`), and a `*_longdescription_parameters.csv` (+ a
`REALISTIC FAB-LEVEL PARAMETERS` column). The descriptions/parameters are optional extra signal.

Load with **stdlib only** (no deps):
```python
from generate_sequences import read_csv_sequences
seqs = read_csv_sequences(Path("MOSFET_variants.csv"))  # -> dict[seq_id, list[step_str]]
```

## Generating more data — `generate_sequences.py`
The combinatorial space is huge, so generate as much as you want (data-scaling is a stretch goal).
```bash
python generate_sequences.py --family mosfet --count 2000 --output extra_mosfet.csv --seed 42
python generate_sequences.py --validate extra_mosfet.csv        # check vs all 10 rules
python generate_sequences.py --family igbt --estimate-only      # combinatoric estimate, no gen
```
Flags: `--family {mosfet,igbt,ic}`, `--count` (default 500), `--output` (default `<FAMILY>_variants.csv`),
`--seed` (default 42), `--validate <csv>`, `--estimate-only`. Public API:
`read_csv_sequences(path) -> dict[id, list[str]]` and `validate_sequence(steps) -> list[Violation]`
(`Violation(rule, description, step_index, step_name)`; empty list = valid).

## The grammar (why sequences look the way they do)
Shared backbone, with family-specific blocks/cycle-counts:
> Logistics(PREFIX) → Initial measurements → Pre-process clean → **Family prep** → First oxidation →
> **Process cycles {3..6}** (litho→etch→strip→clean→implant→anneal) → ILD → Via → Metal →
> Passivation → Backside → Final inspection → Test suite → SUFFIX(ship)

Family prep differs: MOSFET = epitaxy block; IGBT = dual epitaxial-wafer check + 6 litho levels;
IC = early backside grind. A litho block is a fixed sub-sequence: SPIN COAT → SOFT BAKE → ALIGN
MASK L*N* → EXPOSE L*N* → [POST EXPOSE BAKE] → DEVELOP → pattern-inspect → [HARD BAKE]. **11 variation
axes** (cycle count 3–6, optional bakes, intermediate cleans, extra measurements, 2nd metal layer,
step-name synonyms like `STRIP PHOTORESIST`≈`STRIP RESIST`) keep sequences valid while differing.

## The 10 forbidden patterns (what "process logic" means)
Used to build the held-out anomaly set. A sequence breaking any of these is invalid even if every
step is in-vocab. Implemented in `validate_sequence()`:
1. **RULE_DEP_NO_CLEAN** — a deposition must be preceded by a clean within ~12 steps.
2. **RULE_METAL_ETCH_NO_LITHO** — metal etch needs EXPOSE+DEVELOP within ~15 steps before it.
3. **RULE_ETCH_NO_MASK** — any (patterned) etch needs a DEVELOP within ~12 steps (spacer etch exempt).
4. **RULE_LITHO_LEVEL_SKIP** — mask levels must be sequential (L*N+1* not before L*N* completes).
5. **RULE_IMPLANT_NO_MASK** — implant needs a prior oxide-etch / develop within ~15 steps.
6. **RULE_CMP_NO_DEP** — CMP must follow a deposition/fill within ~6 steps.
7. **RULE_PAD_OPEN_BEFORE_DEP** — pad-window open must come after DEPOSIT+CURE PASSIVATION.
8. **RULE_TEST_BEFORE_PASSIVATION** — electrical tests must come after CURE PASSIVATION.
9. **RULE_SHIP_BEFORE_TEST** — SHIP LOT must come after WAFER SORT TEST.
10. **RULE_BACKSIDE_BEFORE_PASSIVATION** — DEPOSIT BACKSIDE METAL must come after CURE PASSIVATION.

## Evaluation — 3 submitted tasks + 1 organizer-only
Organizers distribute two **fixed eval input files** at kickoff (NOT in the public repo):
- `eval_input_valid.csv` — 600 partial sequences (`EXAMPLE_ID, FAMILY, COMPLETION_FRACTION,
  PARTIAL_SEQUENCE`; pipe-separated steps; cut at 60% or 80%). Feeds Tasks 1 & 2.
- `eval_input_anomaly.csv` — 987 unlabeled full sequences (`EXAMPLE_ID, FAMILY, SEQUENCE`); ~387
  with injected rule violations + 600 valid, shuffled. Feeds Task 3.

| # | Task | Metrics | Submit file |
|---|---|---|---|
| 1 | Next-step prediction | Top-1/3/5 Acc, MRR | `nextstep.csv`: `EXAMPLE_ID,RANK_1..RANK_5` |
| 2 | Sequence completion | Exact Match, Norm. Edit Dist, Token Acc, Block Acc | `completion.csv`: `EXAMPLE_ID,PREDICTED_SEQUENCE` (steps **after** cut, pipe-sep) |
| 3 | Anomaly detection | Bin Acc, Precision, Recall, F1, Confusion, ROC-AUC, Rule-Attribution Acc | `anomaly.csv`: `EXAMPLE_ID,IS_VALID,SCORE,PREDICTED_RULE` |

Task 3 columns: `IS_VALID` 1/0 (required), `SCORE`∈[0,1] valid-prob (optional, for AUC),
`PREDICTED_RULE` rule-id if invalid (optional, for attribution).
**Task 4 = OOD generalization**: organizers apply our submitted model to a **hidden 4th product
family** after submission and report the ID→OOD performance drop. We don't submit anything for it,
but our model must be runnable on an unseen family.

**Self-scoring:** `eval_metrics.py` (organizer-provided, **not yet in the repo — arrives at
kickoff**, stdlib only) gives per-task reports with family / truncation breakdowns:
```
python eval_metrics.py --task anomaly --ground-truth <gt.csv> --predictions <your_output.csv>
```
Until it lands, validate our own generated/predicted sequences with
`data/industrial-infineon/generate_sequences.py --validate <csv>` (uses the same 10 rules).

## Track-specific repo deliverables (from SUBMISSION.md + REPORT_TEMPLATE.md)
Beyond the general submission (see hackathon.md), the **Industrial AI** repo must contain:
- The three eval outputs **in `extras/results/`**: `nextstep.csv`, `completion.csv`, `anomaly.csv`
  (the REPORT template's checklist points there; raw scorer output also belongs in `extras/results/`).
- Training artifacts: checkpoint(s), training logs, **loss curves**.
- `eval_metrics.py` scores on all three tasks, **with per-family breakdown** (paste into REPORT Results).
- Demo: **baseline vs. trained output on identical inputs** (the side-by-side is what they want).
- (Optional) an architecture sketch in `extras/`.
- 🎁 **Bonus (rewarded, not required):** a small dashboard — loss curves, metric comparisons across
  families, baseline-vs-trained side-by-side, anomaly confusion matrix, scaling plots, before/after
  examples. Streamlit/Gradio/React all count. **We already have a Next.js dashboard → lean into it.**

## How our scaffold maps (and where it diverges)
The repo was scaffolded for **chat-LLM SFT/GRPO + OpenAI-style eval**. This track is **small-vocab
sequence modeling**. Concretely:
- **Model:** a small token-level Transformer (or even n-gram/LSTM baseline) trained **from scratch**
  on ~120-token vocab is very feasible and fast — you don't need a 1.5B chat model. Family can be a
  conditioning token. (Open base models allowed too, but from-scratch is the clean story here.)
- **Tokenizer:** identity over step strings (vocab = the ~120 step names + BOS/EOS + family tokens).
  Build a tiny `step↔id` map from `generation_rules.md` §1 / by scanning the variant CSVs.
- **Data loader:** parse the long-format `*_variants.csv` (or `read_csv_sequences`) into per-sequence
  token-id lists. This is new code (suggest `packages/training/zo_train/data/process_seq.py`).
- **Eval:** wrap `eval_metrics.py` as zo-eval tasks (next-step / completion / anomaly) instead of the
  OpenAI-`exact_match` harness. Anomaly detection can reuse `validate_sequence()` as a *rule-based*
  baseline to beat — a great baseline-vs-model story.
- **Registry/dashboard/worktrees/cluster flow all still apply unchanged.** `--dry-run` + the
  registry→backend→frontend path are exactly what we want for the before/after demo and scaling plots.

## Strategy (fastest path to a strong submission)
1. **Level 1:** load data → frequency / rule-based baselines for next-step + anomaly (`validate_sequence`).
2. **Level 2:** train a small from-scratch sequence model; show baseline → trained → tuned deltas on
   the three tasks with `eval_metrics.py`; per-family breakdown; loss curves on the dashboard.
3. **Level 3 / stretch:** scaling study — models trained on 100 vs 1k vs 5k+ generated sequences,
   and/or model-size sweep; plot accuracy/F1 vs data/compute. Care about **OOD** (Task 4): don't
   overfit family-specific quirks; family-conditioning + diverse generation help generalization.

## Append below as you learn
- (chosen model arch + hyperparams that worked, baseline numbers, eval_metrics.py exact CLI quirks: TBD)

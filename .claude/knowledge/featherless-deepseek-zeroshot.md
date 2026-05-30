# Featherless + DeepSeek-V4-Flash — zero-shot rules-in-context findings

Notes from May 2026 experiments: zero-shot baseline via `RulesContextLLMPredictor` on
[Featherless.ai](https://featherless.ai) (`deepseek-ai/DeepSeek-V4-Flash`), local proxy eval sets
under `extras/eval_local_mosfet5` (5 sequences → 10 next-step + 10 anomaly).

**CLI:** `uv run zo-track featherless-eval -V <version> --eval-dir extras/eval_local_mosfet5 --tasks nextstep,anomaly --concurrency 1 --no-wandb`

**Env:** `FEATHERLESS_API_KEY`, `FEATHERLESS_MODEL=deepseek-ai/DeepSeek-V4-Flash` (see `.env.example`).

---

## Summary

| What works | What still breaks |
|---|---|
| JSON structured output (reasoning + `steps` / `valid` + `rule`) | Anomaly recall when reasoning is long (256-token cap truncates verdict) |
| Featherless HTTP repair (duplicate `content` keys) | Featherless rate limits at concurrency ≥4 |
| Numbered step input → better next-step top1/MRR | Model still skips micro-steps after via clean without prompt digest changes |
| Reason-first + “RULE_* = violation pattern” → fixes pad-rule false positives on **valid** seqs | Same prompt → 0/5 forbidden caught (truncation + parser fallback) |

Best smoke5 next-step so far: **40% top1**, **0.52 MRR** (numbered input + reason-v1/v2 prompts).

---

## Infrastructure

### Featherless integration

- Code: `packages/eval/zo_eval/featherless.py`, CLI `zo-track featherless-eval`
- Uses same `run_track` path as Leonardo judge eval; `RulesContextLLMPredictor` with `backend="served"`
- **Concurrency:** use **`--concurrency 1`** (or 2 max). At 4 workers: heavy **429** rate limits; at 4 with broken JSON repair, most requests failed entirely.

### HTTP JSON repair (critical)

Featherless DeepSeek sometimes splits the assistant reply across **duplicate `"content"` keys** in the HTTP JSON wrapper, e.g.:

```json
"content":"{\"","content":"reasoning\": \"...\", \"steps\": [...]}"
```

**Bug we hit:** `_parse_chat_response()` tried to “repair” when `content` count > 1, but the regex **broke valid JSON** → 36/40 `JSONDecodeError` on a 40-example run.

**Fix:** `packages/common/zo_common/llm.py` — brace-aware merge of duplicate `content` values before parse. Tests: `tests/test_llm_featherless.py`.

### Anomaly token budget

`predict_llm.py` uses **`max_tokens=256`** for anomaly. With reason-first prompts the model writes a long rule-by-rule checklist and the JSON is **truncated before `"valid"` / `"rule"`**. Parser then falls back to regex: finds `RULE_*` in reasoning text, no `INVALID` keyword → defaults to **valid=1**.

**Symptom:** all forbidden examples marked valid; spurious `rule` field in traces from first rule name mentioned in truncated reasoning.

**Likely fix:** raise anomaly `max_tokens` (512–768) or ask for shorter reasoning.

---

## Prompt design (`packages/eval/zo_eval/rules_context.py`)

### Structure

1. `load_system_general()` — role intro
2. **TASK** block (next-step / completion / anomaly)
3. **OUTPUT FORMAT** — mandatory JSON schema
4. Rules digest from `generation_rules.md` (grammar backbone, step vocab, forbidden rules)

We do **not** edit `generation_rules.md`; only our task/guidance/user-message layers.

### JSON output (scorer-facing)

- **Next-step:** `{"reasoning": "...", "steps": ["BEST", ...]}` — parser in `predict.py` (`extract_answer`, `_try_parse_json_response`)
- **Anomaly:** `{"reasoning": "...", "valid": true/false, "rule": null|"RULE_*"}`

Legacy `Answer:` / last-line fallbacks kept for non-JSON replies.

### Input formatting

| Format | User message |
|---|---|
| **Pipe (original)** | `SEP.join(steps)` → one line: `STEP A \| STEP B \| ...` |
| **Numbered (current)** | One step per line: `1. STEP A`, `2. STEP B`, … plus `Last executed step (#N): "..."` |

Numbered input **helped next-step** (30% → 40% top1 on smoke5) because the model stops jumping to the next grammar phase. **Did not fix anomaly** pad-rule confusion by itself.

### Prompt iterations (anomaly)

| Version | Change | Valid seqs | Forbidden caught |
|---|---|---|---|
| v1 (biased) | “Most sequences are VALID”, “Default valid=true” | 4/5 | 2/5 |
| v2 (pipe + unbiased + no reason-first) | Removed bias; pad FPs on all valid | 0/5 | 5/5 |
| reason-v1/v2 | Reason first; RULE_* = violation pattern; check every rule | **5/5** | **0/5** (truncation) |

**Removed per team preference (reason-v2):** explicit “compare step numbers” hints and worked examples like `#97 after #94 → OK`. Behavior unchanged vs reason-v1 on smoke5.

**Kept:** violation-pattern clarification, reason-then-answer, numbered input.

---

## Next-step failure modes (model behavior)

Errors are mostly **block-level vs step-level**, not JSON format.

1. **VIA_BLOCK digest vs gold:** `generation_rules.md` VIA_BLOCK starts at `DEPOSIT BARRIER METAL`. Gold often wants **`MEASURE VIA CD`** immediately after `CLEAN AFTER VIA ETCH`. Model reasoning: “via litho done → enter VIA_BLOCK → deposit metal.”
2. **Generic litho template:** digest says `CLEAN AFTER ETCH`; gold often wants **`CLEAN AFTER VIA ETCH`**. Model puts generic clean first (sometimes rank 2 hits gold).
3. **Pad window synonyms:** after `PAD WINDOW LITHO`, gold may be `DEVELOP PAD WINDOW` or `DEVELOP PHOTORESIST`; model skips to `PASSIVATION ETCH PAD OPENING`.
4. **Reasoning cites `_BLOCK` labels** (OK in reasoning; forbidden in `steps[]`).

Numbering fixed several **MEASURE VIA CD** cases where pipe format jumped straight to barrier metal.

---

## Anomaly failure modes

### 1. Pad-rule false positives (pipe + biased prompts)

**Rule:** `RULE_PAD_OPEN_BEFORE_DEP` — pad opening must come **after** `DEPOSIT PASSIVATION` and `CURE PASSIVATION`.

On **valid** sequences the model claimed e.g.:

> OPEN PAD WINDOW (step 97) appears **before** DEPOSIT PASSIVATION (step 94)

while numbered indices showed **94 < 97** (correct order). It cited the right numbers but **inverted before/after**. Likely drivers:

- Rule ID reads like the violation name (`PAD_OPEN_BEFORE_DEP`)
- Condensed digest lists pad steps before passivation in the one-liner
- “Check every rule” without a comparison procedure → template violation narrative

**Fix that worked:** reason-first + “RULE_* names violation patterns; valid when pattern does **not** apply” → **0 pad FPs** on valid seqs (reason-v1/v2).

### 2. Missing forbidden (reason-first + 256 tokens)

Model writes long per-rule checklist; response truncated mid-`RULE_DEP_NO_CLEAN` analysis. Incomplete JSON → parser defaults to **VALID**. All 5 forbidden missed on smoke5.

### 3. Obvious violations when not truncated

When responses complete (numbered run, shorter prompts), model **does** catch e.g. `RULE_SHIP_BEFORE_TEST`, `RULE_LITHO_LEVEL_SKIP`, true `RULE_PAD_OPEN_BEFORE_DEP` (pad at step 100, dep at 101).

---

## Eval runs (registry)

Runs under `~/.cache/zo-experiments/` (or `$ZO_EXPERIMENTS_DIR`).

| Run ID (suffix) | Version tag | Notes |
|---|---|---|
| `_0d0422` | `smoke5-v1` | First JSON prompts; pipe input; top1 10% |
| `_953bd9` | `deepseek-json-v1` | 40-ex broken run; 0% top1 (JSONDecodeError) |
| `_ff11d1` | (old prompts) | 40 next-step sequential; **47.5% top1** pre-JSON |
| `_7716a1` | `smoke5-numbered` | Numbered input; pad FP 5/5 on valid |
| `_1bb3f0` | `smoke5-reason-v1` | Valid 5/5; forbidden 0/5; top1 40% |
| `_5bb0b6` | `smoke5-reason-v2` | Same as v1 without step-number hint lines |

Inspect: `<run>/results/examples.jsonl`, `metrics_report.md`.

---

## Generate local eval sets

```bash
uv run zo-track make-local-eval --family MOSFET --n 5 --out extras/eval_local_mosfet5
```

`--n 5` → 5 test sequences → **10 next-step** (frac 0.6 + 0.8) + **10 anomaly** (5 valid + 5 forbidden).

For ~40 next-step only: `--n 20` → `extras/eval_local_mosfet20`.

---

## Recommendations

1. **Always** use JSON repair path in `llm.py` for Featherless DeepSeek.
2. **Featherless eval:** `--concurrency 1` unless you add retry-for-429 pass.
3. **Input:** keep **numbered** steps for zero-shot DeepSeek next-step.
4. **Anomaly:** increase `max_tokens` before scaling eval; verify `examples.jsonl` raw JSON ends with `"valid": ...`.
5. **Prompts:** keep reason-first + violation-pattern wording; avoid “default valid” bias; avoid rule-specific cheat-sheets unless grounded in `generation_rules.md`.
6. **Next-step gains beyond prompting:** may need training or richer digest (still sourced from rules md) — block summaries omit post-etch measure/clean micro-steps.

---

## Related code & tests

| Path | Purpose |
|---|---|
| `packages/eval/zo_eval/rules_context.py` | Prompt assembly |
| `packages/eval/zo_eval/predict.py` | JSON answer parsing |
| `packages/eval/zo_eval/predict_llm.py` | `RulesContextLLMPredictor`, token limits |
| `packages/common/zo_common/llm.py` | Featherless HTTP client + repair |
| `packages/eval/zo_eval/featherless.py` | Featherless eval driver |
| `packages/eval/zo_eval/concurrent_served.py` | Thread-pool for served backend |
| `tests/test_llm_featherless.py` | JSON repair |
| `tests/test_predict_json.py` | Answer extraction |
| `tests/test_rules_context.py` | Prompt content |
| `tests/test_concurrent_served.py` | Concurrency helper |

---

## Comparison vs smaller models (smoke only)

On 2-example smoke (not representative): Qwen 1.5B 0% top1 vs DeepSeek 100%. On 80-example pre-JSON run DeepSeek hit **47.5% top1** sequential — size matters; prompt/pipeline quality also dominated early broken runs (0% from parse failures, not model quality).

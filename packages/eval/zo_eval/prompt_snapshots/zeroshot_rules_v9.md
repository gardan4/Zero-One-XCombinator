# zeroshot_rules_v9 — unified JSON prompts without rules digest (2026-05-31)

Active prompts live in `packages/training/zo_train/prompts.py` (single source of truth).
`packages/eval/zo_eval/rules_context.py` delegates to prompts; set `ZO_RULES_IN_CONTEXT=1`
to restore the v8-style grammar/vocab/forbidden digest ablation.

## What changed from v8

- **Single prompt stack** for SFT train, finetuned eval, and zero-shot eval (base-model rich prompts unchanged).
- **No rules digest by default** — task instructions + JSON OUTPUT FORMAT + behavioral guidance only.
- **Numbered user input** + JSON assistant labels in instruct SFT data.
- **Breaking:** pipe-format finetuned checkpoints require retrain on regenerated `instruct_all.jsonl`.

## Eval version tags

- `unified-prompt-v1` — local compare zero-shot
- `unified-prompt-featherless-*-v1` — Featherless compare suite

## Revert

- Digest ablation: `ZO_RULES_IN_CONTEXT=1`
- Full pipe-format stack: git history before this migration

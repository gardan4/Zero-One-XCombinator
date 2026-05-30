# Kickoff evaluation inputs (organizer-distributed)

Distributed at the hackathon (May 2026). Use these paths for **final submission predictions**;
organizers hold ground-truth labels for scoring.

| File | Rows | Purpose |
|------|------|---------|
| `eval_input_valid.csv` | 600 | Tasks 1 & 2 — partial sequences at 60%/80% (`EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE`) |
| `eval_input_anomaly.csv` | 987 | Task 3 — unlabeled full sequences (`EXAMPLE_ID, FAMILY, SEQUENCE`) |
| `eval_metrics.py` | — | Official scoring script (stdlib only); needs organizer ground-truth CSVs |

## Run predictions on the kickoff set

```bash
uv run zo-track predict -p hf --model XCombinator/<checkpoint> --version v1 \
  --eval-set kickoff --tags split:id
# → experiments/<run>/results/{nextstep,completion,anomaly}.csv
```

## Self-evaluation during development

| What | How |
|------|-----|
| Labeled hold-out proxy | `just local-eval MOSFET` → `extras/eval_local/` + `gold.json` |
| Tagged comparison matrix | `just eval-suite packages/eval/eval_suites/local_compare.yaml` |
| Kickoff submission + promote | `just eval-suite packages/eval/eval_suites/kickoff_submit.yaml --model XCombinator/...` |
| Re-score without re-inference | `just rescore --results experiments/.../results --gold extras/eval_local/gold.json --self-check` |
| Promote to submission folder | `just promote kickoff-final <run_id>` → `extras/results/` |
| Grammar validity of completions | `just validate-completion experiments/.../results/completion.csv` |
| Exact metrics (labeled proxy) | `just self-check --results ... --gold ... --valid ... --anomaly ...` |
| Compare index | `extras/results/INDEX.json` lists all promoted runs + tags |

Formats and metric names: `training_data/generation_rules.md` §5.

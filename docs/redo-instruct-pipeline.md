# Redo runbook — instruct models + scaling study + dashboard (after new system prompts)

**Why this exists.** The system prompts in `zo_train/prompts.py` ARE the train+eval framing for the
instruct models. When new prompts land from main, every instruct SFT model and every eval must be
regenerated (a model trained on old prompts is graded under new ones → mismatch). This runbook is the
push-button redo. Run it from the cluster-driving worktree (the one holding `.env` with
`ZO_CLUSTER_PASSWORD`; currently `…/worktrees/bold-banzai-ac0b36`).

All cluster SSH uses `plink -batch -ssh -pw "$PW" a08trd0f@login01-ext.leonardo.cineca.it` with
`PW=$(python3 -c "print([l.split('=',1)[1].strip().strip(chr(34)).strip(chr(39)) for l in open('.env') if l.startswith('ZO_CLUSTER_PASSWORD=')][0])")`.
Cluster repo = `/leonardo/home/usertrain/a08trd0f/Zero-One-Philyr` (`$R`). Base model (staged) =
`/leonardo_scratch/large/usertrain/a08trd0f/hf-local/Qwen2.5-1.5B-Instruct`. Checkpoints land at
`/leonardo_scratch/large/usertrain/a08trd0f/zo-experiments/<run_id>/artifacts`.

## Already-landed durable fixes (survive the prompt change — do NOT redo)
- `packages/training/configs/leonardo_sft_fab_instruct.yaml` — `model:` is an absolute scratch path
  (NOT `$SCRATCH/…`, which doesn't expand in the SLURM env).
- All `leonardo_sft_scale_instruct_{100,300,800,2000}.yaml` — same absolute model path; `epochs: 1`.
- `scripts/build_scaling_corpora.py` — now emits **balanced INVALID anomaly negatives** (mirrors
  `datagen.build_all`), so scaling models can actually flag violations (was VALID-only → anomaly f1=0).
- `xcombinator-copilot/scripts/build-benchmarks.mjs` — `bestSlug` excludes `data-size:<N>` entries, so
  the headline "best" is the canonical full-data model, never a scaling point.
- `extras/eval_local/MOSFET/{eval_input_valid,eval_input_anomaly}.csv,gold.json` pulled local — the
  judge-eval CLI validates `--eval-dir` **locally** before submit (cluster has it too; they must match).

## Pipeline

### 1. Merge new prompts, then regenerate corpora locally (picks up NEW prompts automatically)
```bash
# (after the prompts PR is merged into this worktree and the tree is clean)
uv run python -m zo_train.datagen --build          # → data/generated/instruct_all.jsonl (canonical, balanced)
uv run python scripts/build_scaling_corpora.py     # → data/generated_scale/<N>/instruct_all.jsonl (balanced)
# sanity: anomaly should be ~50/50 VALID/INVALID
```
`data/` is NOT in the cluster sync-excludes (`_remote.py:_SYNC_EXCLUDES`), so the next `submit` WITH
prep tars + ships local `data/` and overwrites the cluster's stale corpus. Good — that's the delivery path.

### 2. Submit the canonical "best" training (WITH prep → syncs data/ + configs + code)
```bash
uv run zo-cluster submit -c packages/training/configs/leonardo_sft_fab_instruct.yaml
```
Note the printed run_id + job id. ~15–25 min (3 epochs, ~2.7k rows).

### 3. Submit the 4 scaling trainings (--no-prep; data already shipped in step 2)
```bash
for N in 100 300 800 2000; do
  uv run zo-cluster submit --no-prep -c packages/training/configs/leonardo_sft_scale_instruct_$N.yaml
done
```
Rough times @ epochs:1 — 100 ≈ 7 min, 300 ≈ 17 min, 800 ≈ 47 min, 2000 ≈ 1.5 h.

### 4. Wait for COMPLETED (background waiter pattern)
```bash
PW=…; for i in $(seq 1 120); do st=$(plink … "sacct -X -n -j <ids> --format=State%14"); \
  echo "$st" | grep -qE "RUNNING|PENDING" && sleep 30 || { echo DONE; echo "$st"; break; }; done
```
(launch as a background Bash so completion re-invokes you).

### 5. Eval each checkpoint + promote on the compute node
Instruct models, instruct eval → **NO `ZO_PROMPT_LEGACY`** (framing matches). `--tags` REPLACES the
defaults, so pass the full set. `--promote <slug>` writes to `$R/extras/results/<slug>` on the cluster.
```bash
# canonical best:
uv run zo-cluster judge-eval --local --no-prep \
  --model <best ckpt artifacts> --predictor hf --version sft-instruct-all-v2 \
  --eval-dir extras/eval_local/MOSFET \
  --tags real-run,reportable,role:finetuned,split:id,family:MOSFET \
  --train-run <best run_id> --promote hf-sft-instruct-all
# scaling (per N): add scale,data-size:$N and slug hf-sft-scale-$N
uv run zo-cluster judge-eval --local --no-prep \
  --model <scale-$N ckpt artifacts> --predictor hf --version sft-scale-instruct-$N-v2 \
  --eval-dir extras/eval_local/MOSFET \
  --tags real-run,reportable,role:finetuned,split:id,family:MOSFET,scale,data-size:$N \
  --train-run <scale-$N run_id> --promote hf-sft-scale-$N
```
Eval jobs ~2–4 min once RUNNING. Confirm `COMPLETED 0:0` (a missing `gold.json` → eval-dir mismatch).

### 6. Pull cluster-promoted results to laptop + re-promote locally (updates INDEX.json)
```bash
for N in 100 300 800 2000; do bash scripts/pull_promote_scale.sh $N; done
# best: same idea, slug hf-sft-instruct-all (pscp $R/extras/results/hf-sft-instruct-all → stage → zo-track promote)
```

### 7. Remove the broken RAW entries (mis-evaled under instruct framing)
```bash
# hf-sft-all + hf-sft-lofo-{ic,igbt,mosfet} were raw-framed; delete their dirs + INDEX.json keys
# (or re-eval them with ZO_PROMPT_LEGACY=1 IF you want a raw base-vs-best panel — not the default plan)
```

### 8. Rebuild the dashboard
```bash
cd xcombinator-copilot && npm run build:benchmarks && npm run build
# scaling panel = data-size points; best = hf-sft-instruct-all; base = baseline-ngram (or zeroshot)
```

### 9. Update the live Mac copilot demo server for the new instruct "best"
`scripts/serve_copilot_mac.py` currently forces `_completion_mode → True` for ALL models (raw
"Process sequence: a | b | " framing). The new instruct "best" is trained on the chat/system framing,
so for it the server must build prompts via `tok.apply_chat_template(build_messages(...))` instead of
the raw reframe — otherwise the demo model is run off-distribution. Also point `DEFAULT_MODELS`
`sft-fab-all` → the new instruct checkpoint dir (e.g. `~/zo-models/sft-instruct-all`) and re-`pscp` the
weights down. (Don't revert the user's `confidence`/`_step_confidence` additions.)

## What the numbers said on the OLD prompts (provisional baseline to beat)
MOSFET, n=200: n-gram baseline nextstep top1 **0.69**, completion block_acc **0.637**, anomaly f1 **0.89**
(roc_auc 0.9999 — a strong classical baseline). Instruct scaling models: nextstep 0.69–0.71, completion
block_acc ~0.685 but **edit-dist 0.18 vs n-gram 0.50** (the real win), anomaly **f1 0.00** (the VALID-only
bug — fixed in step 1). Scaling was **flat/saturated by 100 seqs/family** for nextstep+completion.

**Story to lead with** (n-gram is hard to beat on top-1): the live copilot demo (base Qwen rambles vs
finetuned emits clean valid steps), completion coherence (edit-dist), anomaly after the balance fix, and
early data-scaling saturation.

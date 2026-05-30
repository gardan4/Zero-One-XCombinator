# Zero One Philyr task runner. Run `just` to list recipes.
set shell := ["bash", "-uc"]

default:
    @just --list

# --- setup -------------------------------------------------------------------

# Install local deps (no GPU/ML deps — those install on the cluster via `just gpu-sync`).
setup:
    uv sync
    cd apps/frontend && npm install

# Install the heavy ML stack (torch/trl/transformers/vllm). Run this on the cluster node.
gpu-sync:
    uv sync --extra gpu

# --- run the stack -----------------------------------------------------------

# Backend + frontend together (Ctrl-C stops both).
dev:
    ./scripts/dev.sh

backend:
    uv run uvicorn zo_backend.main:app --reload --port "${ZO_API_PORT:-8000}"

frontend:
    cd apps/frontend && npm run dev

# --- training / RL -----------------------------------------------------------

# Supervised finetune from a config (local smoke run or inside a SLURM job).
# Extra args pass through, e.g. `just train cfg.yaml --dry-run` or `--run-id <id>`.
train config *args:
    uv run zo-train sft --config {{config}} {{args}}

# GRPO / RL run from a config. Extra args pass through (e.g. --dry-run).
grpo config *args:
    uv run zo-train grpo --config {{config}} {{args}}

# Render + submit a training job to the Leonardo SLURM cluster over SSH.
submit config:
    uv run zo-cluster submit --config {{config}}

# Tail the most recent submitted SLURM job.
cluster-watch:
    uv run zo-cluster watch

# Pull meta.json + metrics.jsonl for a cluster run into the local experiments/ store.
cluster-pull-run run_id:
    uv run zo-cluster pull-run {{run_id}}

# Leonardo smoke finetune (cross-platform — or: python scripts/leonardo_smoke.py).
leonardo-smoke *args:
    uv run zo-cluster leonardo-smoke {{args}}

# --- judge / repro eval (Leonardo login node — see docs/leonardo-eval.md) ------------

# One-time: GPU deps + deterministic local eval set under extras/eval_local/.
judge-setup:
    uv run zo-cluster judge-setup

# Download ZO_INFER_MODEL from Hugging Face to $SCRATCH (login node only).
judge-stage *args:
    uv run zo-cluster stage-model {{args}}

# Submit batch inference + track scoring (default: -p hf, all 3 tasks).
judge-eval *args:
    uv run zo-cluster judge-eval {{args}}

# Optional live demo: vLLM on a GPU node (SSH tunnel — secondary).
judge-serve *args:
    uv run zo-cluster judge-serve {{args}}

# --- eval / agent ------------------------------------------------------------

eval task model:
    uv run zo-eval run --task {{task}} --model {{model}}

# Track eval: predict → 3 CSVs + metrics_report.md + version-tagged registry run.
track *args:
    uv run zo-track predict {{args}}

# Re-score existing CSVs against gold (no re-inference).
rescore *args:
    uv run zo-track rescore {{args}}

# Copy run results to extras/results/<slug>/ for submission.
# Usage: just promote kickoff-final 20260530_123456_eval_abc
promote slug run_id:
    uv run zo-track promote {{run_id}} --slug {{slug}}

# Run tagged eval matrix from YAML (packages/eval/eval_suites/).
eval-suite config *args:
    uv run zo-track suite {{config}} {{args}}

# Official scorer self-check on labeled proxy set.
self-check *args:
    uv run zo-track self-check {{args}}

# Export gold.json → organizer GT CSVs.
export-gt *args:
    uv run zo-track export-gt {{args}}

# Synthesize an organizer-format local eval set (+ gold.json) from a family's held-out test split.
local-eval family *args:
    uv run zo-track make-local-eval --family {{family}} {{args}}

# Run predict on organizer kickoff eval inputs (data/industrial-infineon/eval/).
kickoff-predict *args:
    uv run zo-track predict --eval-set kickoff {{args}}

# Grammar-check completion CSV against kickoff partials (validate_sequence proxy).
validate-completion completion *args:
    uv run zo-track validate --completion {{completion}} {{args}}

# Exact organizer scoring when ground-truth CSVs are available.
score-official *args:
    uv run zo-track score-official {{args}}

agent scenario model:
    uv run zo-agent run --scenario {{scenario}} --model {{model}}

# Serve a (fine-tuned) model with an OpenAI-compatible vLLM endpoint for eval/agent.
serve model port="8001":
    uv run python -m vllm.entrypoints.openai.api_server --model {{model}} --port {{port}}

# --- experiments / worktrees -------------------------------------------------

# List all experiment runs (tags column; filter with --tag).
runs *args:
    uv run zo-runs ls {{args}}

# Find runs by tag substring.
runs-tag query:
    uv run zo-runs tag {{query}}

# Write HF training_manifest.json + README for a run's artifacts folder.
hub-manifest run_id *args:
    uv run zo-runs hub-manifest {{run_id}} {{args}}

# Create an isolated git worktree + branch for a parallel experiment.
wt name:
    ./scripts/wt.sh {{name}}

# --- data --------------------------------------------------------------------

# (Re)generate the shared deterministic corpus → data/generated (or $ZO_DATA_DIR).
# Seeded ⇒ byte-identical everywhere; if `git status` stays clean on splits.json /
# manifest.json afterwards, your corpus matches the team's. A dirty diff = you drifted.
data:
    uv run python -m zo_train.datagen --build

# --- quality -----------------------------------------------------------------

lint:
    uv run ruff check .

fmt:
    uv run ruff format .

test:
    uv run pytest

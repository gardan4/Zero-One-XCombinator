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

# Leonardo smoke finetune (cross-platform — or: uv run python scripts/leonardo_smoke.py).
leonardo-smoke *args:
    uv run zo-cluster leonardo-smoke {{args}}

# --- eval / agent ------------------------------------------------------------

eval task model:
    uv run zo-eval run --task {{task}} --model {{model}}

agent scenario model:
    uv run zo-agent run --scenario {{scenario}} --model {{model}}

# Serve a (fine-tuned) model with an OpenAI-compatible vLLM endpoint for eval/agent.
serve model port="8001":
    uv run python -m vllm.entrypoints.openai.api_server --model {{model}} --port {{port}}

# --- experiments / worktrees -------------------------------------------------

# List all experiment runs.
runs:
    uv run zo-runs ls

# Create an isolated git worktree + branch for a parallel experiment.
wt name:
    ./scripts/wt.sh {{name}}

# --- quality -----------------------------------------------------------------

lint:
    uv run ruff check .

fmt:
    uv run ruff format .

test:
    uv run pytest

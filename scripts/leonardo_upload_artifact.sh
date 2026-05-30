#!/usr/bin/env bash
set -euo pipefail

# Run on the Leonardo LOGIN node after a SLURM job completes.
# Uploads scratch artifacts to Hugging Face (large files never leave via compute proxy).
# W&B sync here is only a fallback when the GPU job could not reach the API live.

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_id> [hub_model_id]" >&2
  exit 2
fi

RUN_ID="$1"
HUB_MODEL_ID="${2:-XCombinator/leonardo-smoke-qwen-0.5b-lora}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

export PATH="$HOME/.local/bin:$PATH"
export ZO_EXPERIMENTS_DIR="${ZO_CLUSTER_EXPERIMENTS_DIR:?Set ZO_CLUSTER_EXPERIMENTS_DIR in .env}"
export WANDB_ENTITY="${WANDB_ENTITY:-XCombinator}"
export WANDB_PROJECT="${WANDB_PROJECT:-XCombinator}"

# Reuse the GPU venv staged on the login node (light `uv run` would drop huggingface_hub).
uv sync --extra gpu --offline 2>/dev/null || uv sync --extra gpu

ARTIFACT_DIR="$ZO_EXPERIMENTS_DIR/$RUN_ID/artifacts"
WANDB_DIR="$ZO_EXPERIMENTS_DIR/$RUN_ID/wandb"

if [[ ! -d "$ARTIFACT_DIR" ]]; then
  echo "Artifact directory not found: $ARTIFACT_DIR" >&2
  exit 1
fi

uv run python - <<PY
import os
from pathlib import Path

from huggingface_hub import HfApi

artifact = Path("$ARTIFACT_DIR")
readme = artifact / "README.md"
if readme.exists():
    text = readme.read_text()
    base_dir = os.environ.get("ZO_SMOKE_BASE_MODEL_DIR", "")
    hub_base = os.environ.get("ZO_SMOKE_BASE_MODEL_HF_ID", "Qwen/Qwen2.5-0.5B-Instruct")
    if base_dir and base_dir in text:
        text = text.replace(base_dir, hub_base)
    readme.write_text(text)

api = HfApi(token="${HF_TOKEN:?Set HF_TOKEN in .env}")api.create_repo(repo_id="$HUB_MODEL_ID", repo_type="model", private=True, exist_ok=True)
api.upload_folder(
    repo_id="$HUB_MODEL_ID",
    repo_type="model",
    folder_path="$ARTIFACT_DIR",
    commit_message="Upload Leonardo run $RUN_ID",
)
print("uploaded $ARTIFACT_DIR to https://huggingface.co/$HUB_MODEL_ID")
PY

if [[ -d "$WANDB_DIR" ]]; then
  offline="$(find "$WANDB_DIR" -maxdepth 3 -type d -name 'offline-run-*' 2>/dev/null | head -1)"
  if [[ -n "$offline" ]]; then
    echo "W&B offline dir found ÔÇö syncing fallback run from login node"
    WANDB_MODE=online uv run wandb sync "$offline" || true
  fi
fi

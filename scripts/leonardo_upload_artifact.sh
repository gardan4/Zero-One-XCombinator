#!/usr/bin/env bash
set -euo pipefail

# Run on the Leonardo LOGIN node after a SLURM job completes.
# Uploads scratch artifacts to Hugging Face (large files never leave via compute proxy).
# Writes training_manifest.json + README.md (tags + hyperparams) before upload.

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_id> [hub_model_id]" >&2
  exit 2
fi

RUN_ID="$1"
HUB_MODEL_ID="${2:-XCombinator/leonardo-smoke-qwen-0.5b-lora}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export PATH="$HOME/.local/bin:$PATH"
export ZO_EXPERIMENTS_DIR="${ZO_CLUSTER_EXPERIMENTS_DIR:?Set ZO_CLUSTER_EXPERIMENTS_DIR in .env}"
export WANDB_ENTITY="${WANDB_ENTITY:-XCombinator}"
export WANDB_PROJECT="${WANDB_PROJECT:-XCombinator}"

uv sync --extra gpu --offline 2>/dev/null || uv sync --extra gpu

ARTIFACT_DIR="$ZO_EXPERIMENTS_DIR/$RUN_ID/artifacts"
RUN_DIR="$ZO_EXPERIMENTS_DIR/$RUN_ID"
WANDB_DIR="$ZO_EXPERIMENTS_DIR/$RUN_ID/wandb"

if [[ ! -d "$ARTIFACT_DIR" ]]; then
  echo "Artifact directory not found: $ARTIFACT_DIR" >&2
  exit 1
fi

# Stamp HF model card + training_manifest.json from registry meta + config.yaml
uv run python - <<PY
from zo_common.hub_metadata import write_hub_artifact_metadata

write_hub_artifact_metadata(
    "$ARTIFACT_DIR",
    "$RUN_ID",
    hub_model_id="$HUB_MODEL_ID",
    config_path="$RUN_DIR/config.yaml",
)
print("wrote training_manifest.json + README.md -> $ARTIFACT_DIR")
PY

uv run python - <<PY
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

artifact = Path("$ARTIFACT_DIR")
run_dir = Path("$RUN_DIR")
api = HfApi(token="${HF_TOKEN:?Set HF_TOKEN in .env}")
api.create_repo(repo_id="$HUB_MODEL_ID", repo_type="model", private=True, exist_ok=True)
commit = api.upload_folder(
    repo_id="$HUB_MODEL_ID",
    repo_type="model",
    folder_path=str(artifact),
    commit_message="Upload Leonardo run $RUN_ID",
)
revision = getattr(commit, "oid", None) or getattr(commit, "commit_hash", None)
(run_dir / "hf_upload.json").write_text(json.dumps({
    "hub_model_id": "$HUB_MODEL_ID",
    "revision": revision,
    "url": "https://huggingface.co/$HUB_MODEL_ID",
}, indent=2))
try:
    from zo_common.wandb_runs import log_hf_to_training_run

    log_hf_to_training_run("$RUN_ID", "$HUB_MODEL_ID", revision=revision)
except Exception as exc:
    print(f"W&B HF metadata log skipped: {exc}")
print("uploaded $ARTIFACT_DIR to https://huggingface.co/$HUB_MODEL_ID")
if revision:
    print(f"hf revision: {revision}")
PY

if [[ -d "$WANDB_DIR" ]]; then
  offline="$(find "$WANDB_DIR" -maxdepth 3 -type d -name 'offline-run-*' 2>/dev/null | head -1)"
  if [[ -n "$offline" ]]; then
    echo "W&B offline dir found — syncing fallback run from login node"
    WANDB_MODE=online uv run wandb sync "$offline" || true
  fi
fi

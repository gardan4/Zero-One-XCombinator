#!/usr/bin/env bash
# Run on a Leonardo LOGIN node before any SLURM GPU job.
# Downloads wheels, Python deps, HF weights, and warms imports — compute runs `uv sync --offline`.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

: "${HF_HOME:?Set HF_HOME in .env (use \$SCRATCH/hf on Leonardo)}"
: "${ZO_SMOKE_BASE_MODEL_DIR:?Set ZO_SMOKE_BASE_MODEL_DIR in .env}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${HF_HOME}/uv-cache}"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME"

echo "==> [staging] uv sync --extra gpu (full ML stack on shared storage)"
uv sync --extra gpu

echo "==> [staging] import warm-up (populate HF/uv caches, no GPU required)"
uv run python - <<'PY'
import importlib

for mod in ("torch", "transformers", "trl", "peft", "datasets", "wandb", "accelerate"):
    importlib.import_module(mod)
print("import warm-up ok")
PY

echo "==> [staging] base model + tokenizer to scratch"
uv run python - <<'PY'
import os

from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = os.environ.get("ZO_PRESTAGE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct").strip()
local_dir = os.environ["ZO_SMOKE_BASE_MODEL_DIR"].strip()
snapshot_download(repo_id=repo_id, local_dir=local_dir)
AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
AutoModelForCausalLM.from_pretrained(local_dir, local_files_only=True)
print(f"pre-staged {repo_id} at {local_dir}")
PY

echo "==> [staging] toy SFT dataset (in-process, no download)"
uv run python - <<'PY'
from zo_train.data import load_sft_dataset
from zo_common import ExperimentConfig

cfg = ExperimentConfig.from_yaml("packages/training/configs/leonardo_smoke_hf.yaml")
ds = load_sft_dataset(cfg)
print(f"toy dataset rows={len(ds)}")
PY

echo "==> [staging] W&B login-node smoke (direct internet, not via proxy)"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_MODE=online uv run zo-train wandb-smoke --run-id "staging-$(date +%Y%m%d%H%M%S)" || true
fi

echo "==> [staging] done — submit SLURM job next (compute uses uv sync --offline + proxy for W&B only)"

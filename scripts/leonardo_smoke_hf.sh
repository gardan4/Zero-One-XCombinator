#!/usr/bin/env bash
set -euo pipefail

# Leonardo pipeline — prefer the cross-platform Python entry point:
#   uv run python scripts/leonardo_smoke.py
# This bash wrapper uses rsync (macOS/Linux with OpenSSH).
# 1. Sync repo to the login node.
# 2. Pre-stage on login: uv/gpu deps, HF weights, import warm-up (leonardo_remote_prestage.sh).
# 3. SLURM GPU job: offline HF, live W&B via proxy (entity/project: XCombinator/XCombinator).
# 4. After job: upload adapter to Hugging Face from login (leonardo_upload_artifact.sh).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-packages/training/configs/leonardo_smoke_hf.yaml}"

set -a
# shellcheck disable=SC1091
source "$ROOT/.env"
set +a

: "${ZO_CLUSTER_HOST:?Set ZO_CLUSTER_HOST in .env}"
: "${ZO_CLUSTER_USER:?Set ZO_CLUSTER_USER in .env}"
: "${ZO_CLUSTER_REPO_DIR:?Set ZO_CLUSTER_REPO_DIR in .env}"
: "${HF_TOKEN:?Set HF_TOKEN in .env so the job can push to Hugging Face}"

TARGET="${ZO_CLUSTER_USER}@${ZO_CLUSTER_HOST}"
SSH_OPTS=(-o ServerAliveInterval=30 -o ServerAliveCountMax=4)

echo "==> Sync repo to ${TARGET}:${ZO_CLUSTER_REPO_DIR}"
ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p '$ZO_CLUSTER_REPO_DIR'"
rsync -az --delete \
  --exclude '.git/' \
  --exclude '.env' \
  --exclude '.venv/' \
  --exclude 'apps/frontend/node_modules/' \
  --exclude 'experiments/*/' \
  --exclude 'hf_cache/' \
  --exclude 'slurm_logs/' \
  --exclude 'wandb/' \
  "$ROOT/" "$TARGET:$ZO_CLUSTER_REPO_DIR/"
ssh "${SSH_OPTS[@]}" "$TARGET" "cd '$ZO_CLUSTER_REPO_DIR' && find scripts -name '*.sh' -exec sed -i 's/\r$//' {} +"

echo "==> Write cluster .env with local secrets (ignored by git)"
tmp_env="$(mktemp)"
env | grep -E '^(ZO_|HF_TOKEN=|HF_HOME=|WANDB_)' \
  | grep -v '^ZO_CLUSTER_PASSWORD=' \
  | grep -v '^ZO_CLUSTER_HOSTKEY=' | tr -d '\r' > "$tmp_env"
scp "${SSH_OPTS[@]}" "$tmp_env" "$TARGET:$ZO_CLUSTER_REPO_DIR/.env"
rm -f "$tmp_env"

echo "==> Pre-stage GPU environment and base model on the login node"
ssh "${SSH_OPTS[@]}" "$TARGET" "bash '$ZO_CLUSTER_REPO_DIR/scripts/leonardo_remote_prestage.sh'"

echo "==> Submit short Leonardo smoke finetune"
uv run zo-cluster submit --config "$CONFIG"

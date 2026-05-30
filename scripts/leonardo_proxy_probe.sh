#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a
source .env
set +a
if [[ -n "${ZO_CLUSTER_PROXY:-}" ]]; then
  PROXY_URL="$ZO_CLUSTER_PROXY"
elif [[ -n "${PROXY_PASS:-}" ]]; then
  PROXY_URL="http://proxyuser:${PROXY_PASS}@10.99.0.1:38425"
else
  echo "Set ZO_CLUSTER_PROXY in .env (deck p.95) or PROXY_PASS=... to probe." >&2
  exit 2
fi
code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 20 -x "$PROXY_URL" https://api.wandb.ai || echo 000)"
echo "proxy_probe http_code=$code"
if [[ "$code" == "000" ]]; then
  exit 1
fi

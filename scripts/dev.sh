#!/usr/bin/env bash
# Run the FastAPI backend and the Next.js frontend together. Ctrl-C stops both.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

port="${ZO_API_PORT:-8000}"

# On Ctrl-C / TERM, kill the whole process group (both servers + their children).
trap 'trap - INT TERM; echo; echo "stopping..."; kill 0' INT TERM

echo "backend  -> http://localhost:$port"
uv run uvicorn zo_backend.main:app --reload --port "$port" &

echo "frontend -> http://localhost:3000"
( cd apps/frontend && npm run dev ) &

wait

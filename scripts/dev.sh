#!/usr/bin/env bash
# Legacy wrapper — prefer: just dev  or  uv run python scripts/dev.py
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
exec uv run python scripts/dev.py

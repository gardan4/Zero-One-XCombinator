#!/usr/bin/env python3
"""First-time repo setup — Windows, macOS, Linux (Python + uv + npm only).

Run from repo root:

  uv run python scripts/setup.py          # uv sync + frontend npm install
  python scripts/setup.py                 # same if uv is on PATH

Or: ``just setup`` (calls this after ``uv sync``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from _tools import resolve_tool, run, tool_cmd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "apps" / "frontend"


def main() -> None:
    parser = argparse.ArgumentParser(description="Install light workspace + frontend deps.")
    parser.add_argument(
        "--npm-only",
        action="store_true",
        help="Skip uv sync (used by `just setup` after it already ran uv sync).",
    )
    args = parser.parse_args()

    if not args.npm_only:
        resolve_tool("uv", "Install: https://docs.astral.sh/uv/ or `mise install`")
        run(tool_cmd("uv", "sync"), ROOT)

    run(tool_cmd("npm", "install"), FRONTEND)
    print("\nSetup complete.")
    print("  uv run python scripts/dev.py   # backend :8000 + frontend :3000")
    print("  uv run pytest                  # unit tests (excludes integration)")


if __name__ == "__main__":
    main()

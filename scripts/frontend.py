#!/usr/bin/env python3
"""Run npm in apps/frontend (install deps first if needed). Cross-platform."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from _tools import npm_next_bin, run, tool_cmd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "apps" / "frontend"


def main() -> None:
    script = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if not npm_next_bin(FRONTEND).exists():
        print("frontend: npm install (first time)...")
        run(tool_cmd("npm", "install"), FRONTEND)
    run(tool_cmd("npm", "run", script), FRONTEND)


if __name__ == "__main__":
    main()

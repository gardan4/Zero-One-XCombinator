#!/usr/bin/env python3
"""Start FastAPI backend + Next.js frontend. Ctrl+C stops both. Windows, macOS, Linux.

  uv run python scripts/dev.py
  just dev
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from _tools import npm_next_bin, resolve_tool, tool_cmd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "apps" / "frontend"
PORT = os.environ.get("ZO_API_PORT", "8000")


def main() -> None:
    resolve_tool("uv", "Run setup first: uv run python scripts/setup.py")
    resolve_tool("npm", "Run setup first: uv run python scripts/setup.py")

    if not npm_next_bin(FRONTEND).exists():
        print("frontend: npm install (first time)...")
        subprocess.run(tool_cmd("npm", "install"), cwd=FRONTEND, check=True)

    procs: list[subprocess.Popen[bytes]] = []

    print(f"backend  -> http://localhost:{PORT}")
    procs.append(
        subprocess.Popen(
            tool_cmd("uv", "run", "uvicorn", "zo_backend.main:app", "--reload", f"--port={PORT}"),
            cwd=ROOT,
        )
    )

    print("frontend -> http://localhost:3000")
    procs.append(subprocess.Popen(tool_cmd("npm", "run", "dev"), cwd=FRONTEND))

    stopping = False

    def shutdown(_signum=None, _frame=None) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        print("\nstopping...")
        for p in procs:
            if p.poll() is None:
                p.terminate()
        deadline = time.time() + 8
        for p in procs:
            while p.poll() is None and time.time() < deadline:
                time.sleep(0.1)
            if p.poll() is None:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, shutdown)

    try:
        while not stopping:
            for p in procs:
                if p.poll() is not None:
                    shutdown()
            time.sleep(0.3)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()

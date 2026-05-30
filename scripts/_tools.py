"""Resolve CLI tools to absolute paths (Windows-safe for subprocess)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def resolve_tool(name: str, hint: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SystemExit(f"{name} not found. {hint}")
    return path


def tool_cmd(name: str, *args: str) -> list[str]:
    return [resolve_tool(name, f"Install {name} or see docs/setup.md"), *args]


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def npm_next_bin(frontend_dir: Path) -> Path:
    name = "next.cmd" if sys.platform == "win32" else "next"
    return frontend_dir / "node_modules" / ".bin" / name

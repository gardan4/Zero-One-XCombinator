#!/usr/bin/env python3
"""Run zo-cluster commands without uv (pip install -r requirements-orchestrator.txt).

Examples:
  python scripts/zo_cluster.py judge-eval --dry-run --no-stage --eval-dir extras/eval_local
  python scripts/zo_cluster.py judge-setup
  python scripts/zo_cluster.py submit --config packages/training/configs/sft_smoke.yaml --dry-run

Optional equivalent if uv is installed:  uv run zo-cluster ÔÇª
"""

from __future__ import annotations

import sys
from pathlib import Path

_ORCHESTRATOR_DEPS = (
    ("typer", "typer"),
    ("pydantic", "pydantic"),
    ("yaml", "pyyaml"),
    ("jinja2", "jinja2"),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_deps() -> None:
    missing = [pip for mod, pip in _ORCHESTRATOR_DEPS if not _can_import(mod)]
    if not missing:
        return
    req = _repo_root() / "requirements-orchestrator.txt"
    hint = f"python -m pip install -r {req}" if req.is_file() else f"python -m pip install {' '.join(missing)}"
    raise SystemExit(
        "Missing packages: "
        + ", ".join(missing)
        + f"\nInstall from repo root:\n  {hint}\n"
        "Then run: python scripts/zo_cluster.py ÔÇª"
    )


def _can_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def _bootstrap() -> None:
    import os

    for sub in ("packages/common", "packages/training"):
        p = str(_repo_root() / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    env_file = _repo_root() / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, _, val = s.partition("=")
        key = key.strip()
        if key:
            os.environ.setdefault(key, val.strip().strip('"').strip("'"))


def main() -> None:
    _ensure_deps()
    _bootstrap()
    from zo_train.cluster.submit import app

    app(prog_name="zo-cluster")


if __name__ == "__main__":
    main()

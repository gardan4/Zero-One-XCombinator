#!/usr/bin/env python3
"""Leonardo smoke finetune — Windows, macOS, Linux (Python + pip + ssh only).

Prerequisites (once):
  python -m pip install -r requirements-orchestrator.txt

Optional: OpenSSH (ssh/scp) or PuTTY (plink/pscp + ZO_CLUSTER_PASSWORD in .env).

Examples:
  python scripts/leonardo_smoke.py --dry-run
  python scripts/leonardo_smoke.py
  python scripts/leonardo_smoke.py --wait-upload
"""

from __future__ import annotations

import argparse
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
        "Then run: python scripts/leonardo_smoke.py --dry-run"
    )


def _can_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def _bootstrap() -> None:
    import os

    training = _repo_root() / "packages" / "training"
    common = _repo_root() / "packages" / "common"
    for p in (str(common), str(training)):
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


def main() -> int:
    _ensure_deps()
    _bootstrap()
    parser = argparse.ArgumentParser(description="Leonardo smoke finetune pipeline")
    parser.add_argument(
        "-c",
        "--config",
        default="packages/training/configs/leonardo_smoke_hf.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--skip-prestage", action="store_true")
    parser.add_argument("--wait-upload", action="store_true")
    args = parser.parse_args()

    from zo_train.cluster.leonardo_smoke import run_leonardo_smoke

    run_leonardo_smoke(
        config=args.config,
        dry_run=args.dry_run,
        skip_sync=args.skip_sync,
        skip_prestage=args.skip_prestage,
        wait_upload=args.wait_upload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

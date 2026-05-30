#!/usr/bin/env python3
"""Leonardo smoke finetune — works on Windows, macOS, and Linux (Python + ssh only).

Prerequisites:
  - Python 3.11+ with ``uv`` (or run via ``uv run python scripts/leonardo_smoke.py``)
  - OpenSSH client (``ssh``/``scp``) or PuTTY (``plink``/``pscp`` + ``ZO_CLUSTER_PASSWORD``)
  - ``.env`` filled from ``.env.example`` (cluster user, HF_TOKEN, etc.)

Examples:
  uv run python scripts/leonardo_smoke.py --dry-run
  uv run python scripts/leonardo_smoke.py
  uv run python scripts/leonardo_smoke.py --wait-upload
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bootstrap() -> None:
    training = _repo_root() / "packages" / "training"
    common = _repo_root() / "packages" / "common"
    for p in (str(common), str(training)):
        if p not in sys.path:
            sys.path.insert(0, p)


def main() -> int:
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

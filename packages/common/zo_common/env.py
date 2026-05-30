"""Best-effort ``.env`` loader — no extra dependency.

Populates ``os.environ`` from ``<repo>/.env`` so the CLIs pick up ``WANDB_API_KEY``, ``HF_TOKEN``,
``ZO_MODEL_BASE_URL``, etc. without exporting them by hand. Real environment variables always win
(``setdefault``); a missing ``.env`` is a no-op. Call this at CLI entry points (not on library
import) so libraries stay side-effect-free.
"""

from __future__ import annotations

import os
from pathlib import Path

from zo_common.paths import repo_root


def load_dotenv(path: str | Path | None = None) -> None:
    env_file = Path(path) if path else (repo_root() / ".env")
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, _, val = s.partition("=")
        key = key.strip()
        if key:
            os.environ.setdefault(key, val.strip().strip('"').strip("'"))

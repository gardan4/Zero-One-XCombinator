"""Best-effort ``.env`` loader — no extra dependency.

Populates ``os.environ`` from ``<repo>/.env`` so the CLIs pick up ``WANDB_API_KEY``, ``HF_TOKEN``,
``ZO_MODEL_BASE_URL``, etc. without exporting them by hand. Real environment variables always win
(``setdefault``); a missing ``.env`` is a no-op. Call this at CLI entry points (not on library
import) so libraries stay side-effect-free.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from zo_common.paths import repo_root

_STRIP_SECRET_KEYS = frozenset({"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "WANDB_API_KEY"})


def normalize_env_value(key: str, value: str) -> str:
    """Strip quotes and trailing/leading whitespace (W&B rejects padded API keys)."""
    val = value.strip().strip('"').strip("'")
    if key in _STRIP_SECRET_KEYS:
        val = val.strip()
    return val


def expand_env_refs(text: str) -> str:
    """Expand ``${VAR}`` and ``$VAR`` using ``os.environ`` (works on Windows, unlike expandvars)."""

    def brace(m: re.Match[str]) -> str:
        return os.environ.get(m.group(1), m.group(0))

    def plain(m: re.Match[str]) -> str:
        return os.environ.get(m.group(1), m.group(0))

    text = re.sub(r"\$\{([^}]+)\}", brace, text)
    return re.sub(r"\$([A-Z_][A-Z0-9_]*)", plain, text)


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
            os.environ.setdefault(key, normalize_env_value(key, val))

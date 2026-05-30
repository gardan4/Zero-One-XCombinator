"""Lightweight Hugging Face repo inspection (stdlib + optional huggingface_hub / httpx)."""

from __future__ import annotations

import os
from pathlib import Path


def _names_indicate_full_weights(names: set[str]) -> bool:
    if "adapter_config.json" in names and "adapter_model.safetensors" in names:
        if "model.safetensors" in names or any(
            n.startswith("model-") and n.endswith(".safetensors") for n in names
        ):
            return True
        return False
    return bool(
        "model.safetensors" in names
        or "pytorch_model.bin" in names
        or any(n.startswith("model-") and n.endswith(".safetensors") for n in names)
    )


def _local_has_full_weights(path: str) -> bool:
    names = {p.name for p in Path(path).iterdir() if p.is_file()}
    return _names_indicate_full_weights(names)


def _remote_list_files(repo_id: str, token: str | None) -> list[str] | None:
    try:
        from huggingface_hub import list_repo_files

        return list_repo_files(repo_id, token=token)
    except ImportError:
        pass
    except Exception:
        return None
    try:
        import httpx

        headers = {"Authorization": f"Bearer {token}"} if token else {}
        for branch in ("main", "master"):
            url = f"https://huggingface.co/api/models/{repo_id}/tree/{branch}"
            resp = httpx.get(url, headers=headers, timeout=30.0)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            return [item["path"] for item in resp.json()]
    except Exception:
        return None
    return None


def hub_has_full_weights(repo_or_path: str, token: str | None = None) -> bool:
    """True when the path or HF repo contains a full model (not adapter-only)."""
    if os.path.isdir(repo_or_path):
        return _local_has_full_weights(repo_or_path)
    files = _remote_list_files(repo_or_path, token)
    if not files:
        return False
    names = {f.rsplit("/", 1)[-1] for f in files}
    return _names_indicate_full_weights(names)

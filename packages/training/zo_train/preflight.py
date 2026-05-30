"""GPU-free config validation before real or dry-run training / cluster submit."""

from __future__ import annotations

import re

from zo_common import ExperimentConfig


def _dataset_looks_local(dataset: str) -> bool:
    return any(ch in dataset for ch in ("/", "\\", "*", "?", ".jsonl"))


def _unexpanded_env_refs(text: str) -> list[str]:
    refs = re.findall(r"\$\{([^}]+)\}", text)
    refs += re.findall(r"\$([A-Z_][A-Z0-9_]*)", text)
    import os

    return [r for r in refs if r not in os.environ]


def validate_experiment(cfg: ExperimentConfig, *, cluster: bool = False) -> None:
    """Raise ``ValueError`` with actionable messages when the config cannot run."""
    errors: list[str] = []

    if cfg.dataset and _dataset_looks_local(cfg.dataset):
        from zo_train.data import _resolve_local_files

        try:
            files = _resolve_local_files(cfg.dataset)
        except FileNotFoundError as exc:
            errors.append(str(exc))
        else:
            if not files:
                errors.append(
                    f"dataset {cfg.dataset!r}: no local files found. "
                    "Run `uv run python -m zo_train.datagen --build` first."
                )

    for label, value in (("model", cfg.model), ("output_dir", cfg.output_dir or "")):
        if not value:
            continue
        missing = _unexpanded_env_refs(value)
        if missing:
            errors.append(
                f"{label} {value!r} has unexpanded env var(s): {', '.join(missing)}. Set them in .env."
            )

    if cluster:
        from zo_train.cluster._slurm import is_hf_repo_id

        if is_hf_repo_id(cfg.model):
            errors.append(
                f"model {cfg.model!r} is a HuggingFace hub id; cluster jobs run with HF offline. "
                "Pre-stage the weights on a login node and point `model` at the local path "
                "(see ZO_BASE_MODEL_DIR / ZO_SFT_CHECKPOINT_DIR in .env.example)."
            )

    if errors:
        raise ValueError("Config preflight failed:\n" + "\n".join(f"  • {e}" for e in errors))


def checkpoint_kwargs(cfg: ExperimentConfig) -> dict[str, object]:
    """Periodic checkpointing — override with ``extra.save_steps: 0`` to disable."""
    save_steps = int(cfg.extra.get("save_steps", 50))
    if save_steps <= 0:
        return {"save_strategy": "no"}
    return {
        "save_strategy": "steps",
        "save_steps": save_steps,
        "save_total_limit": int(cfg.extra.get("save_total_limit", 3)),
    }

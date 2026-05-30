from __future__ import annotations

import glob
from pathlib import Path

from zo_common import ExperimentConfig


def _resolve_local_files(dataset: str) -> list[str]:
    """Resolve ``cfg.dataset`` to a list of local JSONL files, or ``[]`` if it isn't local.

    Accepts (in priority order):
      * a comma-separated list of paths — e.g. a LOFO fold = the two train families'
        ``*_sft_lm.jsonl`` (``a.jsonl,b.jsonl``);
      * a glob pattern — e.g. ``data/generated/*_sft_lm.jsonl``;
      * a single existing path.

    Returns the matched files (deduped, order-preserving). An empty list signals "not local"
    so the caller can fall back to treating ``cfg.dataset`` as a HuggingFace dataset id.
    """
    parts = [p.strip() for p in dataset.split(",") if p.strip()]
    files: list[str] = []
    for part in parts:
        if any(ch in part for ch in "*?[") or part.endswith("]"):
            files.extend(sorted(glob.glob(part)))
        elif Path(part).exists():
            files.append(part)
        else:
            # A comma-joined spec with a missing member is a config error, not an HF id.
            # A lone missing path falls through to the HF branch.
            if len(parts) > 1:
                raise FileNotFoundError(f"dataset path {part!r} (from {dataset!r}) does not exist")
    # Dedupe while preserving first-seen order (globs can overlap across patterns).
    return list(dict.fromkeys(files))


def load_sft_dataset(cfg: ExperimentConfig):
    """One or more local .jsonl files, a HuggingFace dataset id, or a built-in toy set.

    ``cfg.dataset`` may be a comma-separated list of paths or a glob — both load as a single
    concatenated ``train`` split. This is how a leave-one-family-out (LOFO) fold is expressed:
    the two train families' ``*_sft_lm.jsonl`` files joined with a comma.
    """
    from datasets import load_dataset

    if not cfg.dataset:
        return _toy_sft()
    files = _resolve_local_files(cfg.dataset)
    if files:
        return load_dataset("json", data_files=files, split="train")
    return load_dataset(cfg.dataset, split=cfg.dataset_split)


def load_prompt_dataset(cfg: ExperimentConfig):
    """For GRPO: needs a `prompt` column. Falls back to a toy prompt set.

    Like ``load_sft_dataset``, ``cfg.dataset`` may be a comma-separated list of paths or a glob,
    all loaded as a single concatenated ``train`` split — so an all-families GRPO run is the three
    families' ``*_eval_nextstep.jsonl`` / ``*_sft_completion.jsonl`` joined with a comma (no need to
    pre-concatenate a combined file). A single path or a HuggingFace id still work.
    """
    from datasets import load_dataset

    if not cfg.dataset:
        return _toy_prompts()
    files = _resolve_local_files(cfg.dataset)
    if files:
        return load_dataset("json", data_files=files, split="train")
    return load_dataset(cfg.dataset, split=cfg.dataset_split)


def _toy_sft():
    from datasets import Dataset

    rows = [{"text": f"### Question:\nWhat is 2 + {i}?\n\n### Answer:\n{2 + i}"} for i in range(64)]
    return Dataset.from_list(rows)


def _toy_prompts():
    from datasets import Dataset

    return Dataset.from_list([{"prompt": f"Write the number {i} in words."} for i in range(64)])

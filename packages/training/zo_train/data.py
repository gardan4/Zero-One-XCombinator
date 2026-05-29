from __future__ import annotations

from pathlib import Path

from zo_common import ExperimentConfig


def load_sft_dataset(cfg: ExperimentConfig):
    """Local .jsonl, a HuggingFace dataset id, or a tiny built-in toy set."""
    from datasets import load_dataset

    if not cfg.dataset:
        return _toy_sft()
    if Path(cfg.dataset).exists():
        return load_dataset("json", data_files=cfg.dataset, split="train")
    return load_dataset(cfg.dataset, split=cfg.dataset_split)


def load_prompt_dataset(cfg: ExperimentConfig):
    """For GRPO: needs a `prompt` column. Falls back to a toy prompt set."""
    from datasets import load_dataset

    if not cfg.dataset:
        return _toy_prompts()
    if Path(cfg.dataset).exists():
        return load_dataset("json", data_files=cfg.dataset, split="train")
    return load_dataset(cfg.dataset, split=cfg.dataset_split)


def _toy_sft():
    from datasets import Dataset

    rows = [{"text": f"### Question:\nWhat is 2 + {i}?\n\n### Answer:\n{2 + i}"} for i in range(64)]
    return Dataset.from_list(rows)


def _toy_prompts():
    from datasets import Dataset

    return Dataset.from_list([{"prompt": f"Write the number {i} in words."} for i in range(64)])

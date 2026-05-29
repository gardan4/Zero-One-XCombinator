from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

RunKind = Literal["sft", "dpo", "grpo", "eval", "agent"]


class ExperimentConfig(BaseModel):
    """A single experiment, serialized to/from YAML in packages/*/configs.

    `extra` holds recipe-specific knobs so configs stay forward-compatible without
    schema churn mid-hackathon.
    """

    name: str
    kind: RunKind = "sft"
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset: str | None = None
    dataset_split: str = "train"
    text_field: str = "text"

    # optimization
    learning_rate: float = 2.0e-5
    epochs: float = 1.0
    batch_size: int = 4
    grad_accum: int = 8
    max_seq_len: int = 2048
    warmup_ratio: float = 0.03

    # PEFT / LoRA
    lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    bf16: bool = True
    seed: int = 42
    output_dir: str | None = None

    extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.model_dump(), sort_keys=False))

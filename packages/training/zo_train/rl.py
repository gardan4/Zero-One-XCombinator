"""GRPO / RL recipe — **RLVR reasoning spike** on top of a Stream-1 SFT checkpoint.

Our track (Industrial AI / Infineon) is small-vocab *sequence modeling* of fab steps. This
loop runs **GRPO with a verifiable reward**: the model emits a process sequence, and
``zo_train.rewards.reward_validate`` scores it with the free, perfect ``validate_sequence``
verifier (dense ``1 - n_viol/n``, plus anti-degenerate guards). The reward IS the product
here — see ``rewards.py``. Pick the reward(s) via ``cfg.extra["reward"]`` (``"validate"`` or
``"validate+format"``). ``max_completion_length`` defaults to **64** (our completions are
short — a handful of pipe-separated steps; the trl default of 256 just wastes compute).

This is an abandonable spike; the SFT spine is the must-ship floor. See
`.claude/knowledge/track-industrial-ai.md`.
"""

from __future__ import annotations

import inspect
import os

from zo_common import ExperimentConfig, append_metric, run_dir, update_run
from zo_common.wandb_runs import finish_run, init_run, log_metrics, wandb_enabled
from zo_common.wandb_schema import merge_tags, pytest_auto_tags


def _report_to() -> list[str]:
    if os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE") != "disabled":
        return ["wandb"]
    return []


def _supported_kwargs(fn, kwargs: dict[str, object]) -> dict[str, object]:  # noqa: ANN001
    params = inspect.signature(fn).parameters
    return {key: value for key, value in kwargs.items() if key in params}


def run_grpo(cfg: ExperimentConfig, run_id: str, dry_run: bool = False) -> None:
    if dry_run:
        from zo_train.sim import simulate_training

        simulate_training(run_id, cfg=cfg)
        return

    try:
        from peft import LoraConfig
        from transformers import TrainerCallback
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        raise RuntimeError(
            "ML deps missing. Run `just gpu-sync` on a cluster node, or pass --dry-run."
        ) from e

    from zo_common.registry import get_run

    from zo_train.data import load_prompt_dataset
    from zo_train.preflight import checkpoint_kwargs
    from zo_train.rewards import select_rewards

    def _as_list(value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [t.strip() for t in value.split(",") if t.strip()]
        if isinstance(value, list):
            return [str(t).strip() for t in value if str(t).strip()]
        return [str(value)]

    update_run(run_id, status="running")
    out_dir = cfg.output_dir or str(run_dir(run_id) / "artifacts")
    dataset = load_prompt_dataset(cfg)

    meta = get_run(run_id)
    train_tags = merge_tags(_as_list(cfg.extra.get("tags")), meta.tags if meta else [], extra=pytest_auto_tags())
    if wandb_enabled():
        init_run(run_id, "train", tags=train_tags, config=cfg.model_dump(), group=cfg.name)

    reward_funcs = select_rewards(str(cfg.extra.get("reward", "validate")))

    class _RegistryCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
            # Rank-0-only registry writes (GRPO stays single-GPU today, but keep it multi-GPU safe).
            if not logs or not state.is_world_process_zero:
                return
            scalars = {k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}
            if scalars:
                append_metric(run_id, step=int(state.global_step), **scalars)
                log_metrics(scalars, step=int(state.global_step), prefix="train")

    peft_cfg = (
        LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, task_type="CAUSAL_LM")
        if cfg.lora
        else None
    )

    config_kwargs = dict(
        output_dir=out_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_generations=int(cfg.extra.get("num_generations", 8)),
        max_completion_length=int(cfg.extra.get("max_completion_length", 64)),
        num_train_epochs=cfg.epochs,
        logging_steps=1,
        bf16=cfg.bf16,
        max_steps=int(cfg.extra.get("max_steps", -1)),
        report_to=_report_to(),
        seed=cfg.seed,
        **checkpoint_kwargs(cfg),
    )
    # trl versions differ on the prompt-length kwarg name.
    for key in ("max_prompt_length", "max_seq_length", "max_length"):
        if key in inspect.signature(GRPOConfig.__init__).parameters:
            config_kwargs[key] = cfg.max_seq_len
            break

    args = GRPOConfig(**_supported_kwargs(GRPOConfig.__init__, config_kwargs))

    trainer = GRPOTrainer(
        model=cfg.model,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=dataset,
        peft_config=peft_cfg,
        callbacks=[_RegistryCallback()],
    )
    trainer.train()
    trainer.save_model(out_dir)
    if wandb_enabled():
        finish_run(exit_code=0)
    update_run(run_id, status="completed", metrics={"output_dir": out_dir})

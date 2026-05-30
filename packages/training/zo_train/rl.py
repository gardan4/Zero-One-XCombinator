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

import os

from zo_common import ExperimentConfig, append_metric, run_dir, update_run


def _report_to() -> list[str]:
    return ["wandb"] if os.environ.get("WANDB_API_KEY") else []


def run_grpo(cfg: ExperimentConfig, run_id: str, dry_run: bool = False) -> None:
    if dry_run:
        from zo_train.sim import simulate_training

        simulate_training(run_id)
        return

    try:
        from peft import LoraConfig
        from transformers import TrainerCallback
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        raise RuntimeError(
            "ML deps missing. Run `just gpu-sync` on a cluster node, or pass --dry-run."
        ) from e

    from zo_train.data import load_prompt_dataset
    from zo_train.rewards import select_rewards

    update_run(run_id, status="running")
    out_dir = cfg.output_dir or str(run_dir(run_id) / "artifacts")
    dataset = load_prompt_dataset(cfg)

    # The reward IS the product (RLVR): a verifier-grounded score, not the old length stub.
    # ``reward`` selects the function(s): "validate" (dense validate_sequence reward with
    # anti-hack guards) or "validate+format" (also reward a clean <think>…</think> shape).
    reward_funcs = select_rewards(str(cfg.extra.get("reward", "validate")))

    class _RegistryCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
            if not logs:
                return
            scalars = {k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}
            if scalars:
                append_metric(run_id, step=int(state.global_step), **scalars)

    peft_cfg = (
        LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, task_type="CAUSAL_LM")
        if cfg.lora
        else None
    )

    args = GRPOConfig(
        output_dir=out_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_generations=int(cfg.extra.get("num_generations", 8)),
        # Our completions are short (a few pipe-separated steps) — 64 default, not trl's 256.
        max_completion_length=int(cfg.extra.get("max_completion_length", 64)),
        num_train_epochs=cfg.epochs,
        logging_steps=1,
        bf16=cfg.bf16,
        max_steps=int(cfg.extra.get("max_steps", -1)),
        report_to=_report_to(),
        seed=cfg.seed,
        save_strategy="no",
    )

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
    update_run(run_id, status="completed", metrics={"output_dir": out_dir})

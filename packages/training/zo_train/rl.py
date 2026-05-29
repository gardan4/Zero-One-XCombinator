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

    update_run(run_id, status="running")
    out_dir = cfg.output_dir or str(run_dir(run_id) / "artifacts")
    dataset = load_prompt_dataset(cfg)

    # TODO(track): replace this toy reward with one that reflects YOUR task — a verifier,
    # unit tests, a judge model, a format/JSON checker, tool-call success, etc. The reward
    # IS the product here; this length-targeting stub just proves the loop runs.
    target_len = int(cfg.extra.get("target_len", 200))

    def reward_len(completions, **kwargs):
        return [-abs(len(c) - target_len) / target_len for c in completions]

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
        max_completion_length=int(cfg.extra.get("max_completion_length", 256)),
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
        reward_funcs=[reward_len],
        args=args,
        train_dataset=dataset,
        peft_config=peft_cfg,
        callbacks=[_RegistryCallback()],
    )
    trainer.train()
    trainer.save_model(out_dir)
    update_run(run_id, status="completed", metrics={"output_dir": out_dir})

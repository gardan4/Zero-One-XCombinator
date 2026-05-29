from __future__ import annotations

import os

from zo_common import ExperimentConfig, append_metric, run_dir, update_run


def _report_to() -> list[str]:
    return ["wandb"] if os.environ.get("WANDB_API_KEY") else []


def run_sft(cfg: ExperimentConfig, run_id: str, dry_run: bool = False) -> None:
    if dry_run:
        from zo_train.sim import simulate_training

        simulate_training(run_id)
        return

    try:
        from peft import LoraConfig
        from transformers import AutoTokenizer, TrainerCallback
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:  # deps live on the cluster, not laptops
        raise RuntimeError(
            "ML deps missing. Run `just gpu-sync` on a cluster node, or pass --dry-run to "
            "exercise the pipeline locally without torch."
        ) from e

    from zo_train.data import load_sft_dataset

    update_run(run_id, status="running")
    out_dir = cfg.output_dir or str(run_dir(run_id) / "artifacts")
    dataset = load_sft_dataset(cfg)

    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    class _RegistryCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
            if not logs:
                return
            scalars = {k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}
            if scalars:
                append_metric(run_id, step=int(state.global_step), **scalars)

    peft_cfg = (
        LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            task_type="CAUSAL_LM",
        )
        if cfg.lora
        else None
    )

    sft_kwargs = {}
    if cfg.text_field and cfg.text_field != "messages":
        sft_kwargs["dataset_text_field"] = cfg.text_field

    args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        max_seq_length=cfg.max_seq_len,
        bf16=cfg.bf16,
        logging_steps=1,
        max_steps=int(cfg.extra.get("max_steps", -1)),
        report_to=_report_to(),
        seed=cfg.seed,
        save_strategy="no",
        **sft_kwargs,
    )

    trainer = SFTTrainer(
        model=cfg.model,
        args=args,
        train_dataset=dataset,
        peft_config=peft_cfg,
        callbacks=[_RegistryCallback()],
    )
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    update_run(run_id, status="completed", metrics={"output_dir": out_dir})

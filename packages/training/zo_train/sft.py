from __future__ import annotations

import inspect
import os
from pathlib import Path

from zo_common import ExperimentConfig, append_metric, run_dir, update_run


def _report_to() -> list[str]:
    if os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE") != "disabled":
        return ["wandb"]
    return []


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _supported_kwargs(fn, kwargs: dict[str, object]) -> dict[str, object]:  # noqa: ANN001
    params = inspect.signature(fn).parameters
    return {key: value for key, value in kwargs.items() if key in params}


def run_sft(cfg: ExperimentConfig, run_id: str, dry_run: bool = False) -> None:
    if dry_run:
        from zo_train.sim import simulate_training

        simulate_training(run_id)
        return

    try:
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
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

    local_files_only = _truthy(cfg.extra.get("local_files_only", False)) or Path(cfg.model).is_absolute()
    tok = AutoTokenizer.from_pretrained(cfg.model, local_files_only=local_files_only)
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

    config_kwargs = dict(
        output_dir=out_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        max_seq_length=cfg.max_seq_len,
        max_length=cfg.max_seq_len,
        bf16=cfg.bf16,
        logging_steps=1,
        max_steps=int(cfg.extra.get("max_steps", -1)),
        report_to=_report_to(),
        seed=cfg.seed,
        save_strategy="no",
    )
    config_kwargs.update(sft_kwargs)
    args = SFTConfig(**_supported_kwargs(SFTConfig.__init__, config_kwargs))

    model = (
        AutoModelForCausalLM.from_pretrained(cfg.model, local_files_only=True)
        if local_files_only
        else cfg.model
    )
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=peft_cfg,
        callbacks=[_RegistryCallback()],
        processing_class=tok,
        tokenizer=tok,
    )

    trainer = SFTTrainer(**_supported_kwargs(SFTTrainer.__init__, trainer_kwargs))
    hub_base = str(cfg.extra.get("base_model_hub_id") or "Qwen/Qwen2.5-0.5B-Instruct")

    def _wandb_fail(exc: BaseException) -> None:
        if os.environ.get("WANDB_MODE") == "disabled":
            return
        try:
            import wandb

            if wandb.run is not None:
                wandb.log({"train/status": "failed", "train/error": str(exc)[:2000]})
                wandb.alert(title="Training failed", text=str(exc)[:500], level="ERROR")
                wandb.finish(exit_code=1)
        except Exception:
            pass

    try:
        trainer.train()
    except Exception as exc:
        _wandb_fail(exc)
        update_run(run_id, status="failed", metrics={"error": str(exc)})
        raise

    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    readme = Path(out_dir) / "README.md"
    if readme.exists():
        text = readme.read_text()
        if cfg.model.startswith("/"):
            text = text.replace(cfg.model, hub_base)
        readme.write_text(text)

    metrics = {"output_dir": out_dir}
    hub_model_id = cfg.extra.get("hub_model_id")
    if hub_model_id and _truthy(cfg.extra.get("push_to_hub", False)):
        trainer.model.push_to_hub(
            str(hub_model_id),
            private=bool(cfg.extra.get("hub_private", True)),
            token=os.environ.get("HF_TOKEN"),
        )
        tok.push_to_hub(
            str(hub_model_id),
            private=bool(cfg.extra.get("hub_private", True)),
            token=os.environ.get("HF_TOKEN"),
        )
        metrics["hub_model_id"] = hub_model_id

    update_run(run_id, status="completed", metrics=metrics)

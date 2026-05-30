from __future__ import annotations

import inspect
import os
from pathlib import Path

from zo_common import ExperimentConfig, append_metric, run_dir, update_run
from zo_common.wandb_runs import finish_run, init_run, log_metrics, wandb_enabled
from zo_common.wandb_schema import merge_tags, pytest_auto_tags


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

        simulate_training(run_id, cfg=cfg)
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

    from zo_common.registry import get_run

    from zo_train.data import load_sft_dataset
    from zo_train.preflight import checkpoint_kwargs

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
    dataset = load_sft_dataset(cfg)

    meta = get_run(run_id)
    train_tags = merge_tags(
        _as_list(cfg.extra.get("tags")),
        meta.tags if meta else [],
        extra=pytest_auto_tags(),
    )
    if wandb_enabled():
        init_run(
            run_id,
            "train",
            tags=train_tags,
            config={**cfg.model_dump(), "git_sha": meta.git_sha if meta else None},
            group=cfg.name,
        )

    local_files_only = (
        _truthy(cfg.extra.get("local_files_only", False)) or Path(cfg.model).is_absolute()
    )
    tok = AutoTokenizer.from_pretrained(cfg.model, local_files_only=local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    class _RegistryCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
            # Under multi-GPU (FSDP) only rank 0 writes the shared run store — otherwise the node's
            # processes race on metrics.jsonl. Single-process runs are always world-process-zero.
            if not logs or not state.is_world_process_zero:
                return
            scalars = {k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}
            if scalars:
                append_metric(run_id, step=int(state.global_step), **scalars)
                log_metrics(scalars, step=int(state.global_step), prefix="train")

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
        gradient_checkpointing=_truthy(cfg.extra.get("gradient_checkpointing", False)),
        logging_steps=1,
        max_steps=int(cfg.extra.get("max_steps", -1)),
        report_to=_report_to(),
        seed=cfg.seed,
        **checkpoint_kwargs(cfg),
    )
    config_kwargs.update(sft_kwargs)
    # Instruction tuning: prompt/completion dataset (no text_field) → loss on the answer only.
    if _truthy(cfg.extra.get("completion_only_loss", False)):
        config_kwargs["completion_only_loss"] = True
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

    # FSDP save is collective (all ranks join the FULL_STATE_DICT gather); only rank 0 then writes
    # the tokenizer / metadata / hub upload, so the node's processes don't race on the same files.
    trainer.save_model(out_dir)

    metrics = {"output_dir": out_dir}
    if trainer.is_world_process_zero():
        tok.save_pretrained(out_dir)

        hub_model_id = cfg.extra.get("hub_model_id")
        from zo_common.hub_metadata import write_hub_artifact_metadata

        write_hub_artifact_metadata(
            out_dir,
            run_id,
            cfg,
            hub_model_id=str(hub_model_id) if hub_model_id else None,
            notes=str(cfg.extra.get("hub_notes") or ""),
        )

        if hub_model_id and _truthy(cfg.extra.get("push_to_hub", False)):
            from huggingface_hub import HfApi

            api = HfApi(token=os.environ.get("HF_TOKEN"))
            repo_id = str(hub_model_id)
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=bool(cfg.extra.get("hub_private", True)),
                exist_ok=True,
            )
            api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=out_dir,
                commit_message=f"Training run {run_id}",
            )
            metrics["hub_model_id"] = hub_model_id
            from zo_common.wandb_runs import log_hf_to_training_run

            log_hf_to_training_run(
                run_id,
                str(hub_model_id),
                tags=train_tags,
                config=cfg.model_dump(),
            )

    if wandb_enabled():
        finish_run(exit_code=0)
    update_run(run_id, status="completed", metrics=metrics)

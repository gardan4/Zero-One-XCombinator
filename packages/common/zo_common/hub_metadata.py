"""Write training provenance onto Hugging Face artifact folders.

Every checkpoint upload should include:

- ``training_manifest.json`` — machine-readable run id, tags, full training config
- ``README.md`` — HF model card (YAML frontmatter tags + human-readable hyperparams)

Look up a model on HF: open the repo → ``README.md`` or download ``training_manifest.json``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from zo_common.config import ExperimentConfig


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [t.strip() for t in value.split(",") if t.strip()]
    if isinstance(value, list):
        return [str(t).strip() for t in value if str(t).strip()]
    return [str(value)]


def collect_tags(cfg: ExperimentConfig | dict[str, Any], run_meta: dict | None = None) -> list[str]:
    """Merge tags from config ``extra.tags``, ``extra.hub_tags``, and run registry."""
    if isinstance(cfg, ExperimentConfig):
        extra = cfg.extra or {}
    else:
        extra = cfg.get("extra") or {}
    tags: list[str] = []
    for source in (_as_list(extra.get("tags")), _as_list(extra.get("hub_tags"))):
        for t in source:
            if t not in tags:
                tags.append(t)
    if run_meta:
        for t in run_meta.get("tags") or []:
            if t not in tags:
                tags.append(t)
    return tags


def build_training_manifest(
    *,
    run_id: str,
    cfg: ExperimentConfig | dict[str, Any],
    run_meta: dict | None = None,
    hub_model_id: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Structured metadata stored beside weights on Hugging Face."""
    if isinstance(cfg, ExperimentConfig):
        training = cfg.model_dump()
        name = cfg.name
    else:
        training = dict(cfg)
        name = str(training.get("name", run_id))

    meta = run_meta or {}
    extra = training.get("extra") or {}
    manifest_notes = notes or meta.get("notes") or extra.get("hub_notes") or ""

    return {
        "schema": "zero-one-philyr/training-manifest/v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "run_name": meta.get("name") or name,
        "run_kind": meta.get("kind") or training.get("kind"),
        "run_status": meta.get("status"),
        "created_at": meta.get("created_at"),
        "git_branch": meta.get("git_branch"),
        "git_sha": meta.get("git_sha"),
        "cluster": meta.get("cluster"),
        "slurm_job_id": meta.get("slurm_job_id"),
        "hub_model_id": hub_model_id or extra.get("hub_model_id"),
        "tags": collect_tags(cfg, meta),
        "notes": manifest_notes,
        "training": training,
        "metrics_summary": meta.get("metrics") or {},
    }


def render_model_card(manifest: dict[str, Any]) -> str:
    """HF ``README.md`` with frontmatter tags + training table."""
    tags = manifest.get("tags") or []
    training = manifest.get("training") or {}
    extra = training.get("extra") or {}

    frontmatter = {
        "tags": tags,
        "license": extra.get("hub_license", "mit"),
        "library_name": "transformers",
        "base_model": extra.get("base_model_hub_id") or training.get("model"),
    }
    if manifest.get("hub_model_id"):
        frontmatter["model-index"] = [{"name": manifest["hub_model_id"]}]

    lines = ["---", yaml.safe_dump(frontmatter, sort_keys=False).strip(), "---", ""]

    title = manifest.get("hub_model_id") or manifest.get("run_name") or manifest.get("run_id")
    lines.append(f"# {title}")
    lines.append("")
    lines.append("Fine-tuned checkpoint from the **Zero One Philyr / Industrial AI (Infineon)** track.")
    lines.append("")

    lines.append("## Run identity")
    lines.append("")
    lines.append(f"- **Run ID:** `{manifest.get('run_id', '?')}`")
    if manifest.get("git_sha"):
        lines.append(f"- **Git SHA:** `{manifest['git_sha']}` ({manifest.get('git_branch', '?')})")
    if manifest.get("slurm_job_id"):
        lines.append(f"- **SLURM job:** `{manifest['slurm_job_id']}`")
    if tags:
        lines.append(f"- **Tags:** {', '.join(f'`{t}`' for t in tags)}")
    if manifest.get("notes"):
        lines.append(f"- **Notes:** {manifest['notes']}")
    lines.append("")
    lines.append("Full machine-readable config: [`training_manifest.json`](./training_manifest.json)")
    lines.append("")

    lines.append("## Training hyperparameters")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    rows = [
        ("name", training.get("name")),
        ("kind", training.get("kind")),
        ("base model (config)", training.get("model")),
        ("dataset", training.get("dataset")),
        ("learning_rate", training.get("learning_rate")),
        ("epochs", training.get("epochs")),
        ("batch_size", training.get("batch_size")),
        ("grad_accum", training.get("grad_accum")),
        ("max_seq_len", training.get("max_seq_len")),
        ("warmup_ratio", training.get("warmup_ratio")),
        ("lora", training.get("lora")),
        ("lora_r", training.get("lora_r") if training.get("lora") else "—"),
        ("bf16", training.get("bf16")),
        ("seed", training.get("seed")),
    ]
    for key, val in rows:
        if val is not None:
            lines.append(f"| {key} | `{val}` |")
    lines.append("")

    if extra:
        lines.append("## Extra config (`extra` block)")
        lines.append("")
        lines.append("```yaml")
        lines.append(yaml.safe_dump(extra, sort_keys=False).strip())
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def write_hub_artifact_metadata(
    artifact_dir: str | Path,
    run_id: str,
    cfg: ExperimentConfig | dict[str, Any] | None = None,
    *,
    hub_model_id: str | None = None,
    notes: str | None = None,
    config_path: str | Path | None = None,
) -> Path:
    """Write ``training_manifest.json`` + ``README.md`` into an artifact folder."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    from zo_common.registry import get_run, run_dir

    meta = get_run(run_id)
    run_meta = meta.model_dump() if meta else None

    if cfg is None:
        cfg_path = Path(config_path) if config_path else (run_dir(run_id) / "config.yaml")
        if cfg_path.exists():
            cfg = ExperimentConfig.from_yaml(cfg_path)
        elif meta and meta.config:
            cfg = meta.config
        else:
            cfg = {"name": run_id, "extra": {}}

    manifest = build_training_manifest(
        run_id=run_id,
        cfg=cfg,
        run_meta=run_meta,
        hub_model_id=hub_model_id,
        notes=notes,
    )
    manifest_path = artifact_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (artifact_dir / "README.md").write_text(render_model_card(manifest), encoding="utf-8")
    return manifest_path

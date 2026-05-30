"""Publish eval/inference results to W&B (no-op when unavailable)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zo_common.wandb_runs import (
    finish_run,
    init_run,
    log_artifact,
    log_metrics,
    pytest_auto_tags,
    wandb_enabled,
)
from zo_common.wandb_schema import (
    ARTIFACT_EVAL_RESULTS,
    TAG_DEBUG,
    TAG_PROXY_ONLY,
    TAG_REAL_RUN,
    merge_tags,
    validate_run_tags,
)


def apply_eval_tags(
    tags: list[str],
    *,
    has_gold: bool,
    source: str | None = None,
) -> list[str]:
    """Auto-apply safety tags for eval/inference runs."""
    extra: list[str] = list(pytest_auto_tags())
    ts = set(tags)
    if not has_gold and TAG_PROXY_ONLY not in ts:
        extra.append(TAG_PROXY_ONLY)
    if source in ("dashboard-inference", "dashboard") or "source:dashboard-inference" in ts:
        if TAG_REAL_RUN not in ts:
            extra.append(TAG_DEBUG)
    return merge_tags(tags, extra=extra)


def publish_eval_run(
    run_id: str,
    results_dir: str | Path,
    metrics: dict[str, Any],
    *,
    tags: list[str],
    config: dict[str, Any] | None = None,
    job_type: str = "eval",
    artifact_aliases: list[str] | None = None,
    use_wandb: bool = True,
) -> None:
    """Log eval metrics + result artifacts to W&B."""
    if not use_wandb or not wandb_enabled():
        return
    tag_list = list(tags)
    for w in validate_run_tags(tag_list, job_type=job_type):
        import warnings

        warnings.warn(f"W&B eval run {run_id}: {w}", stacklevel=2)

    init_run(
        run_id,
        job_type,
        tags=tag_list,
        config=config or {},
    )
    if metrics:
        log_metrics(metrics, step=0, use_eval_map=True)
    aliases = artifact_aliases or ["latest"]
    ver_tag = next((t.split(":", 1)[1] for t in tag_list if t.startswith("version:")), None)
    if ver_tag and f"version:{ver_tag}" not in aliases:
        aliases.append(f"version:{ver_tag}")
    log_artifact(
        results_dir,
        name=f"eval-{run_id}",
        artifact_type=ARTIFACT_EVAL_RESULTS,
        aliases=aliases,
        metadata={"run_id": run_id, "tags": tag_list},
    )
    finish_run(exit_code=0)

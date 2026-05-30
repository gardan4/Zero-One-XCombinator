"""Shared SLURM template rendering for Leonardo cluster jobs."""

from __future__ import annotations

import os
import re
from pathlib import Path

from jinja2 import Template
from zo_common.env import load_dotenv

from zo_train.cluster._platform import local_model_cache_dir, posix_path

_TEMPLATES = Path(__file__).parent / "slurm"


def cluster_env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


def ensure_cluster_env() -> None:
    load_dotenv()


def slurm_context(**overrides: object) -> dict:
    """Build the common SLURM template context from ``.env``."""
    from zo_train.cluster._remote import cluster_repo_dir, expand_cluster_path, load_dotenv

    load_dotenv()
    repo_dir = cluster_repo_dir()
    # GPU count: an explicit per-run override (e.g. a 4-GPU FSDP job via extra.gpus_per_node) wins
    # over the .env default, and mem/cpus scale with it unless pinned via ZO_SLURM_MEM/ZO_SLURM_CPUS.
    gpus = int(overrides.pop("gpus_per_node", None) or cluster_env("ZO_SLURM_GPUS_PER_NODE", "1"))
    mem = cluster_env("ZO_SLURM_MEM") or f"{120 * gpus}GB"
    cpus = int(cluster_env("ZO_SLURM_CPUS") or str(8 * gpus))
    experiments = expand_cluster_path(
        cluster_env("ZO_CLUSTER_EXPERIMENTS_DIR", f"{repo_dir}/experiments")
        or f"{repo_dir}/experiments"
    )
    ctx: dict = dict(
        job_name="zo-job",
        account=cluster_env("ZO_SLURM_ACCOUNT", ""),
        partition=cluster_env("ZO_SLURM_PARTITION", "boost_usr_prod"),
        reservation=cluster_env("ZO_SLURM_RESERVATION", ""),
        qos=cluster_env("ZO_SLURM_QOS", ""),
        nodes=int(cluster_env("ZO_SLURM_NODES", "1")),
        gpus_per_node=gpus,
        mem=mem,
        cpus=cpus,
        time=cluster_env("ZO_SLURM_TIME", "02:00:00"),
        repo_dir=repo_dir,
        experiments_dir=experiments,
        hf_home=expand_cluster_path(cluster_env("HF_HOME") or f"{repo_dir}/hf_cache"),
        proxy=cluster_env("ZO_CLUSTER_PROXY", ""),
        accelerate_config=cluster_env(
            "ZO_ACCELERATE_CONFIG",
            "packages/training/zo_train/cluster/accelerate/fsdp_a100.yaml",
        ),
    )
    ctx.update(overrides)
    return ctx


def render_template(template_name: str, **ctx: object) -> str:
    tpl = (_TEMPLATES / template_name).read_text()
    return Template(tpl).render(**ctx)


def is_hf_repo_id(model: str) -> bool:
    """True for ``org/name`` Hugging Face ids (not local / env-expanded paths)."""
    if not model or model.startswith(("/", ".", "$", "~")):
        return False
    if "\\" in model or Path(model).is_absolute():
        return False
    return bool(re.fullmatch(r"[^/]+/[^/]+", model))


def default_model_cache_dir() -> str:
    explicit = cluster_env("ZO_MODEL_CACHE_DIR")
    if explicit:
        return posix_path(os.path.expandvars(explicit))
    if os.environ.get("SCRATCH"):
        return posix_path(Path(os.environ["SCRATCH"]) / "zo-models")
    return posix_path(local_model_cache_dir())


def staged_model_path(hf_repo: str) -> str:
    short = hf_repo.split("/")[-1]
    scratch = cluster_env("ZO_CLUSTER_SCRATCH") or os.environ.get("SCRATCH")
    if scratch:
        return posix_path(Path(os.path.expandvars(scratch)) / "hf-local" / short)
    slug = hf_repo.replace("/", "--")
    return posix_path(Path(default_model_cache_dir()) / slug)


def resolve_infer_model(model: str | None = None) -> str:
    """Return a path or HF id suitable for ``--model`` on the compute node."""
    raw = (
        model or cluster_env("ZO_INFER_MODEL") or cluster_env("ZO_INFER_MODEL_PATH") or ""
    ).strip()
    if not raw:
        raise ValueError(
            "Set ZO_INFER_MODEL (HF repo id, e.g. XCombinator/sft-fab-all) or ZO_INFER_MODEL_PATH "
            "(local checkpoint directory on shared storage)."
        )
    if is_hf_repo_id(raw):
        # Explicit --model HF id always maps to that repo's staged dir (not a stale ZO_INFER_MODEL_PATH).
        if model and is_hf_repo_id(model.strip()):
            return staged_model_path(model.strip())
        staged = cluster_env("ZO_INFER_MODEL_PATH")
        if staged:
            return posix_path(os.path.expandvars(staged))
        return staged_model_path(raw)
    return posix_path(os.path.expandvars(raw))

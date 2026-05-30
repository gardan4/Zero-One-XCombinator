# Zero One Philyr

Monorepo for the **Zero One Hack** (Vienna, May 29–31). Built for one thing: spin up
**finetuning / RL / eval / agent** experiments fast, run many in parallel, and watch
them from one dashboard — with the Leonardo A100 cluster doing the heavy lifting.

> New here? Read [`CLAUDE.md`](CLAUDE.md) and [`.claude/knowledge/INDEX.md`](.claude/knowledge/INDEX.md) first.

## Layout

```
apps/
  backend/      FastAPI control plane — reads the run registry, exposes runs/metrics
  frontend/     Next.js dashboard — list runs, inspect metrics & eval results
packages/
  common/       zo_common — experiment config + file-based run registry (the shared contract)
  training/     zo_train  — SFT + GRPO/RL recipes, configs, and SLURM cluster submission
  eval/         zo_eval   — model eval harness (tasks → metrics → run registry)
  agent/        zo_agent  — agent rollout harness (tools, scenarios, task success)
experiments/    per-run outputs (meta.json, metrics.jsonl, artifacts/) — gitignored
.claude/        team Claude config: settings, slash commands, subagents, knowledge base
scripts/        wt.sh (worktree-per-experiment), dev.sh
```

## Quickstart

```bash
mise install          # python 3.12, node 20, uv, just  (or install these yourself)
cp .env.example .env  # fill in cluster user, HF_TOKEN, wandb (all optional to start)
just setup            # uv sync (light, no GPU deps) + npm install
just dev              # backend :8000 + frontend :3000
```

Heavy ML deps (torch/trl/transformers/vllm) are an optional extra and install **on the
cluster**, not your laptop:

```bash
just gpu-sync         # uv sync --extra gpu   (run on a Leonardo node)
```

`requirements.txt` (repo root) is the **light** base set for `pip` users / submission; the exact
GPU pin lives in `uv.lock` and installs via the extra above. Regenerate:
`uv export --no-hashes --no-emit-workspace -o requirements.txt`.

## The core loop

```bash
just train packages/training/configs/sft_smoke.yaml   # local smoke / SLURM job
just submit packages/training/configs/sft_qwen.yaml   # render + sbatch on Leonardo
just runs                                             # list every run, newest first
just eval packages/eval/tasks/example.yaml $MODEL     # score a model
just agent packages/agent/scenarios/example.yaml $MODEL  # measure agentic task success
```

Every run — training, eval, or agent — writes through `zo_common` into `experiments/`,
which is exactly what the backend serves and the dashboard shows.

## Working in parallel (4 people, many experiments)

```bash
just wt grpo-reward-v2   # new git worktree + branch ../zo-grpo-reward-v2, isolated experiments
```

Each worktree is an independent checkout, so teammates (or parallel Claude sessions) can train
and eval different ideas at once without stepping on each other. See [`CLAUDE.md`](CLAUDE.md).

## Leonardo smoke finetune (Windows / macOS / Linux)

End-to-end: sync repo → prestage on login → SLURM LoRA train → optional HF upload.

```bash
cp .env.example .env   # ZO_CLUSTER_USER, HF_TOKEN, ZO_CLUSTER_* paths
uv run python scripts/leonardo_smoke.py --dry-run   # render sbatch locally
uv run python scripts/leonardo_smoke.py             # full pipeline over SSH
uv run python scripts/leonardo_smoke.py --wait-upload
```

Same via `uv run zo-cluster leonardo-smoke`. Legacy wrappers: `scripts/leonardo_smoke_hf.ps1` (Windows),
`scripts/leonardo_smoke_hf.sh` (macOS/Linux with rsync). Config: `packages/training/configs/leonardo_smoke_hf.yaml`.

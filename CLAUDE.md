# Zero One Philyr

Hackathon monorepo for **finetuning / RL / agentifying** language models. The laptop stays
light; the **Leonardo (CINECA)** GPU cluster does the heavy lifting. four-person team, moving
fast, prototyping experiments **in parallel** with git worktrees.

> **Theme:** real model training — not prompt engineering. When in doubt, train or measure something.
>
> **Our track: Industrial AI (Infineon)** — learning & benchmarking *process logic* by sequence-modeling
> semiconductor fab steps (next-step prediction, sequence completion, anomaly detection over a ~120-token
> step vocabulary). This is **small-vocab sequence modeling, not chat-LLM SFT** — the scaffold's
> SFT/GRPO/OpenAI-eval pieces are a starting point, not a direct fit. See
> [`track-industrial-ai.md`](.claude/knowledge/track-industrial-ai.md) for the full spec and how the
> scaffold maps.

## Read this first: the knowledge base

Everything the team learns lives in [`.claude/knowledge/`](.claude/knowledge/). It is our shared
brain — your teammate's Claude session cannot see your context, only what's written down.

1. **At the start of a session**, skim [`.claude/knowledge/INDEX.md`](.claude/knowledge/INDEX.md).
2. **When you discover something non-obvious** (a cluster quirk, a `trl` API change, a dataset
   gotcha, a partition that actually works) — **append it**. Run `/log-learning` to do this in one
   step, or edit the relevant topic file and add a line to the INDEX log.
3. Keep entries short and concrete. A stale or vague note is worse than none.

Topic files: [stack](.claude/knowledge/stack.md) · [cluster](.claude/knowledge/cluster.md) ·
[**track-industrial-ai**](.claude/knowledge/track-industrial-ai.md) · [training](.claude/knowledge/training.md) ·
[eval](.claude/knowledge/eval.md) · [agents](.claude/knowledge/agents.md) ·
[hackathon](.claude/knowledge/hackathon.md)

## Layout

```
apps/
  backend/    FastAPI control plane — reads the run registry, launches jobs (zo_backend)
  frontend/   Next.js 16 + React 19 dashboard — lists runs, plots metrics
packages/
  common/     zo_common: run registry, config schema, paths, LLM client   → CLI: zo-runs
  training/   zo_train: SFT + GRPO via trl; cluster/ submits SLURM jobs    → CLI: zo-train, zo-cluster
  eval/       zo_eval: task-based eval harness against an OpenAI endpoint  → CLI: zo-eval
  agent/      zo_agent: tool-calling rollout + scenario harness            → CLI: zo-agent
experiments/  one dir per run (gitignored): meta.json, metrics.jsonl, config.yaml, logs/, artifacts/
data/         industrial-infineon/: vendored track data + grammar + generate_sequences.py (our track)
docs/         track briefing, Leonardo deck (Z10_compressed.pdf), submission/ (REPORT_TEMPLATE, SUBMISSION)
scripts/      wt.sh (worktree-per-experiment), dev.sh, setup.sh
.claude/      this folder: CLAUDE.md, knowledge base, slash commands, subagents
```

It's a **uv workspace**: one virtual root `pyproject.toml`, one lockfile, members under
`apps/backend` + `packages/*`. `apps/frontend` is a separate npm app. Tools are pinned with
**mise**; tasks run through **just**.

## Common commands

| Command | What it does |
|---|---|
| `just setup` | First-time: `uv sync` (light deps) + `npm install` in the frontend |
| `just gpu-sync` | Install the heavy ML stack (`torch/trl/transformers/vllm`) — **run on a GPU box / the cluster** |
| `just dev` | Backend + frontend together (Ctrl-C stops both) |
| `just train <config>` | Local SFT from a YAML config. Add `--dry-run` to skip torch entirely |
| `just grpo <config>` | Local GRPO / RL run |
| `just submit <config>` | Render + submit a training job to Leonardo SLURM over SSH |
| `just cluster-watch` | `squeue --me` on the cluster |
| `just eval <task> <model>` | Run an eval task against a served model |
| `just agent <scenario> <model>` | Run an agent scenario (tool-use rollout) |
| `just serve <model>` | Start a vLLM OpenAI-compatible server (default port 8001) |
| `just runs` | List all experiment runs |
| `just wt <name>` | New git worktree + branch for parallel work |
| `just lint` / `just fmt` / `just test` | ruff check / ruff format / pytest |

CLIs also run directly: `uv run zo-train sft -c <config> --dry-run`, `uv run zo-runs show <id>`, etc.

## The run registry — the shared contract

Every run is a directory: `experiments/<run_id>/` where `run_id = <YYYYMMDD_HHMMSS>_<kind>_<slug>`.

- `meta.json` — `RunMeta` (id, name, kind, status, git info, slurm_job_id, config, metrics summary)
- `metrics.jsonl` — one JSON object per logged step (append-only)
- `config.yaml` — the exact config the run used
- `logs/`, `artifacts/` — free-form

The flow is **one direction**: training/eval/agent code writes via
`zo_common.registry.append_metric(run_id, step=i, **metrics)` → the backend reads the files →
the frontend plots them. To surface a new metric anywhere, just append it; nothing else needs to
change. **`zo_common` is shared by every package** — coordinate before changing its schemas.

## Local-light / cluster-heavy

- Base `uv sync` installs only light deps (pydantic, typer, httpx, fastapi). All CLIs *import*
  fine on a laptop.
- Heavy deps (torch, trl, transformers, peft, accelerate, datasets, vllm, wandb) are the optional
  **`[gpu]` extra**. Run `uv sync --extra gpu` only where there's a GPU.
- Training code **lazy-imports** torch/trl inside functions. That's deliberate: `--dry-run`
  simulates a training curve (decaying loss, rising reward) with no torch, so you can validate a
  config + the whole registry→backend→frontend path on your laptop before burning cluster time.

## Working in parallel (worktrees)

- `just wt <name>` creates `../Zero-One-Philyr-<name>` on a fresh branch. Each teammate or
  experiment gets an isolated checkout — train two things at once without stepping on each other.
- Runs are isolated per-worktree by default. To **share one run store** across worktrees (so the
  dashboard sees everything), set `ZO_EXPERIMENTS_DIR` to a common path in each `.env`. On the
  cluster, point it at shared scratch and run the backend on the login node.

## Working norms across sessions

- **Read the knowledge INDEX first; append learnings as you go.** Don't hoard them in your head.
- **`--dry-run` before you submit.** Validate the config and pipeline locally, then spend GPU time.
- **Never commit secrets.** `.env` is gitignored; only `.env.example` (placeholders) is tracked.
- **Never commit weights or experiment outputs.** `experiments/`, `*.safetensors`, checkpoints,
  `wandb/`, HF caches are all gitignored. Commit *configs and code*, not artifacts.
- **Small, focused commits.** We move fast, but `zo_common` is shared infrastructure — flag schema
  changes to your teammate.
- **Prefer editing existing files** over adding new ones; this scaffold already has a home for most
  things (a new eval = a YAML in `packages/eval/tasks/`, a new tool = a function in
  `zo_agent/tools.py`).

## Gotchas

- The local directory is `Zero One Philyr` (with spaces); the remote is `Zero-One-Philyr`. Spaces
  break some tooling — consider renaming the local dir to match.
- **`trl` API drifts between versions.** If `SFTConfig` / `GRPOConfig` kwargs error out, check the
  installed version against [`training.md`](.claude/knowledge/training.md) and note what you found.
- The frontend needs the backend running (`NEXT_PUBLIC_API_URL`, default `http://localhost:8000`).
  `just dev` starts both.
- Cluster host / partition / account / QOS in `.env.example` are **best guesses** — confirm with
  the organizers on-site and record the real values in [`cluster.md`](.claude/knowledge/cluster.md).

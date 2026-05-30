# Zero One Philyr

Hackathon monorepo for **finetuning / RL / agentifying** language models. The laptop stays
light; the **Leonardo (CINECA)** GPU cluster does the heavy lifting. Four-person team, moving
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
  backend/    FastAPI control plane — reads the run registry, serves runs/compare/inference (zo_backend)
packages/
  common/     zo_common: run registry, config schema, paths, W&B/HF helpers, LLM client → CLI: zo-runs
  training/   zo_train: SFT + GRPO via trl; data factory (grammar/datagen); cluster/ SLURM → CLI: zo-train, zo-cluster
  eval/       zo_eval: track predict→score→CSVs (the real harness) + legacy tasks harness → CLI: zo-eval, zo-track
  agent/      zo_agent: tool-calling rollout + scenario harness            → CLI: zo-agent
infineon-results-dashboard/  standalone Cloudflare Workers dashboard (:8787) — the submission UI (replaced apps/frontend)
data/         industrial-infineon/: vendored track data + grammar + generate_sequences.py; eval/ kickoff inputs
extras/       eval_local/ labeled proxy eval sets (+ _base, _sbatch); results/ promoted CSVs for the report
submissions/  per-team REPORT.md (submissions/XCombinator/)
tests/        top-level pytest suite (unit by default; `-m integration` for live HF)
experiments/  legacy in-repo scratch (default is ~/.cache/zo-experiments) — outputs gitignored
docs/         track briefing, Leonardo deck (Z10_compressed.pdf), submission/ + playbooks (eval-and-artifacts, leonardo-eval)
scripts/      dev.py, setup.py, leonardo_*.py, zo_cluster.py (no-uv judge path), wt.sh
.claude/      this folder: CLAUDE.md, knowledge base, slash commands, subagents
```

It's a **uv workspace**: one virtual root `pyproject.toml`, one lockfile, members under
`apps/backend` + `packages/*`. The dashboard (`infineon-results-dashboard/`) is a **standalone
Cloudflare Workers / npm app**, not a uv member — the old in-repo `apps/frontend` Next.js app was
**removed** (see Gotchas). Tools are pinned with **mise**; tasks run through **just**.

## Common commands

| Command | What it does |
|---|---|
| `just setup` | First-time `uv sync` (its npm step still targets the removed `apps/frontend` — see Gotchas) — [docs/setup.md](docs/setup.md) |
| `just gpu-sync` | Install the heavy ML stack (`torch/trl/transformers/vllm`) — **run on a GPU box / the cluster** |
| `just backend` | FastAPI control plane on `:8000` (`zo_backend.main:app`) — what the dashboard reads |
| `just dev` | ⚠️ Backend + the **removed** `apps/frontend` — frontend half now fails; run the dashboard separately (Gotchas) |
| `just train <config>` | Local SFT from a YAML config. Add `--dry-run` to skip torch entirely |
| `just grpo <config>` | Local GRPO / RL run |
| `just submit <config>` | Render + submit a training job to Leonardo SLURM over SSH |
| `just cluster-watch` | `squeue --me` on the cluster |
| `just eval <task> <model>` | Run a *legacy* task-based eval against a served model |
| `just track <args>` | **The track eval**: predict → 3 CSVs + metrics → tagged registry/W&B run (`zo-track`) |
| `just data` | (Re)generate the shared deterministic fab corpus → `data/generated` (or `$ZO_DATA_DIR`) |
| `just judge-eval <args>` | Leonardo batch inference + track scoring (login node; see `docs/leonardo-eval.md`) |
| `just agent <scenario> <model>` | Run an agent scenario (tool-use rollout) |
| `just serve <model>` | Start a vLLM OpenAI-compatible server (default port 8001) |
| `just runs` | List all experiment runs |
| `just wt <name>` | New git worktree + branch for parallel work |
| `just lint` / `just fmt` / `just test` | ruff check / ruff format / pytest (unit only; `pytest -m integration` for live HF) |

CLIs also run directly: `uv run zo-train sft -c <config> --dry-run`, `uv run zo-runs show <id>`, etc.

**Dashboard (submission UI)** is run separately, not via `just`:
`cd infineon-results-dashboard && npm install && npm run build:data && npm run dev -- --port 8787`
→ http://localhost:8787. Static headline numbers render alone; live registry/compare/inference
sections call the backend at `:8000` when it's up. See its [README](infineon-results-dashboard/README.md).

## The run registry — the shared contract

Every run is a directory: `<ZO_EXPERIMENTS_DIR>/<run_id>/` where
`run_id = <YYYYMMDD_HHMMSS>_<kind>_<slug>_<rand>` (default scratch:
`~/.cache/zo-experiments`; set `ZO_EXPERIMENTS_DIR=./experiments` for the legacy in-repo folder).

- `meta.json` — `RunMeta` (id, name, kind, status, git info, slurm_job_id, config, metrics summary)
- `metrics.jsonl` — one JSON object per logged step (append-only)
- `config.yaml` — the exact config the run used
- `logs/`, `artifacts/` — free-form

The flow is **one direction**: training/eval/agent code writes via
`zo_common.registry.append_metric(run_id, step=i, **metrics)` → the backend reads the files →
the frontend plots them. When `WANDB_API_KEY` is set, metrics also land in W&B; checkpoints on HF.
Dashboard source: `ZO_RESULTS_SOURCE=local|wandb|repo`. Playbook: `docs/eval-and-artifacts.md`.
To surface a new metric anywhere, just append it; nothing else needs to change.
**`zo_common` is shared by every package** — coordinate before changing its schemas.

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

- The local directory is `Zero One Philyr` (with spaces); the **git remote is
  `gardan4/Zero-One-XCombinator`** and the team/org is **XCombinator** (W&B `XCombinator/XCombinator`,
  HF `XCombinator/*`). Spaces in the local path break some tooling — consider renaming to match.
- **The in-repo `apps/frontend` Next.js dashboard was removed** (commit "remove front"). The
  submission dashboard is the standalone **`infineon-results-dashboard/`** (Cloudflare Workers, `:8787`).
  **`scripts/{setup,dev,frontend}.py` still target the deleted `apps/frontend`** → the npm steps of
  `just setup` / `just dev` / `just frontend` fail (`uv sync` and the backend still work). Use
  `just backend` + run the dashboard per its README until those scripts are repointed.
- A literal **`$SCRATCH/` directory** can appear at the repo root on a laptop where `$SCRATCH` is
  unset (an HF cache written to the unexpanded path). It's gitignored — safe to `rm -rf '$SCRATCH'`.
- **`trl` API drifts between versions.** If `SFTConfig` / `GRPOConfig` kwargs error out, check the
  installed version against [`training.md`](.claude/knowledge/training.md) and note what you found.
- Cluster host / partition / account / QOS in `.env.example` are **best guesses** — confirm with
  the organizers on-site and record the real values in [`cluster.md`](.claude/knowledge/cluster.md).

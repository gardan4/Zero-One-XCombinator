# ADR 2026-05-29 — Initial architecture

**Status:** accepted (scaffold). Revisit if the cluster or hackathon constraints differ from guesses.

## Context
2-person team, ~2.5-day hackathon, theme is real model training / RL / agents. We need to prototype
fast and **in parallel**, train heavy jobs on the Leonardo GPU cluster, and demo measurable results.
Laptops have no GPU; the cluster does.

## Decisions
1. **uv workspace monorepo** (`apps/*` + `packages/*`, one lockfile). Shared code (`zo_common`) is a
   real package every member imports; avoids copy-paste drift.
2. **Local-light / cluster-heavy** via an optional **`[gpu]` extra**. Base installs are fast on a
   laptop; the cluster runs `uv sync --extra gpu`. Training code lazy-imports torch/trl.
3. **File-based run registry** (`experiments/<run_id>/`), **no database**. Producers append to
   `metrics.jsonl`; the FastAPI backend reads files; the Next.js frontend plots them. Survives
   crashes, works over shared scratch, trivially git-diffable, zero infra to stand up.
4. **`--dry-run`** simulates training curves with no torch, so the full pipeline (config → registry
   → API → dashboard) is testable on a laptop before spending cluster time.
5. **SLURM-over-SSH submission** (`zo-cluster`) rendering an sbatch template, rather than a
   persistent cluster agent — simplest thing that works for a hackathon.
6. **OpenAI-compatible LLM client** (httpx) targeting a **vLLM** server, shared by eval + agent
   harness — one interface for both, and it's how we eval fine-tuned checkpoints.
7. **Hand-rolled SVG sparkline** on the dashboard — no chart library, no build weight.
8. **git worktrees** (`just wt`) for parallel experiments, with optional shared `ZO_EXPERIMENTS_DIR`.

## Alternatives considered
- **W&B / MLflow as the source of truth** → kept W&B as *optional* logging, but the registry is the
  contract so the stack works fully offline / without accounts.
- **A database (sqlite/postgres)** for runs → overkill for a weekend; files are simpler and shareable.
- **A monolithic package** → workspace keeps train/eval/agent/backend boundaries clean for two people
  working in parallel.

## Consequences
- `zo_common` schemas are a shared contract — coordinate changes.
- The backend only sees runs on its local filesystem; sharing requires `ZO_EXPERIMENTS_DIR` on shared
  scratch (or rsync). Documented in [cluster.md](../cluster.md).
- `trl` version drift is the most likely source of breakage; mitigated by `--dry-run` + a gotchas
  note in [training.md](../training.md).

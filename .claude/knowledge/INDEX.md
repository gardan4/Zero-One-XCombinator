# Knowledge base — Zero One Philyr

The team's shared, **ever-updating** brain. Your teammate's Claude can't see your session — only
what's written here. Read this at the start of a session; append whenever you learn something
non-obvious. `/log-learning` does the append in one step.

## How to use
- One file per topic (below). Keep entries short and concrete.
- When you learn something: add it to the right topic file **and** drop a one-line dated entry in
  the Learnings log at the bottom of this file.
- Fix or replace stale notes rather than piling on. A wrong note costs more than a missing one.

## Topics
- [stack.md](stack.md) — monorepo layout, uv workspace, tooling, dep split, recommended Claude settings
- [cluster.md](cluster.md) — Leonardo (CINECA) SLURM: login, partitions, submission flow, scratch
- [training.md](training.md) — trl SFT + GRPO, LoRA, config schema, dry-run, version gotchas
- [eval.md](eval.md) — task spec format, metrics, serving a model, adding a task
- [agents.md](agents.md) — tool registry, rollout loop, scenarios, adding a tool
- [hackathon.md](hackathon.md) — Zero One hackathon: theme, dates, rules, logistics
- [decisions/](decisions/) — architecture decision records (ADRs)

## Open questions to resolve on-site
Things the scaffold guessed at — confirm and update the relevant file:
- Real cluster host / account / partition / QOS / time limits → [cluster.md](cluster.md)
- Which base models are allowed / available in cache → [hackathon.md](hackathon.md) + [training.md](training.md)
- Judging criteria (what gets scored?) → [hackathon.md](hackathon.md)
- Installed `trl` version on the cluster and its `SFTConfig`/`GRPOConfig` API → [training.md](training.md)

## Learnings log
Newest first. Format: `YYYY-MM-DD — one line — (topic file)`

- 2026-05-29 — Run `mise trust` once per machine after cloning, or mise blocks `node`/`uv` (npm install fails). — (stack)
- 2026-05-29 — Root `pyproject.toml` must depend on all members or `uv sync` installs nothing; `gpu` extra re-exposed at root so `uv sync --extra gpu` works. — (stack)
- 2026-05-29 — Initial scaffold: uv workspace, file-based run registry, local-light/cluster-heavy split via `[gpu]` extra, `--dry-run` path that simulates metrics without torch. — (stack, decisions/2026-05-29-initial-architecture)

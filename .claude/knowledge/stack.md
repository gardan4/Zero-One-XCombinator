# Stack & monorepo

## Shape
- **uv workspace**: one virtual root `pyproject.toml` (no `[build-system]`), one `uv.lock`.
  Members: `apps/backend`, `packages/common`, `packages/training`, `packages/eval`,
  `packages/agent`. Cross-deps wired via `[tool.uv.sources]` → `{ workspace = true }`.
- `apps/frontend` is a **separate npm app** (Next.js 16 / React 19 / Tailwind v4), not a uv member.
- Non-code dirs (not workspace members): `data/industrial-infineon/` (vendored track data + grammar +
  `generate_sequences.py`, ~21 MB of CSV — tracked, not gitignored) and `docs/` (track briefing,
  Leonardo deck, `submission/` templates). See [track-industrial-ai.md](track-industrial-ai.md).
- Tooling pinned by **mise** (`mise.toml`: python 3.12, node 20, uv, just). Tasks run via **just**
  (`Justfile`). First-time per machine: **`mise trust`** (mise refuses to provide `node`/`uv` until
  the repo's `mise.toml` is trusted — `npm`/`uv` error otherwise), then `mise install`, then
  `just setup`.
- The root `pyproject.toml` **depends on every member** so a plain `uv sync` installs the whole
  workspace; the `gpu` extra is re-exposed at the root (`gpu = ["zo-train[gpu]"]`) so
  `uv sync --extra gpu` works from the repo root.

## Packages & entry points
| Package | Module | CLI(s) | Role |
|---|---|---|---|
| common | `zo_common` | `zo-runs` | run registry, config schema, paths, OpenAI-compatible LLM client |
| training | `zo_train` | `zo-train`, `zo-cluster` | SFT + GRPO (trl); SLURM submission |
| eval | `zo_eval` | `zo-eval` | task-based eval harness |
| agent | `zo_agent` | `zo-agent` | tool-calling rollout + scenario harness |
| backend | `zo_backend` | (uvicorn) | FastAPI control plane over the registry |

## Dependency split (the important bit)
- Base `uv sync` = **light** deps only (pydantic, typer, rich, httpx, fastapi, jinja2). CLIs import
  on a laptop.
- **`[gpu]` extra** = heavy ML (torch, trl, transformers, peft, accelerate, datasets, vllm, wandb,
  bitsandbytes). `uv sync --extra gpu` on a GPU box only.
- Training code lazy-imports torch/trl *inside functions* so `--dry-run` works with no GPU stack.

## The run registry (shared contract — `zo_common.registry`)
- `experiments/<run_id>/` with `meta.json` (`RunMeta`), `metrics.jsonl`, `config.yaml`, `logs/`,
  `artifacts/`. `run_id = <YYYYMMDD_HHMMSS>_<kind>_<slug>_<rand>` (6-hex random suffix so
  same-second runs on a shared store don't collide).
- Producers (train/eval/agent) call `append_metric(run_id, step=i, **metrics)`; the backend reads
  the files; the frontend plots them. Change `zo_common` schemas only with teammate sign-off.
- `ZO_EXPERIMENTS_DIR` overrides the store location (point at shared scratch to share runs).

## Recommended Claude permissions (opt-in)
The scaffold does **not** ship an `allow` list (granting your own agent permissions is your call).
If you want less prompting during the hackathon, create `.claude/settings.local.json` (gitignored,
per-user) or `.claude/settings.json` (shared) with something like:

```json
{
  "permissions": {
    "allow": [
      "Bash(just:*)", "Bash(uv run:*)", "Bash(uv sync:*)", "Bash(uv add:*)",
      "Bash(npm install)", "Bash(npm run:*)", "Bash(ruff:*)", "Bash(pytest:*)",
      "Bash(git status:*)", "Bash(git diff:*)", "Bash(git log:*)", "Bash(git add:*)",
      "Bash(git commit:*)", "Bash(git worktree:*)", "Bash(./scripts/wt.sh:*)"
    ],
    "ask": [
      "Bash(just submit:*)", "Bash(uv run zo-cluster:*)", "Bash(ssh:*)", "Bash(scp:*)",
      "Bash(git push:*)"
    ],
    "deny": ["Bash(git push --force:*)", "Read(.env)", "Read(*.pem)"]
  }
}
```
Cluster submission and pushes are in `ask` on purpose — they spend shared compute / touch the remote.

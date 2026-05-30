# Stack & monorepo

## Shape
- **uv workspace**: one virtual root `pyproject.toml` (no `[build-system]`), one `uv.lock`.
  Members: `apps/backend`, `packages/common`, `packages/training`, `packages/eval`,
  `packages/agent`. Cross-deps wired via `[tool.uv.sources]` → `{ workspace = true }`.
- The dashboard is **`infineon-results-dashboard/`** — a standalone **Cloudflare Workers / npm app**
  (Worker on `:8787` + a `public/` static UI), **not a uv member**. It replaced the old in-repo
  `apps/frontend` Next.js dashboard, which was **deleted** (commit "remove front", 2026-05-30).
  ⚠️ `scripts/{setup,dev,frontend}.py` (and so the npm steps of `just setup` / `just dev` /
  `just frontend`) still point at the removed `apps/frontend` and fail there — `uv sync`, `just backend`,
  plus the dashboard's own `npm install && npm run dev -- --port 8787` is the working path. See its README.
- Non-code / non-member dirs: `data/industrial-infineon/` (vendored track data + grammar +
  `generate_sequences.py`, ~21 MB CSV — tracked) · `extras/` (labeled proxy eval sets + promoted
  `results/`) · `submissions/XCombinator/REPORT.md` · `tests/` (top-level pytest, `testpaths=["tests"]`) ·
  `docs/` (track briefing, Leonardo deck, `submission/` templates, playbooks). See
  [track-industrial-ai.md](track-industrial-ai.md).
- Minimum local dashboard setup is **Python 3.11+ + uv + Node/npm 20+**. See
  **[docs/setup.md](../../docs/setup.md)**. **No bash/WSL, `just`, `mise`, global `next`, or GPU
  stack required.** Run `uv run python scripts/setup.py`, then `uv run python scripts/dev.py`.
- Optional teammate tooling is pinned by **mise** (`mise.toml`: python 3.12, node 20, uv, just).
  `just setup` and `just dev` are shortcuts for the Python scripts.
- The root `pyproject.toml` **depends on every member** so a plain `uv sync` installs the whole
  workspace; the `gpu` extra is re-exposed at the root (`gpu = ["zo-train[gpu]"]`) so
  `uv sync --extra gpu` works from the repo root.

## Packages & entry points
| Package | Module | CLI(s) | Role |
|---|---|---|---|
| common | `zo_common` | `zo-runs` | run registry, config schema, paths, W&B/HF helpers, OpenAI-compatible LLM client |
| training | `zo_train` | `zo-train`, `zo-cluster` | SFT + GRPO (trl); data factory (grammar/datagen); SLURM submission |
| eval | `zo_eval` | `zo-eval`, **`zo-track`** | `zo-track` = the track pipeline (predict→3 CSVs→score→tagged run); `zo-eval` = legacy task harness |
| agent | `zo_agent` | `zo-agent` | tool-calling rollout + scenario harness |
| backend | `zo_backend` | (uvicorn) | FastAPI control plane over the registry (runs/compare/inference) |

## Dependency split (the important bit)
- Base `uv sync` = **light** deps only (pydantic, typer, rich, httpx, fastapi, jinja2). CLIs import
  on a laptop.
- **`[gpu]` extra** = heavy ML (torch, trl, transformers, peft, accelerate, datasets, vllm, wandb,
  bitsandbytes). `uv sync --extra gpu` on a GPU box only.
- Training code lazy-imports torch/trl *inside functions* so `--dry-run` works with no GPU stack.

## The run registry (shared contract — `zo_common.registry`)
- `$ZO_EXPERIMENTS_DIR/<run_id>/` (default `~/.cache/zo-experiments`; legacy `./experiments`) with
  `meta.json` (`RunMeta`), `metrics.jsonl`, `config.yaml`, `logs/`, `artifacts/`.
  `run_id = <YYYYMMDD_HHMMSS>_<kind>_<slug>_<rand>`.
- Producers dual-write to local scratch; with `WANDB_API_KEY`, metrics also go to W&B. Checkpoints on HF.
  Dashboard: `ZO_RESULTS_SOURCE=local|wandb|repo`. Playbook: [docs/eval-and-artifacts.md](../../docs/eval-and-artifacts.md).
- Change `zo_common` schemas only with teammate sign-off.

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

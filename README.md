# Zero One Philyr

Monorepo for the **Zero One Hack** (Vienna, May 29–31). Built for one thing: spin up
**finetuning / RL / eval / agent** experiments fast, run many in parallel, and watch
them from one dashboard — with the Leonardo A100 cluster doing the heavy lifting.

> New here? Read [`CLAUDE.md`](CLAUDE.md) and [`.claude/knowledge/INDEX.md`](.claude/knowledge/INDEX.md) first.

**Team report:** [`REPORT.md`](REPORT.md) (also [`submissions/XCombinator/REPORT.md`](submissions/XCombinator/REPORT.md)).  
**Organizer eval CSVs:** [`extras/results/kickoff-final/`](extras/results/kickoff-final/) (`nextstep.csv`, `completion.csv`, `anomaly.csv`).

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
experiments/    local scratch run registry (default: ~/.cache/zo-experiments) — gitignored
.claude/        team Claude config: settings, slash commands, subagents, knowledge base
scripts/        dev.py, setup.py (cross-platform), wt.sh (worktrees)
```

## Quickstart

**Full first-time guide:** **[docs/setup.md](docs/setup.md)** (Windows, macOS, Linux).

```bash
git clone <repo-url> && cd Zero-One-XCombinator
uv run python scripts/setup.py   # uv sync (light, no GPU) + frontend npm install
uv run python scripts/dev.py     # backend :8000 + frontend :3000
```

Requirements: **Python 3.11+**, **uv**, and **Node.js/npm 20+**. No bash, WSL, `just`, `mise`, global
`next`, or GPU libraries are required for the dashboard smoke path.

Install `uv`: [official guide](https://docs.astral.sh/uv/getting-started/installation/)
(`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` on Windows,
`curl -LsSf https://astral.sh/uv/install.sh | sh` on macOS/Linux).

Optional teammate shortcuts: `mise trust && mise install`, then `just setup` and `just dev`.

Copy `.env.example` to `.env` for W&B, Hugging Face, cluster, and dashboard source settings
(`WANDB_API_KEY`, `HF_TOKEN`, `ZO_RESULTS_SOURCE`, etc.). See [docs/eval-and-artifacts.md](docs/eval-and-artifacts.md).

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
just train packages/training/configs/sft_smoke.yaml --dry-run   # validate config + registry path
just submit packages/training/configs/sft_fab_lofo_mosfet.yaml # Leonardo training
just runs                                                     # list local scratch runs
just track "-p hf --model XCombinator/... -V v1 --gold ..."   # eval → W&B + CSVs
```

Every run writes a local scratch copy **and** (when `WANDB_API_KEY` is set) logs to W&B.
Weights go to Hugging Face; the dashboard reads `source=local|wandb|repo`.
Full playbook: **[docs/eval-and-artifacts.md](docs/eval-and-artifacts.md)**.

## Working in parallel (4 people, many experiments)

```bash
just wt grpo-reward-v2   # new git worktree + branch ../zo-grpo-reward-v2, isolated experiments
```

Each worktree is an independent checkout, so teammates (or parallel Claude sessions) can train
and eval different ideas at once without stepping on each other. See [`CLAUDE.md`](CLAUDE.md).

## Track eval, artifacts, and known gaps

Industrial AI (Infineon): submit three CSVs (`nextstep`, `completion`, `anomaly`); organizers score
them with vendored `data/industrial-infineon/eval/eval_metrics.py` (they hold kickoff labels).
Self-eval, tagging, and HF model cards: **[docs/eval-and-artifacts.md](docs/eval-and-artifacts.md)**.
See also [docs/track-industrial-sources.md](docs/track-industrial-sources.md).

| Where | What |
|-------|------|
| **W&B** `XCombinator/XCombinator` | Training + eval metrics, eval CSV artifacts |
| **Hugging Face** `XCombinator/*` | Checkpoints, `training_manifest.json`, model card |
| **Local scratch** | Disposable cache (`~/.cache/zo-experiments` by default) |
| **`extras/results/`** | Promoted final CSVs for pitch / REPORT |

Production run checklist: **[docs/eval-and-artifacts.md](docs/eval-and-artifacts.md)** (playbook at top).

Repro eval: `just kickoff-predict "-p hf --model XCombinator/<repo> -V <tag> …"` or local proxy via
`just local-eval MOSFET` (see `docs/leonardo-eval.md`).

## Leonardo inference & eval

**Judges:** see **[docs/judge-quickstart.md](docs/judge-quickstart.md)** — Python + pip only; **uv is optional**.

Full guide: **[docs/leonardo-eval.md](docs/leonardo-eval.md)**.

**Local smoke test (Windows/macOS — pip only, no uv):**

```bash
python -m pip install -r requirements-inference.txt
python scripts/hub_infer.py --prompt "Say hello"
```

**Dry-run batch eval sbatch from laptop (pip only):**

```bash
python -m pip install -r requirements-orchestrator.txt
python scripts/zo_cluster.py judge-eval --dry-run --no-stage --eval-dir extras/eval_local
```

**Optional (if you have [uv](https://docs.astral.sh/uv/) installed):**  
`uv run python scripts/hub_infer.py …` · `uv run zo-cluster judge-eval --dry-run …` · `just judge-eval --dry-run --local`

Quick path on a Leonardo login node:

```bash
cp .env.example .env   # ZO_CLUSTER_USER, HF_TOKEN, ZO_INFER_MODEL, ZO_CLUSTER_ON_LOGIN=1
just judge-setup && just judge-stage && just judge-eval --local
```

## Leonardo smoke finetune (Windows / macOS / Linux)

End-to-end: sync repo → prestage on login → SLURM LoRA train → optional HF upload.

```bash
cp .env.example .env
python -m pip install -r requirements-orchestrator.txt
python scripts/leonardo_smoke.py --dry-run
python scripts/leonardo_smoke.py
python scripts/leonardo_smoke.py --wait-upload
```

**Optional (uv):** `uv run zo-cluster leonardo-smoke …` or `just leonardo-smoke`.  
Configs: `packages/training/configs/leonardo_smoke_hf.yaml`, `leonardo_sft_fab*.yaml`.

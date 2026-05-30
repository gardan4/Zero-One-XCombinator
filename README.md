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

## Track eval, artifacts, and known gaps

Industrial AI (Infineon) scoring uses three submission CSVs and metrics documented in
`data/industrial-infineon/training_data/generation_rules.md`. The organizer scorer
`eval_metrics.py` and kickoff eval CSVs are **not** in the public repo — see
**[docs/track-industrial-sources.md](docs/track-industrial-sources.md)**.

| Where | What |
|-------|------|
| Hugging Face **`XCombinator`** | Fine-tuned checkpoints (not committed to git) |
| W&B **`XCombinator/XCombinator`** | Training logs and loss curves (`WANDB_ENTITY` / `WANDB_PROJECT` in `.env`) |
| `experiments/<run_id>/` | Registry runs, metrics, `results/*.csv`, `metrics_report.md` |

Repro eval: `just track "-p hf --model XCombinator/<repo> --version <tag> …"` (see `docs/leonardo-eval.md`).

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

# Judge quickstart — choose the smallest path

There are two common judge paths:

- **Dashboard/local smoke:** needs **Python 3.11+**, **uv**, and **Node.js/npm 20+**. See
  [setup.md](setup.md).
- **Leonardo-only CLI flows:** need only **Python 3.11+**, **pip**, and SSH tooling on the laptop.

**`uv`**, **`just`**, **`mise`**, and bash are optional for the Leonardo-only path.

---

## Finetune (`leonardo-finetune-reference`)

Train a smoke LoRA on Leonardo and upload to Hugging Face.

### Laptop (Windows / macOS) — pip only, no dashboard

```bash
git clone <repo> && cd Zero-One-Philyr
cp .env.example .env          # fill ZO_CLUSTER_USER, HF_TOKEN, paths
python -m pip install -r requirements-orchestrator.txt

python scripts/leonardo_smoke.py --dry-run    # validate sbatch locally
python scripts/leonardo_smoke.py              # sync → prestage → submit
python scripts/leonardo_smoke.py --wait-upload
```

Needs **OpenSSH** (`ssh`/`scp`) or **PuTTY** (`plink`/`pscp` + `ZO_CLUSTER_PASSWORD` in `.env`).

### Optional: uv (developers)

```bash
uv sync
uv run python scripts/leonardo_smoke.py --dry-run
uv run zo-cluster leonardo-smoke --dry-run
just leonardo-smoke --dry-run
```

---

## Inference & track eval

Run a finetuned checkpoint locally or batch-eval on Leonardo. We submit three CSVs; organizers score
with their script. Self-eval: [track-industrial-sources.md](track-industrial-sources.md).
Team checkpoints: Hugging Face **`XCombinator`**; training logs: W&B **`XCombinator/XCombinator`**.
Every `zo-track predict` run requires `--version` and writes `metrics_report.md` under the run results dir.

### Laptop — pip only (local HF inference, no dashboard)

```bash
python -m pip install -r requirements-inference.txt
cp .env.example .env          # HF_TOKEN for private XCombinator models
python scripts/hub_infer.py --prompt "Say hello"
```

### Laptop — pip only (SLURM dry-run / submit)

```bash
python -m pip install -r requirements-orchestrator.txt
python scripts/zo_cluster.py judge-eval --dry-run --no-stage --eval-dir extras/eval_local
python scripts/zo_cluster.py judge-eval          # submit over SSH
```

### Leonardo login node — pip or uv

```bash
# pip (no uv required on laptop)
python -m pip install -r requirements-orchestrator.txt
python scripts/zo_cluster.py judge-setup
python scripts/zo_cluster.py judge-stage
python scripts/zo_cluster.py judge-eval --local

# uv / just (optional)
just judge-setup && just judge-stage && just judge-eval --local
```

For **local HF inference on the login node**: `requirements-inference.txt` + `scripts/hub_infer.py`.

---

## Requirements files

| File | Purpose | uv alternative |
|------|---------|----------------|
| `requirements-orchestrator.txt` | Laptop SLURM / judge CLIs (`zo_cluster.py`, `leonardo_smoke.py`) | `uv sync` (light workspace) |
| `requirements-inference.txt` | Local HF inference (`hub_infer.py`) | `uv sync --extra gpu` |

Leonardo **GPU jobs** use `uv sync --extra gpu` on the login node (prestage installs uv there) — judges do not need uv on their laptop.

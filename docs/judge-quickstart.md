# Judge quickstart — no uv required

Judges only need **Python 3.11+** and **`pip`** on a laptop (Windows or macOS).  
**`uv`**, **`just`**, and **bash** are optional — developers on the team may prefer them.

---

## Finetune (`leonardo-finetune-reference`)

Train a smoke LoRA on Leonardo and upload to Hugging Face.

### Laptop (Windows / macOS) — pip only

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

## Inference (`feature/leonardo-inference-eval`)

Run a finetuned checkpoint locally or batch-eval on Leonardo.

### Laptop — pip only (local inference)

```bash
python -m pip install -r requirements-inference.txt
cp .env.example .env          # HF_TOKEN for private XCombinator models
python scripts/hub_infer.py --prompt "Say hello"
```

### Leonardo login node — pip or uv

On the cluster login node (after SSH), either:

```bash
# pip
python -m pip install -r requirements-orchestrator.txt
python scripts/leonardo_smoke.py --dry-run   # if testing submit from login

# uv (optional — prestage installs uv on Leonardo anyway)
just judge-setup && just judge-stage && just judge-eval --local
```

For **local HF inference on the login node**, use `requirements-inference.txt` + `scripts/hub_infer.py` — no uv required.

For **batch eval CSVs without just**:

```bash
python -m pip install -r requirements-orchestrator.txt   # light CLIs
python -m pip install -r requirements-inference.txt        # + torch for -p hf
uv run zo-track predict -p hf ...   # or PYTHONPATH=packages/... python -m zo_eval.track_cli
```

Prefer **`python scripts/hub_infer.py`** for a one-line smoke test; use **`python scripts/leonardo_smoke.py`** / judge eval when you need the full SLURM pipeline from a laptop.

---

## Requirements files

| File | Purpose | uv alternative |
|------|---------|----------------|
| `requirements-orchestrator.txt` | Laptop SLURM submit / leonardo_smoke | `uv sync` (light, whole workspace) |
| `requirements-inference.txt` | Local HF inference (`hub_infer.py`) | `uv sync --extra gpu` |

Leonardo **GPU jobs** always use `uv sync --extra gpu` on the login node (prestage script installs uv there) — judges do not need uv on their laptop for that.

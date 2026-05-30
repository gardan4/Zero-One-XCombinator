# Cluster — Leonardo (CINECA)

Real facts from the official Leonardo onboarding deck (Simeon Harrison / Martin Pfister,
`docs/Z10_compressed.pdf` pp. 80–96) and the AI:AT HPC onboarding kit
(https://ai-at.eu/hpc-onboarding/ — Ch. 5 "First steps on LEONARDO", Ch. 6 "Software").
Leonardo = #10 on the Top500, **Nvidia A100 (64 GB VRAM)** booster nodes. Storage is shared across
all login + compute nodes.

## Our resource budget (confirmed by team lead)
| Limit | Value |
|---|---|
| GPUs / node | **4** (A100, **64 GB VRAM each** → 256 GB VRAM/node) |
| RAM / node | **up to 512 GB** |
| CPUs / node | 32 (8 × gpus) |
| **Nodes (with reservation `s_tra_ncc`)** | **up to 4** → max **16 GPUs** for our team |

> So the reservation is **not** single-node — we can request up to 4 nodes × 4 GPUs through it.
> Still spend deliberately and `--dry-run` / smoke-test on a login node or `lrd_all_serial` first.
> Scale up only when a smaller run proves the pipeline.

## Login (no 2FA at the hackathon)
Plain SSH, any of these login nodes (round-robin if one is busy):
```
ssh <username>@login01-ext.leonardo.cineca.it
ssh <username>@login02-ext.leonardo.cineca.it
ssh <username>@login05-ext.leonardo.cineca.it
ssh <username>@login07-ext.leonardo.cineca.it
```
Credentials come by e-mail / Discord at the event. **For the hackathon, two-factor auth is OFF** —
so for production CINECA accounts later this differs (they normally require 2FA/OTP).

## Storage (know which to use)
| Path | Limit | Use for |
|---|---|---|
| `$HOME` | 50 GB | code, small configs, pixi env |
| `$SCRATCH` | large (files **deleted after 40 days**) | **datasets, checkpoints, HF caches, big outputs** |
| `$PUBLIC` | 50 GB | sharing files with other Leonardo users |
| `$FAST`, `$WORK` | — | **do NOT use during the hackathon** |

Point HF/torch caches at scratch: `export HF_HOME=$SCRATCH/hf` (home is tiny).

## Login-node limits + interactive work
- **Login nodes kill any process after 10 min of CPU time.** Don't train, build big wheels, or pull
  large containers directly on the login node.
- For longer interactive prep (compiles, container pulls, data gen, debugging) grab a serial node:
  ```
  srun --partition=lrd_all_serial --time 04:00:00 --gres=tmpfs:100G --mem=16G --pty bash
  ```

## Package manager: Pixi (not uv on the cluster)
The cluster onboarding standardizes on **Pixi** (conda-forge + PyPI, reproducible):
```
curl -fsSL https://pixi.sh/install.sh | bash
pixi init myproj && cd myproj
pixi add python                # from conda-forge
pixi add --pypi torch trl      # from PyPI
pixi run python script.py
```
Inside SLURM the deck runs jobs as: `pixi run --as-is [--manifest-path <pixi_project>] python3 script.py`.
> Our repo uses **uv** locally. On Leonardo either (a) recreate the heavy env with pixi, or
> (b) use a Singularity container (below). uv *can* work on a login node, but pixi is the supported
> path and avoids fighting the module system.

## Containers: Singularity / Apptainer (no Docker on HPC)
Convert a Docker image to a `.sif` (do the pull on a serial node — it's CPU-heavy + needs internet):
```
srun --partition=lrd_all_serial --time 04:00:00 --gres=tmpfs:100G --mem=16G --pty \
  singularity pull vllm-openai.sif docker://docker.io/vllm/vllm-openai:0.21.0-cu129
```
Run inside it (`--nv` = expose GPUs, `--bind` = mount scratch):
```
singularity exec --nv --bind $SCRATCH:/scratch container.sif python3 script.py
```

## SLURM — the hackathon job template
Key bits: partition `boost_usr_prod`, **reservation `s_tra_ncc`** (covers up to **4 nodes**), and
fair-share scaling **cpus = 8 × gpus-per-task**. **4 GPUs/node** (64 GB VRAM each); RAM up to
**512 GB/node**. The deck's own scripts request **mem = 120 GB × gpus** (conservative, leaves OS
headroom) — fine to use, but you may go up to 512 GB on a full node.
```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc   # hackathon reservation (up to 4 nodes for our team — don't hog)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1         # up to 4 on Leonardo (64 GB VRAM each)
#SBATCH --mem=120GB               # 120GB × gpus (node max 512GB)
#SBATCH --cpus-per-task=8         # 8 × gpus-per-task
#SBATCH --time=0:30:00            # HH:MM:SS, up to 24:00:00

# Run via pixi:
pixi run --as-is python3 script.py
# …or via a container:
# singularity exec --nv container.sif python3 script.py
```
Scale GPUs by editing three lines together, e.g. 2 GPUs → `--gpus-per-task=2 --mem=240GB --cpus-per-task=16`;
4 GPUs → `--gpus-per-task=4 --mem=480GB --cpus-per-task=32` (deck slides 87–91 walk 1→2→4 GPU; bump
mem toward 512GB if a full-node job needs it).
**Multi-node** (slide 92, `--nodes=2…4`): keep `--reservation=s_tra_ncc` (it spans up to 4 nodes) and
launch the workload with `srun … $CONTAINER …`. *(Slide 92 shows the reservation line struck out, but
that's a generic multi-node example — our team's reservation does cover multi-node.)* Our ceiling:
**4 nodes × 4 GPUs = 16 A100s**. Prove the pipeline at 1 node × N GPUs before scaling out.

### Useful SLURM commands
```
sbatch job.sh                         # submit
squeue --me                           # my queued/running jobs
cat slurm-<job_id>.out                # job output
tail -c +0 -f slurm-<job_id>.out      # follow output live
scancel <job_id>                      # cancel
srun --overlap --pty --jobid=<id> bash  # shell into a running job's node
```

## Internet access (the big gotcha)
- **Login nodes HAVE internet** → do all large downloads here (models, datasets, container pulls).
- **Compute nodes have NO internet.** Pre-stage everything to `$SCRATCH` from a login node, OR
  route low-bandwidth traffic through the proxy by exporting these in the SLURM script:
  ```bash
  export HTTP_PROXY=http://proxyuser:<PROXY_PASSWORD>@10.99.0.1:38425
  export HTTPS_PROXY=$HTTP_PROXY
  export http_proxy=$HTTP_PROXY
  export https_proxy=$HTTP_PROXY
  ```
  > The exact `<PROXY_PASSWORD>` is printed on `docs/Z10_compressed.pdf` p. 95. **Do not commit the
  > literal credential** — this knowledge base ships in our public+MIT repo at submission and the
  > jury checks for "no secrets in git." Keep the password in your local `.env` / shell only.
  - The proxy restarts every ~10 min (login-node CPU limit) → TCP connections drop briefly.
  - **Only use the proxy for low-bandwidth traffic.** Big files → always from a login node.

## How this maps to our scaffold
`zo-cluster submit` renders `slurm/train.sbatch.j2` from `.env`. Status of the known gaps:
1. ✅ **DONE** — `_render`/template now emit `#SBATCH --reservation` (gated to **1-node only**, per
   slide 92), `--gpus-per-task=N`, and fair-share `--mem=120×N` / `--cpus-per-task=8×N` (auto-derived
   from `ZO_SLURM_GPUS_PER_NODE`; override via `ZO_SLURM_MEM`/`ZO_SLURM_CPUS`). Verified by dry-run.
2. ⚠️ **Partly done — still the real env decision.** The template now runs `uv sync --extra gpu
   **--offline**` on the compute node and exports the proxy if `ZO_CLUSTER_PROXY` is set. This means you
   **must pre-stage on a login node first**: `cd $HOME/Zero-One-Philyr && uv sync --extra gpu` (populates
   `./.venv` on shared `$HOME`, visible to compute nodes). Open question: torch+vllm may blow the 50 GB
   `$HOME` quota → may need a pixi env / `.sif` on `$SCRATCH` instead. Decide on-site.
3. **Pre-stage** the base model + dataset to `$SCRATCH` from a login node (runtime step). `HF_HOME` is
   already exported by the template (`.env` → `$SCRATCH/hf`).
4. ✅ `ZO_EXPERIMENTS_DIR` is exported by the template (`.env` → `$SCRATCH/zo-experiments`) so runs
   survive and the backend (run on a login node) can read them.

## Append below as you learn the real cluster

### Leonardo finetune recipe (2026-05-30) — adapted from the `leonardo-finetune-reference` branch
End-to-end flow: `scripts/leonardo_smoke_hf.sh [config]` (default = the toy smoke; pass
`packages/training/configs/leonardo_smoke_fab.yaml` for OUR data):
1. **rsync** repo → login node (excludes .git/.env/.venv/node_modules/experiments/caches; writes a
   cluster `.env` from local env, stripping the SSH password).
2. **Pre-stage on the login node** (`scripts/leonardo_remote_prestage.sh`, has internet):
   `uv sync --extra gpu`, warm imports, `snapshot_download` the base model → `$ZO_SMOKE_BASE_MODEL_DIR`,
   a `zo-train wandb-smoke` connectivity check.
3. **SLURM GPU job** (`zo-cluster submit`): `uv sync --extra gpu --offline` (reuses the staged venv),
   `TRANSFORMERS_OFFLINE=1`/`HF_HUB_OFFLINE=1`, model loaded with `local_files_only` (config `model:`
   = the scratch dir), **live W&B via the proxy** (XCombinator/XCombinator) + GPU sanity checks.
4. **After the job**: `scripts/leonardo_upload_artifact.sh` pushes the adapter to HF Hub from login.
- **Env pins that resolve/train on Leonardo:** `transformers>=4.44,<5`, `trl>=0.12,<1`, `torch<2.8`;
  **vLLM + bitsandbytes dropped** from the gpu extra (conflicting torch + 50 GB $HOME bloat). `sft.py`
  uses `_supported_kwargs()` to stay robust to SFTConfig/SFTTrainer kwarg drift. Serving installs vLLM
  separately later.
- Set each Leonardo config's `model:` to your absolute `$ZO_SMOKE_BASE_MODEL_DIR` (configs don't expand
  `$SCRATCH`). `uv run zo-cluster submit -c <cfg> --dry-run` renders the sbatch locally (no SSH) to check it.

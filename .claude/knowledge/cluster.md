# Cluster — Leonardo (CINECA)

Real facts from the official Leonardo onboarding deck (Simeon Harrison / Martin Pfister,
`docs/Z10_compressed.pdf` pp. 80–96) and the AI:AT HPC onboarding kit
(https://ai-at.eu/hpc-onboarding/ — Ch. 5 "First steps on LEONARDO", Ch. 6 "Software").
Leonardo = #10 on the Top500, **Nvidia A100** (booster module). Storage is shared across all
login + compute nodes.

> Spend GPU time deliberately: the hackathon reservation is **only enough for ~1 node per team**.
> Always `--dry-run` / smoke-test on a login node or `lrd_all_serial` before submitting a GPU job.

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
Key bits: partition `boost_usr_prod`, **reservation `s_tra_ncc`**, and fair-share scaling where
**mem = 120 GB × gpus-per-task** and **cpus = 8 × gpus-per-task**. Up to **4 GPUs** per node.
```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc   # hackathon reservation (≈1 node per team — don't hog)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1         # up to 4 on Leonardo
#SBATCH --mem=120GB               # 120GB × gpus-per-task
#SBATCH --cpus-per-task=8         # 8 × gpus-per-task
#SBATCH --time=0:30:00            # HH:MM:SS, up to 24:00:00

# Run via pixi:
pixi run --as-is python3 script.py
# …or via a container:
# singularity exec --nv container.sif python3 script.py
```
Scale GPUs by editing three lines together, e.g. 2 GPUs → `--gpus-per-task=2 --mem=240GB --cpus-per-task=16`;
4 GPUs → `--gpus-per-task=4 --mem=480GB --cpus-per-task=32`. Multi-node (`--nodes=2`) needs `srun $CONTAINER …`
but the reservation usually only covers **1 node/team**, so prefer 1 node × N GPUs.

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

## How this maps to our scaffold (gaps to close on-site)
Our `zo-cluster` submit flow + `slurm/train.sbatch.j2` were scaffolded against guesses. Before the
first real submit, update the template to:
1. Add `#SBATCH --reservation=s_tra_ncc` and switch the GPU request to `--gpus-per-task` with the
   `mem=120×N`, `cpus=8×N` fair-share scaling above.
2. Replace `uv sync --extra gpu` on the compute node — **it won't work (no internet there)**. Either
   pre-build a pixi env / `.sif` on a login node and call `pixi run …` / `singularity exec --nv …`,
   or set the proxy env vars for small installs.
3. Pre-stage the base model + dataset to `$SCRATCH` from a login node; set `HF_HOME=$SCRATCH/hf`.
4. `ZO_EXPERIMENTS_DIR=$SCRATCH/experiments` so runs survive and the backend (run on a login node)
   can read them.

## Append below as you learn the real cluster
- (fill in actual account name, observed queue times, what env recipe worked, once on Leonardo)

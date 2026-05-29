# Cluster — Leonardo (CINECA) via SLURM

> ⚠️ Host / account / partition / QOS below are **best guesses** from `.env.example`. Confirm with
> the organizers on-site and overwrite the real values here.

## What we think we know
- Login host: `login.leonardo.cineca.it` (`ZO_CLUSTER_HOST`).
- Scheduler: SLURM. Guessed GPU partition `boost_usr_prod` (4× A100 64GB per node on the booster).
- You need: `ZO_CLUSTER_USER`, `ZO_SLURM_ACCOUNT`, and likely `ZO_SLURM_QOS`. Set in `.env`.
- Defaults: `ZO_SLURM_TIME=02:00:00`, `ZO_SLURM_NODES=1`, `ZO_SLURM_GPUS_PER_NODE=4`.

## Submission flow (`zo-cluster`, in `packages/training/zo_train/cluster/`)
`just submit <config>` → `uv run zo-cluster submit --config <config>`:
1. Reads `.env` for cluster/SLURM settings.
2. Creates (or attaches to) a run in the registry so it has a `run_id`.
3. Renders `slurm/train.sbatch.j2` into a concrete sbatch script (job-name, account, partition,
   qos, `--gres=gpu:N`, time, output path) for the run.
4. `scp`s the config + sbatch script to `ZO_CLUSTER_REPO_DIR` on the cluster.
5. `ssh`s in and `sbatch`es it, parses the returned job id, writes it back to the run's `meta.json`
   (`slurm_job_id`).
- `just cluster-watch` → `uv run zo-cluster watch` → `ssh ... squeue --me`.

The sbatch job runs `uv sync --extra gpu` then `uv run zo-train <sft|grpo> --config <cfg> --run-id <id>`.
A multi-GPU `accelerate launch` line is included but commented — uncomment for >1 GPU.

## Making the dashboard see cluster runs
The backend reads the local filesystem registry. Two options:
1. **Run the backend on the login node** with `ZO_EXPERIMENTS_DIR` pointed at the same shared
   scratch the job writes to. Cleanest.
2. Periodically `rsync` `experiments/` back to your laptop.

## Practical notes / to confirm
- Put HF caches on scratch, not `$HOME`: set `HF_HOME` (quotas on home are small).
- The repo must exist at `ZO_CLUSTER_REPO_DIR` (default `$HOME/Zero-One-Philyr`) — clone it once,
  `git pull` before submitting, or have submit handle sync (currently it scp's only config+sbatch,
  not the whole repo).
- Confirm whether `uv` is available on compute nodes or must be module-loaded / installed first.
- Confirm internet egress on compute nodes (needed for HF model/dataset downloads, or pre-stage them).

## Append below as you learn the real cluster
- (nothing yet — fill in once you're on Leonardo)

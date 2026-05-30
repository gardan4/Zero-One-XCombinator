#!/bin/bash
#SBATCH --job-name=eval-instruct-smoke2
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=/leonardo/home/usertrain/a08trd0f/Zero-One-Philyr/slurm_logs/%x-%j.out
#SBATCH --error=/leonardo/home/usertrain/a08trd0f/Zero-One-Philyr/slurm_logs/%x-%j.out

set -euo pipefail
cd "/leonardo/home/usertrain/a08trd0f/Zero-One-Philyr"
mkdir -p slurm_logs "/leonardo_scratch/large/usertrain/a08trd0f/zo-experiments/eval-out/eval-instruct-smoke2"

export ZO_EXPERIMENTS_DIR="/leonardo_scratch/large/usertrain/a08trd0f/zo-experiments"
export HF_HOME="/leonardo_scratch/large/usertrain/a08trd0f/hf"

export HTTP_PROXY="http://proxyuser:5dd1d2bd00@10.99.0.1:38425"; export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"; export https_proxy="$HTTP_PROXY"


# GPU stack must be pre-staged on a login node:  cd /leonardo/home/usertrain/a08trd0f/Zero-One-Philyr && uv sync --extra gpu
uv sync --extra gpu --offline

# Batch inference on a fine-tuned checkpoint (no vLLM server required).
uv run zo-track predict \
  --predictor hf \
  --version "instruct-all-v2" \
  --model "/leonardo_scratch/large/usertrain/a08trd0f/zo-experiments/20260530_125539_sft_leonardo-sft-fab-instruct-all_cdd9a0/artifacts" \
  --eval-set "local" \
  --valid "/leonardo/home/usertrain/a08trd0f/Zero-One-Philyr/extras/eval_local/smoke20_MOSFET/eval_input_valid.csv" \
  --anomaly "/leonardo/home/usertrain/a08trd0f/Zero-One-Philyr/extras/eval_local/smoke20_MOSFET/eval_input_anomaly.csv" \
  --gold "/leonardo/home/usertrain/a08trd0f/Zero-One-Philyr/extras/eval_local/smoke20_MOSFET/gold.json" \
  --tasks "nextstep,completion,anomaly" \
  --out "/leonardo_scratch/large/usertrain/a08trd0f/zo-experiments/eval-out/eval-instruct-smoke2" \
  --tags "eval,instruct,split:id,family:MOSFET" \
  --run-id ""

echo "=== judge eval complete ==="
echo "Results: /leonardo_scratch/large/usertrain/a08trd0f/zo-experiments/eval-out/eval-instruct-smoke2"
ls -la "/leonardo_scratch/large/usertrain/a08trd0f/zo-experiments/eval-out/eval-instruct-smoke2"
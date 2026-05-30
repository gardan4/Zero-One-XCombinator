#!/bin/bash
#SBATCH --job-name=eval-all-MOSFET-id
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/leonardo/home/usertrain/a08trd0f/Zero-One-Philyr/slurm_logs/%x-%j.out
#SBATCH --error=/leonardo/home/usertrain/a08trd0f/Zero-One-Philyr/slurm_logs/%x-%j.out

set -euo pipefail
cd "/leonardo/home/usertrain/a08trd0f/Zero-One-Philyr"
mkdir -p slurm_logs "/leonardo_scratch/large/usertrain/a08trd0f/zo-experiments/eval-out/eval-all-MOSFET-id"

export ZO_EXPERIMENTS_DIR="/leonardo_scratch/large/usertrain/a08trd0f/zo-experiments"
export HF_HOME="/leonardo_scratch/large/usertrain/a08trd0f/hf"

export HTTP_PROXY="http://proxyuser:5dd1d2bd00@10.99.0.1:38425"; export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"; export https_proxy="$HTTP_PROXY"


# GPU stack must be pre-staged on a login node:  cd /leonardo/home/usertrain/a08trd0f/Zero-One-Philyr && uv sync --extra gpu
uv sync --extra gpu --offline

# Batch inference on a fine-tuned checkpoint (no vLLM server required).
uv run zo-track predict \
  --predictor hf \
  --model "/leonardo_scratch/large/usertrain/a08trd0f/zo-experiments/20260530_105017_sft_leonardo-sft-fab-all_e639f1/artifacts" \
  --valid "/leonardo/home/usertrain/a08trd0f/Zero-One-Philyr/extras/eval_local/MOSFET/eval_input_valid.csv" \
  --anomaly "/leonardo/home/usertrain/a08trd0f/Zero-One-Philyr/extras/eval_local/MOSFET/eval_input_anomaly.csv" \
  --gold "/leonardo/home/usertrain/a08trd0f/Zero-One-Philyr/extras/eval_local/MOSFET/gold.json" \
  --tasks "nextstep,completion,anomaly" \
  --out "/leonardo_scratch/large/usertrain/a08trd0f/zo-experiments/eval-out/eval-all-MOSFET-id" \
  --tags "eval,split:id,family:MOSFET,model:all,predictor:hf"

echo "=== judge eval complete ==="
echo "Results: /leonardo_scratch/large/usertrain/a08trd0f/zo-experiments/eval-out/eval-all-MOSFET-id"
ls -la "/leonardo_scratch/large/usertrain/a08trd0f/zo-experiments/eval-out/eval-all-MOSFET-id"
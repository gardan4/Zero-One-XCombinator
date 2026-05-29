"""Training recipes (SFT, GRPO/RL) + SLURM submission.

Heavy imports (torch/trl/transformers) live INSIDE the recipe functions so the CLI and
cluster-submission paths import cleanly on a laptop without the GPU extra installed.
"""

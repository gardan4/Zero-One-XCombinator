"""Agent harness: run tool-using rollouts against a served model and score task success.

This is how you measure whether finetuning/RL actually made the model a better *agent*,
not just a better next-token predictor. Add domain tools in `tools.py`.
"""

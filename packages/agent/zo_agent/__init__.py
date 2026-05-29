"""Agent harness: run tool-using rollouts against a served model and score task success.

This is how you measure whether finetuning/RL actually made the model a better *agent*,
not just a better next-token predictor. Add domain tools in `tools.py`.

NOTE: generic scaffold, **not our track's primary path.** The Industrial AI (Infineon) track
is sequence modeling of fab steps, not tool-using agents. Keep as reference; don't invest here
first. See `.claude/knowledge/track-industrial-ai.md`.
"""

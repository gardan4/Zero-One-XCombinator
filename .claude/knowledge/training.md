# Training — SFT & GRPO (trl)

Code: `packages/training/zo_train/` (`sft.py`, `rl.py`, `sim.py`, `cli.py`, `cluster/`).
Configs: `packages/training/configs/*.yaml`.

> **Track reality ([track-industrial-ai.md](track-industrial-ai.md)):** our task is **small-vocab
> sequence modeling** (~120 step tokens), not chat-LLM SFT. A small token-level Transformer trained
> **from scratch** is the clean, fast story — likely a new trainer/data-loader
> (`zo_train/data/process_seq.py`) rather than `SFTTrainer` over chat text. The config schema,
> registry logging, `--dry-run`, and cluster submit flow still apply; the chat-SFT/GRPO bodies below
> are a reference, not a direct fit.

## Config schema (`zo_common.config.ExperimentConfig`)
`name`, `kind` (`sft|dpo|grpo|eval|agent`), `model` (default `Qwen/Qwen2.5-1.5B-Instruct`),
`dataset`, `dataset_split`, `text_field`, `learning_rate`, `epochs`, `batch_size`, `grad_accum`,
`max_seq_len`, `warmup_ratio`, `lora` (bool) + `lora_r`/`lora_alpha`/`lora_dropout`, `bf16`, `seed`,
`output_dir`, and a free-form `extra: dict` (e.g. GRPO's `num_generations`, `max_completion_length`).
Load/save via `from_yaml()` / `to_yaml()`.

## Running
- `just train <cfg>` → `uv run zo-train sft -c <cfg>`. `just grpo <cfg>` for GRPO.
- **`--dry-run`** skips torch entirely and calls `sim.simulate_training()` to write a plausible
  decaying-loss / rising-reward curve into the registry. Use it to validate config + the
  registry→backend→frontend path **before** spending GPU time.
- `--run-id <id>` attaches to an existing run instead of creating one (this is how cluster jobs
  log into the run created at submit time).

## SFT (`sft.py`)
- Lazy-imports `peft`/`transformers`/`trl`. Builds an `SFTConfig` + `SFTTrainer` with a
  `TrainerCallback` that forwards logs to `append_metric`.
- LoRA via `peft` when `cfg.lora`. `bf16` when set.
- Logs to **wandb** automatically iff `WANDB_API_KEY` is set (else local `metrics.jsonl` only).
- If `text_field != "messages"`, passes `dataset_text_field`; otherwise assumes chat-format messages.

## GRPO / RL (`rl.py`)
- Same shape as SFT but `GRPOConfig` + `GRPOTrainer`, with `num_generations` /
  `max_completion_length` pulled from `cfg.extra`.
- Ships a **toy reward** `reward_len` (targets a length) marked `# TODO(track)` — **replace it with
  your real reward** for whatever you're optimizing. This is usually where hackathon points are won.

## Known gotchas
- **trl API drifts.** `SFTConfig`/`GRPOConfig` kwargs get renamed between versions
  (e.g. `max_seq_length` vs `max_seq_len`, `dataset_text_field` location). If a kwarg errors,
  check `uv run python -c "import trl; print(trl.__version__)"` and adjust — then **record the
  working version + kwargs here**.
- On ImportError of the ML stack, the CLI raises "ML deps missing. Run `just gpu-sync`" — that means
  you're on a base (laptop) install; use `--dry-run` or sync the `[gpu]` extra.

## Append below as you learn
- (trl version that worked on the cluster: TBD)

# Knowledge base — Zero One Philyr

The team's shared, **ever-updating** brain. Your teammate's Claude can't see your session — only
what's written here. Read this at the start of a session; append whenever you learn something
non-obvious. `/log-learning` does the append in one step.

## How to use
- One file per topic (below). Keep entries short and concrete.
- When you learn something: add it to the right topic file **and** drop a one-line dated entry in
  the Learnings log at the bottom of this file.
- Fix or replace stale notes rather than piling on. A wrong note costs more than a missing one.

## Topics
- [stack.md](stack.md) — monorepo layout, uv workspace, tooling, dep split, recommended Claude settings
- [cluster.md](cluster.md) — **Leonardo (CINECA)**: SSH login (no 2FA), SLURM + reservation `s_tra_ncc`, Pixi, Singularity, storage, no-compute-internet + proxy
- [track-industrial-ai.md](track-industrial-ai.md) — **our track (Infineon)**: data, grammar, 10 rules, 3 eval tasks + metrics, deliverables, how the scaffold maps
- [training.md](training.md) — trl SFT + GRPO, LoRA, config schema, dry-run, version gotchas
- [eval.md](eval.md) — task spec format, metrics, serving a model, adding a task
- [agents.md](agents.md) — tool registry, rollout loop, scenarios, adding a tool
- [hackathon.md](hackathon.md) — Zero One hackathon: schedule, submission rules (public+MIT+REPORT.md), judging
- [decisions/](decisions/) — architecture decision records (ADRs)

## Sources & where things live
- **Hackathon docs:** https://docs.zero-one.lumos-consulting.at/getting-started/welcome/
- **Track + data repo:** https://github.com/Lumos-Data/zero_one_hack_01 (MIT). Our track under
  `tracks/industrial-infineon/`; submission spec under `submission/`. **Both are vendored locally**
  → `data/industrial-infineon/` is a **byte-faithful 1:1 mirror** of the upstream track folder
  (track docs at root + `training_data/` subfolder with CSVs/grammar/`generate_sequences.py`),
  and `docs/submission/` (`REPORT_TEMPLATE.md`, `SUBMISSION.md`). Refresh from upstream:
  `git clone --depth 1 --filter=blob:none --sparse <repo> /tmp/up && cd /tmp/up && git sparse-checkout set tracks/industrial-infineon submission`
  then copy the two folders. (`eval_metrics.py` + `judging/rubrics.md` are referenced but NOT upstream
  — organizers ship them at kickoff.)
- **HPC onboarding kit:** https://ai-at.eu/hpc-onboarding/ (Ch. 5 first steps, Ch. 6 software).
- **Local `docs/`:** `Track One Assignment.txt` (EN track brief) · `Z10_compressed.pdf` (event +
  Leonardo onboarding deck; cluster info pp. 80–96 — incl. the proxy password on p. 95, **do not
  commit it**).
- **Eval input files** (`eval_input_valid.csv`, `eval_input_anomaly.csv`) are **distributed by
  organizers at kickoff** — not in the public repo.

## Open questions to resolve on-site
Mostly answered now (track chosen = Industrial AI; Leonardo facts captured). Remaining:
- Per-team GPU quota / time budget on Leonardo (deck says ~1 node/team) → [cluster.md](cluster.md)
- Your Leonardo account name + emailed creds; the proxy password (deck p.95, keep out of git) → [cluster.md](cluster.md)
- Env recipe that actually works on a compute node (pixi env vs prebuilt `.sif`) → [cluster.md](cluster.md)
- Exact `eval_metrics.py` CLI flags per task once the eval inputs are distributed → [track-industrial-ai.md](track-industrial-ai.md)
- Whether we train a small seq-model from scratch or fine-tune an open base → [track-industrial-ai.md](track-industrial-ai.md) + [training.md](training.md)

## Learnings log
Newest first. Format: `YYYY-MM-DD — one line — (topic file)`

- 2026-05-30 — **Stream 0 (shared inference/predict core) shipped:** `zo_eval/predict.py` (Predictor + free-text→exact-vocab normalizer, strict/lenient for OOD), `baselines.py` (n-gram/oracle/freq), `track.py`+`track_cli.py` (`zo-track`; `just track`/`local-eval`) → submission CSVs + scored, tagged runs. GPU-free E2E proven; `zo-eval` now declares `zo-train`. — (eval)
- 2026-05-30 — **OOD correction (important):** under the eval's 60/80% cuts next-step TRANSFERS (measured ID top-1 0.69 ≈ OOD 0.705 — shared back-half backbone); the OOD collapse is in the **learned anomaly detector (n-gram AUC 0.9999→0.50)**, not next-step. Headline anomaly/likelihood, not the all-positions next-step drop. — (track-industrial-ai)
- 2026-05-30 — **Strategy locked:** pretrained-LLM substrate + RL-reasoning (idea #1) + agentic copilot (idea #2), parallelized (3–4 ppl, worktrees). Plan = **spine + spike + demo**; from-scratch token model dropped. Measured: n-gram next-step top-1 ~0.80 ID → **~0.35–0.50 LOFO-OOD** (the headline); `validate_sequence` is a free perfect verifier (Task-3 oracle + GRPO reward + negative/explanation factory); report edit-dist not exact-match. — (track-industrial-ai)
- 2026-05-30 — **Data factory shipped:** `zo_train/grammar.py` (single import for the vendored validator + rule sets) + `zo_train/datagen.py` (splits incl. LOFO; verified rule-negatives **+ free repairs** at 94–100%; correct-by-construction explanations/CoT; LLM text framing). Shared corpus via new `ZO_DATA_DIR` (`paths.generated_data_dir`) + `ZO_EXPERIMENTS_DIR` per-worktree → one dashboard, one corpus. — (track-industrial-ai, stack)
- 2026-05-29 — Made `data/industrial-infineon/` a **byte-faithful 1:1 mirror** of upstream `tracks/industrial-infineon/`: moved the CSVs/grammar/`generate_sequences.py` into a `training_data/` subfolder (git renames) and added the 3 track-level docs (`README.md`, `Track_industrial{,_en}.md`). `diff -r` clean; 1000 seqs/family intact. No code referenced the old flat paths. Staged, not committed. — (track-industrial-ai)
- 2026-05-29 — **Resource budget (team lead):** 4 GPUs/node (A100 **64 GB VRAM** each), up to **512 GB RAM/node**, and the reservation `s_tra_ncc` covers **up to 4 nodes** → max **16 GPUs**. (Corrects an earlier misread of deck slide 92 — the reservation is NOT single-node.) — (cluster)
- 2026-05-29 — Reviewed the official Leonardo onboarding deck (pp. 80–96); cluster.md matches. **Wired the cluster plumbing**: `submit.py`/`train.sbatch.j2` now emit `--reservation` (any node count), `--gpus-per-task=N`, fair-share `--mem=120×N`/`--cpus=8×N` (auto-derived), proxy export, and `uv sync --extra gpu --offline` (pre-stage on a login node first). Verified by dry-run. — (cluster)
- 2026-05-29 — Created root `.env` (gitignored, real Leonardo creds) + refreshed `.env.example` (placeholders only). SSH login needs no setup (no 2FA); automating `zo-cluster submit` needs a one-time `ssh-copy-id` (submit.py uses passwordless ssh/scp). — (cluster, stack)
- 2026-05-29 — Vendored the track + submission folders into the repo: `data/industrial-infineon/` (3×1,000 validated seqs ~21MB, grammar, `generate_sequences.py`) + `docs/submission/`. Verified data loads & validates. **`eval_metrics.py` + `judging/rubrics.md` are NOT upstream** — expect them at kickoff. — (track-industrial-ai, hackathon)
- 2026-05-29 — Committed to **Industrial AI (Infineon)** track = small-vocab sequence modeling (next-step / completion / anomaly over fab steps), NOT chat-LLM eval. Full spec captured. — (track-industrial-ai)
- 2026-05-29 — Leonardo: plain SSH no-2FA to `login0{1,2,5,7}-ext.leonardo.cineca.it`; SLURM partition `boost_usr_prod` + reservation `s_tra_ncc`; **Pixi + Singularity** (no Docker/uv on compute); `$SCRATCH` for big files; **no internet on compute nodes** (download on login, or low-bw proxy). — (cluster)
- 2026-05-29 — Submission (Sun 10:00 via Tally): **public + MIT-licensed** repo, root `LICENSE` + `README.md` + `REPORT.md` + `requirements.txt`, **no secrets in git**; ≤10-slide PDF + ≤2-min demo video. Dashboard is a rewarded bonus on our track. — (hackathon, track-industrial-ai)
- 2026-05-29 — Run `mise trust` once per machine after cloning, or mise blocks `node`/`uv` (npm install fails). — (stack)
- 2026-05-29 — Root `pyproject.toml` must depend on all members or `uv sync` installs nothing; `gpu` extra re-exposed at root so `uv sync --extra gpu` works. — (stack)
- 2026-05-29 — Initial scaffold: uv workspace, file-based run registry, local-light/cluster-heavy split via `[gpu]` extra, `--dry-run` path that simulates metrics without torch. — (stack, decisions/2026-05-29-initial-architecture)

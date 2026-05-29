# Zero One Hackathon

- **Event:** Zero One Hackathon — https://zero-one.lumos-consulting.at
- **Where:** Vienna.
- **When:** Starts the evening of **2026-05-29**, runs through **2026-05-31**.
- **Team:** 2 people.
- **Theme (our read):** real model work — **finetuning / training / RL / agentifying** a model.
  Explicitly *not* prompt engineering. Build something that involves training or measurably changing
  a model's behavior.
- **Compute:** GPU access on the **Leonardo (CINECA)** supercomputer (A100s) via SLURM. See
  [cluster.md](cluster.md).

## To confirm on-site (then update this file)
- Exact judging criteria — what actually scores points? (demo? metrics? novelty?)
- Allowed / provided base models and whether weights are pre-cached on the cluster.
- Cluster credentials, account, partition, QOS, and per-team GPU/time budget.
- Submission format + deadline for the final demo.
- Any dataset / licensing constraints.

## Strategy notes
- The fastest path to a working demo: pick a **small base model** (e.g. Qwen2.5-0.5B/1.5B), a narrow
  task, SFT or GRPO it, and **show a before/after metric** via the eval harness on the dashboard.
- Use `--dry-run` + worktrees to develop two experiments in parallel without waiting on the cluster.
- Keep the story tight: one model, one measurable improvement, one clean dashboard view.

## Append below as you learn (rules, schedule, contacts)
- (fill in after kickoff)

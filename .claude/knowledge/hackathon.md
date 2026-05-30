# Zero One Hackathon — logistics, rules, judging

- **Event:** Zero One Hack — https://zero-one.lumos-consulting.at · docs:
  https://docs.zero-one.lumos-consulting.at/getting-started/welcome/
- **Where / when:** Vienna, **29–31 May 2026**. 36 hours, ~120 selected builders, €10K+ prize pool,
  3 tracks. Theme: **real model training on European supercompute — not prompt engineering, not
  wrappers.** Fine-tune/train/deploy real models on Leonardo (64× A100).
- **Team:** 2 people.
- **Organizers:** Lumos Consulting + AI Factory Austria (AI:AT). Comms via **Discord** (per-track
  channels) + on-site Lumos team / front desk. Emergency: Lumos +43 660 3482771, AI:AT +43 676 5118499.

## Our track
**Industrial AI (Infineon)** — "Learning & Benchmarking Process Logic" (sequence modeling of
semiconductor fab steps). Full details in **[track-industrial-ai.md](track-industrial-ai.md)**.
On-site mentor: **Simeon Harrison**. The other two tracks (for context only): **Insurance (UNIQA)** —
conversion-coach for a health-insurance funnel; **Forecasting (Sybilion)** — decision agent on top of
forecasting APIs.

## Schedule (key moments)
- **Fri 29 May:** 18:00 registration · 19:30 kickoff · 20:30 case reveal · **22:00 hack begins**.
- **Sat 30 May:** 11:00 panel · 13:00–17:00 mentoring · 14:00 HPE HPC talk · meals throughout.
- **Sun 31 May:** **10:00 submission deadline** · 13:30 final pitches · 14:45 results · 16:30 venue close.

## Submission (by Sun 10:00, via a single Tally form)
Four form fields: **team name**, **public repo URL**, **slides PDF** (≤10 slides, 3-min pitch),
**demo video** (MP4 1080p+audio or unlisted link, **hard ≤2 min**). Tally timestamp = official time.
**Submit by ~09:45** — Tally can flake; you can re-submit until 10:00.

The **repository** must:
- Be **public** at submission time (no private repos / temporary tokens).
- Be **MIT-licensed** — a `LICENSE` file at root with team as copyright holders.
- Have a root **`README.md`** (honest setup/run instructions — jury clones & runs it) and a root
  **`REPORT.md`** (required; the jury reads it carefully).
- Include a **`requirements.txt`** or equivalent manifest.
- **Run from a clean checkout** and contain **NO secrets** in git or history (jury checks this).

`REPORT.md` sections (2–4 pages, from `REPORT_TEMPLATE.md`): Team · **TL;DR** · **Problem** (specific,
not "improve X") · **Approach** (3–5 bullets incl. where it runs) · **How to run it** (exact commands) ·
**Results** (headline metric + baseline comparison + per-family breakdown; paste `eval_metrics.py`
scores) · **What worked** · **What didn't** · **Next 36 h** (concrete) · **Credits & dependencies**
(libs+versions, models, APIs, **AI coding tools used**, datasets+licenses) · **A note on honesty**
(disclose anything mocked/hardcoded — the jury asks in Q&A). Template + spec are vendored at
**`docs/submission/REPORT_TEMPLATE.md`** and **`docs/submission/SUBMISSION.md`**.

Repo-layout conventions the template assumes: eval outputs + raw scores in **`extras/results/`**,
optional architecture sketch in **`extras/`**. (Minor inconsistency: SUBMISSION.md says `REPORT.md` at
**repo root**, REPORT_TEMPLATE.md's header says `/submissions/{team}/REPORT.md` — go with **root**, it's
the authoritative checklist.) Note `judging/rubrics.md` is linked from SUBMISSION.md but **absent
upstream** — only the general rubric below is published. Industrial AI: [docs/track-industrial-sources.md](../../docs/track-industrial-sources.md); HF **`XCombinator`**; W&B **`XCombinator/XCombinator`**.

> **Action items this implies for us:** confirm the root `LICENSE` is MIT (it exists — verify text),
> write a root **`REPORT.md`** from the vendored template, ensure a **`requirements.txt`**-equivalent
> exists (we use uv — also emit a `requirements.txt`), put eval outputs in **`extras/results/`**, and
> double-check no secrets land in the public repo (incl. the Leonardo proxy password — keep it out;
> see [cluster.md](cluster.md)).

## What the judges reward (general rubric)
1. A **working artifact** that actually runs.
2. **Honest, reproducible evaluation** with real numbers.
3. **Visible technical choices** — what you decided and why.
4. **Genuine use of infrastructure** — Leonardo, the partner data/API.
5. **No basic LLM wrappers** — real engineering underneath.
> "Polish does not beat substance. A rough demo with strong results wins over a slick demo with no
> measurement." A good 2-min demo shows: the problem (15 s), the solution running live, **one
> concrete result with a number/comparison**, and the reasoning visible.

Track-specific deliverables live in the repo and are referenced from the REPORT — for ours see
[track-industrial-ai.md](track-industrial-ai.md) ("Track-specific repo deliverables").

## Judges & mentors (names that may appear)
Judges incl. Simeon Harrison (AI:AT), Miguel Peixoto (Sybilion, ML eng), Philipp Omenitsch (sequel
CTO), René Fabien de Montmorency (Infineon), Marcel Moosbrugger, Alexander Spreckelsen, et al.
Mentors incl. Simeon Harrison (**our track**), Miguel Peixoto, Alexander Sing (Accenture),
Johannes Oster (Innovatic). HPE HPC talk by Iveta Lohovska (HPE CTO AI & HPC).

## Access & infra (see cluster.md for the details)
GPU = **Leonardo (CINECA)** via plain SSH (no 2FA at the event), SLURM reservation `s_tra_ncc`,
**Pixi + Singularity**, no internet on compute nodes. Wi-Fi `Featherless` (personal creds via QR).
Sybilion provides an API for the forecasting track (not ours).

## Rules of conduct
Wear your tag · don't break/burn/steal anything · be nice · **no alcohol** · hack responsibly.

## Append below as you learn (corrections, room/contact specifics)
- (fill in after kickoff)

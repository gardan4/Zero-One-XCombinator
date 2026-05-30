#!/usr/bin/env python3
"""Build per-size INSTRUCT corpora (instruct_all.jsonl) for the data-scaling study.

Mirrors ``datagen.build_all``'s instruct rows on a SUBSAMPLE of N sequences/family, in the
**unified JSON format** (numbered input, ``{"reasoning":…, "steps":[…]}`` /
``{"reasoning":…, "valid":…, "rule":…}`` assistant labels via ``zo_train.prompts``). Small sizes
subsample the vendored pool; the large size GENERATES fresh valid routes.

Per training sequence: 1 anomaly(VALID) + 2 completion(60/80) + up to 4 next-step, PLUS ~1 balanced
INVALID anomaly negative per sequence (so the model learns to FLAG violations, not always say valid).
Output: data/generated_scale/<N>/instruct_all.jsonl. The scaling study is report-only (W&B / REPORT) —
the copilot dashboard panel was removed upstream.

Run:  uv run python scripts/build_scaling_corpora.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from zo_train.datagen import ALL_RULES, anomaly_example, make_negative, nextstep_example
from zo_train.fab import read_sequences
from zo_train.grammar import generate_dataset
from zo_train.prompts import PromptItem, build_instruct_messages, build_json_completion_completion

OUT = Path("data/generated_scale")
FAMILIES = ["MOSFET", "IGBT", "IC"]
SIZES = [(100, "sub"), (300, "sub"), (800, "sub"), (2000, "gen")]  # sequences per family


def rows_for(fam: str, seqs, rng: random.Random) -> list[dict]:
    out: list[dict] = []
    seqs = [list(s) for s in seqs]
    for s in seqs:
        # anomaly — VALID
        ex_valid = anomaly_example(fam, list(s), True)
        out.append({
            "messages": build_instruct_messages(
                "anomaly", PromptItem(fam, sequence=list(s)), ex_valid["completion"].lstrip()
            ),
            "family": fam, "task": "anomaly",
        })
        # completion — 60/80 cuts
        for frac in (0.6, 0.8):
            cut = int(len(s) * frac)
            comp = build_json_completion_completion(list(s[cut:]))
            out.append({
                "messages": build_instruct_messages(
                    "completion", PromptItem(fam, partial_sequence=list(s[:cut])), comp
                ),
                "family": fam, "task": "completion",
            })
        # next-step — up to 4 sampled positions
        if len(s) >= 3:
            for i in rng.sample(range(1, len(s)), k=min(4, len(s) - 1)):
                ex = nextstep_example(fam, s[:i], s[i])
                out.append({
                    "messages": build_instruct_messages(
                        "nextstep", PromptItem(fam, partial_sequence=list(s[:i])), ex["completion"].lstrip()
                    ),
                    "family": fam, "task": "nextstep",
                })
    # anomaly — balanced INVALID negatives (mirror datagen.build_all): ~1 per sequence → anomaly ≈ 50/50
    n_neg, made, guard, ti = len(seqs), 0, 0, 0
    while made < n_neg and guard < n_neg * 40:
        guard += 1
        rule = ALL_RULES[ti % len(ALL_RULES)]
        ti += 1
        neg = make_negative(rng.choice(seqs), rng, rule=rule)
        if not neg:
            continue
        ex = anomaly_example(fam, neg["steps"], False, neg["rules"])
        out.append({
            "messages": build_instruct_messages(
                "anomaly", PromptItem(fam, sequence=list(neg["steps"])), ex["completion"].lstrip()
            ),
            "family": fam, "task": "anomaly",
        })
        made += 1
    return out


for n, mode in SIZES:
    rng = random.Random(42)
    rows: list[dict] = []
    for fam in FAMILIES:
        seqs = read_sequences(fam)[:n] if mode == "sub" else generate_dataset(fam.lower(), n, seed=42)
        rows.extend(rows_for(fam, list(seqs), rng))
    d = OUT / str(n)
    d.mkdir(parents=True, exist_ok=True)
    (d / "instruct_all.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    print(f"size {n:>5} ({mode}): {len(rows)} instruct rows from {n} seqs/family")
print("done ->", OUT.resolve())

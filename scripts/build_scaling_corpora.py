#!/usr/bin/env python3
"""Build per-size INSTRUCT corpora (instruct_all.jsonl) for the data-scaling study.

Uses the shared instruct system-prompt stack (zo_train.prompts) so the scaling models match
leonardo_sft_fab_instruct.yaml's framing (system + task chat rows, completion-only loss).
Small sizes SUBSAMPLE the vendored pool; the large size GENERATES fresh valid routes
(the grammar space is billions+, so we are not capped at the vendored 1,000).

Per training sequence we emit the same row mix datagen uses: 1 anomaly(VALID) + 2 completion
(60/80 cuts) + up to 4 next-step. Output: data/generated_scale/<N>/instruct_all.jsonl.

Run:  uv run python scripts/build_scaling_corpora.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from zo_train.datagen import ALL_RULES, SEP, anomaly_example, make_negative, nextstep_example
from zo_train.fab import read_sequences
from zo_train.grammar import generate_dataset
from zo_train.prompts import PromptItem, build_instruct_messages

OUT = Path("data/generated_scale")
FAMILIES = ["MOSFET", "IGBT", "IC"]
SIZES = [(100, "sub"), (300, "sub"), (800, "sub"), (2000, "gen")]  # sequences per family


def rows_for(fam: str, seqs, rng: random.Random) -> list[dict]:
    out: list[dict] = []
    for s in seqs:
        s = list(s)
        out.append({
            "messages": build_instruct_messages("anomaly", PromptItem(fam, sequence=s), " VALID."),
            "family": fam, "task": "anomaly",
        })
        for frac in (0.6, 0.8):
            cut = int(len(s) * frac)
            out.append({
                "messages": build_instruct_messages(
                    "completion", PromptItem(fam, partial_sequence=s[:cut]), " " + SEP.join(s[cut:])
                ),
                "family": fam, "task": "completion",
            })
        if len(s) >= 3:
            for i in rng.sample(range(1, len(s)), k=min(4, len(s) - 1)):
                ex = nextstep_example(fam, s[:i], s[i])
                out.append({
                    "messages": build_instruct_messages(
                        "nextstep", PromptItem(fam, partial_sequence=s[:i]), ex["completion"]
                    ),
                    "family": fam, "task": "nextstep",
                })
    # Balanced INVALID anomaly negatives (mirror datagen.build_all) so the model learns to FLAG
    # rule violations instead of always answering VALID. ~1 negative per sequence → anomaly ≈ 50/50.
    pool = [list(s) for s in seqs]
    n_neg, made, guard, ti = len(pool), 0, 0, 0
    while made < n_neg and guard < n_neg * 40:
        guard += 1
        rule = ALL_RULES[ti % len(ALL_RULES)]
        ti += 1
        neg = make_negative(rng.choice(pool), rng, rule=rule)
        if not neg:
            continue
        ex = anomaly_example(fam, neg["steps"], False, neg["rules"])
        out.append({
            "messages": build_instruct_messages(
                "anomaly", PromptItem(fam, sequence=list(neg["steps"])), ex["completion"]
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

#!/usr/bin/env python3
"""Build per-size SFT-LM corpora for the data-scaling study ("does more data help?").

Small sizes SUBSAMPLE the existing 800/family corpus; the large size GENERATES fresh valid
routes with the grammar (its space is billions+, so we are NOT capped at the vendored 1,000).
Output: data/generated_scale/<N>/<FAMILY>_sft_lm.jsonl  — same row format as data/generated.

Run:  uv run python scripts/build_scaling_corpora.py
"""

from __future__ import annotations

import json
from pathlib import Path

from zo_train.datagen import lm_example
from zo_train.grammar import generate_dataset

EXISTING = Path("data/generated")
OUT = Path("data/generated_scale")
FAMILIES = ["MOSFET", "IGBT", "IC"]
SUBSAMPLE = [100, 300]  # per family, drawn from the existing 800
GENERATE = [2000]  # per family, fresh valid routes from the grammar


def _write(d: Path, fam: str, rows: list[str]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{fam}_sft_lm.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")


for n in SUBSAMPLE:
    for fam in FAMILIES:
        lines = (EXISTING / f"{fam}_sft_lm.jsonl").read_text(encoding="utf-8").splitlines()
        _write(OUT / str(n), fam, lines[:n])
        print(f"[subsample] {fam:6s} {n:>5}: {min(n, len(lines))} rows")

for n in GENERATE:
    for fam in FAMILIES:
        seqs = generate_dataset(fam.lower(), n, seed=42)
        rows = [json.dumps(lm_example(fam, s), ensure_ascii=False) for s in seqs]
        _write(OUT / str(n), fam, rows)
        print(f"[generate]  {fam:6s} {n:>5}: {len(seqs)} seqs")

print("done ->", OUT.resolve())

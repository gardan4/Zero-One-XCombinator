"""Build a unified instruction-tuning corpus (prompt/completion) for the 3 graded tasks + step
reasoning, drawn from the **train** split (eval uses the held-out test split).

Why prompt/completion (not the plain ``*_sft_lm.jsonl`` text field): an instruction-tuned model is
*promptable* — we can ask it for exactly the output each graded task needs, in the right format, and
optionally a justification. ``trl`` applies **completion-only loss** (loss on the answer, not the
prompt) when the dataset has ``prompt``/``completion`` columns and no ``text`` field — see
``sft.py`` (``completion_only_loss``). Uniform two-column schema so multiple families load as one set.

Tasks emitted (tagged via ``task`` for inspection; trl only reads prompt+completion):
  - ``nextstep``      — "Next process step?" → the next step
  - ``nextstep-cot``  — same, "Briefly justify." → step + a verifier-checked justification
  - ``completion``    — "Complete the remaining steps" → the rest of the route (pipe-joined)
  - ``anomaly``       — "Is this valid? name the violated rule" → VALID / INVALID + explanation

CLI:  python -m zo_train.instruct_data build --families MOSFET,IGBT,IC
      python -m zo_train.instruct_data build --families IGBT,IC --out data/generated/instruct_lofo-mosfet.jsonl
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, help="Build instruction-tuning corpus (prompt/completion).")


def _pc(prompt: str, completion: str, task: str, family: str) -> dict:
    return {"prompt": prompt, "completion": completion, "task": task, "family": family}


def examples_for_family(fam: str, rng: random.Random, n_seqs: int, reason_frac: float) -> list[dict]:
    from zo_train import datagen as D
    from zo_train.fab import read_sequences

    sp = D.make_splits()
    seqs = read_sequences(fam)
    train_idx = list(sp["per_family"][fam]["train"])[:n_seqs]
    rows: list[dict] = []
    for idx in train_idx:
        steps = seqs[idx]
        n = len(steps)
        if n < 6:
            continue
        # next-step (bare + reasoned) at two interior positions
        for frac in (0.4, 0.7):
            i = max(1, int(n * frac))
            if i >= n:
                continue
            prefix, target = steps[:i], steps[i]
            ns = D.nextstep_example(fam, prefix, target)
            rows.append(_pc(ns["prompt"], ns["completion"], "nextstep", fam))
            if rng.random() < reason_frac:
                j = D.justify_next_step(steps, i)
                p = ns["prompt"].replace("Next process step?", "Next process step? Briefly justify.")
                rows.append(_pc(p, f" {target}. {j['justification']}", "nextstep-cot", fam))
        # completion at 60%
        cut = max(1, int(n * 0.6))
        if cut < n:
            ce = D.completion_example(fam, steps[:cut], steps[cut:])
            rows.append(_pc(ce["prompt"], ce["completion"], "completion", fam))
        # anomaly: the valid route + one explained violation
        av = D.anomaly_example(fam, steps, True)
        rows.append(_pc(av["prompt"], av["completion"], "anomaly", fam))
        neg = D.make_negative(steps, rng)
        if neg:
            ai = D.anomaly_example(fam, neg["steps"], False, rules=neg.get("rules"), explain=True)
            rows.append(_pc(ai["prompt"], ai["completion"], "anomaly", fam))
    return rows


@app.command()
def build(
    families: str = typer.Option("MOSFET,IGBT,IC", help="Comma-separated train families."),
    out: str = typer.Option(None, help="Output .jsonl (default data/generated/instruct_<slug>.jsonl)."),
    n_seqs: int = typer.Option(150, help="Sequences per family to expand into examples."),
    reason_frac: float = typer.Option(0.5, help="Fraction of next-step examples that include a justification."),
    seed: int = typer.Option(42),
) -> None:
    from zo_common.paths import repo_root

    fams = [f.strip().upper() for f in families.split(",") if f.strip()]
    rng = random.Random(seed)
    rows: list[dict] = []
    for fam in fams:
        rows.extend(examples_for_family(fam, rng, n_seqs, reason_frac))
    rng.shuffle(rows)

    slug = "-".join(f.lower() for f in fams)
    out_path = Path(out) if out else (repo_root() / "data" / "generated" / f"instruct_{slug}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    hist = dict(Counter(r["task"] for r in rows))
    typer.echo(f"wrote {len(rows)} examples → {out_path}")
    typer.echo(f"  families={fams}  by_task={hist}")


if __name__ == "__main__":  # pragma: no cover
    app()

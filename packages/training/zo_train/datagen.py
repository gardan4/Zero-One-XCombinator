"""Data factory for the Industrial AI (Infineon) track — GPU-free, deterministic, shared.

Turns the shipped 3,000 validated sequences + the grammar/validator into every dataset
the training / eval / agent streams need. Run once; every stream pulls from the identical
frozen corpus under ``generated_data_dir()`` (env ``ZO_DATA_DIR``).

What it produces
----------------
- **splits** — per-family train/val/test + **leave-one-family-out (LOFO)** folds (the OOD proxy).
- **negatives** — rule-violating sequences, each LABELED with the rule(s) it breaks (by the
  verifier, so labels are ground truth) and carrying its own **repair** (the valid original).
- **explanations / CoT** — natural-language rationales that are *correct by construction*:
  anomaly explanations come straight from ``validate_sequence``'s own ``Violation``; next-step
  justifications are checked by a counterfactual re-validation. We never emit a wrong rationale.
- **text examples** — next-step / completion / anomaly framed as LLM prompts (family-conditioned,
  optional NL augmentation), ready for SFT/GRPO.

Stdlib + the vendored grammar only — imports fine on a laptop, no torch.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

from zo_common.paths import generated_data_dir

from zo_train import grammar
from zo_train.fab import FAMILIES, default_data_dir, read_sequences

SEP = " | "  # how steps are joined in a prompt (matches the eval pipe format)
DEVELOP_STEPS = frozenset({"DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"})

# Rules we mint targeted negatives for (one perturbation operator each, below).
ALL_RULES = [
    "RULE_DEP_NO_CLEAN",
    "RULE_ETCH_NO_MASK",
    "RULE_METAL_ETCH_NO_LITHO",
    "RULE_IMPLANT_NO_MASK",
    "RULE_CMP_NO_DEP",
    "RULE_LITHO_LEVEL_SKIP",
    "RULE_TEST_BEFORE_PASSIVATION",
    "RULE_SHIP_BEFORE_TEST",
    "RULE_BACKSIDE_BEFORE_PASSIVATION",
    "RULE_PAD_OPEN_BEFORE_DEP",
]


# --------------------------------------------------------------------------------------
# Natural-language step metadata (the under-used signal: descriptions + fab parameters)
# --------------------------------------------------------------------------------------


def load_descriptions(family: str, data_dir: Path | None = None) -> dict[str, dict[str, str]]:
    """``STEP -> {"description", "parameters"}`` from the Longdescr + parameters CSVs."""
    data_dir = data_dir or default_data_dir()
    out: dict[str, dict[str, str]] = {}

    ld = data_dir / f"{family}_Longdescr.csv"
    with ld.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            step = (row.get("STEP") or "").strip()
            if step:
                out.setdefault(step, {"description": "", "parameters": ""})
                out[step]["description"] = (row.get("DESCRIPTION") or "").strip()

    pp = data_dir / f"{family}_longdescription_parameters.csv"
    if pp.exists():
        with pp.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                step = (row.get("STEP") or "").strip()
                if not step:
                    continue
                out.setdefault(step, {"description": "", "parameters": ""})
                # The params header uses a non-ASCII hyphen ("FAB‑LEVEL"); match loosely.
                for key, val in row.items():
                    if key and "PARAMETER" in key.upper():
                        out[step]["parameters"] = (val or "").strip()
    return out


# --------------------------------------------------------------------------------------
# Splits (incl. leave-one-family-out for the OOD study)
# --------------------------------------------------------------------------------------


def make_splits(
    seed: int = 42,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    data_dir: Path | None = None,
) -> dict:
    """Deterministic per-family train/val/test indices + LOFO fold definitions."""
    rng = random.Random(seed)
    per_family: dict[str, dict[str, list[int]]] = {}
    for fam in FAMILIES:
        n = len(read_sequences(fam, data_dir))
        idx = list(range(n))
        rng.shuffle(idx)
        nt, nv = int(n * test_frac), int(n * val_frac)
        per_family[fam] = {
            "test": sorted(idx[:nt]),
            "val": sorted(idx[nt : nt + nv]),
            "train": sorted(idx[nt + nv :]),
        }
    lofo = {
        fam: {"train_families": [f for f in FAMILIES if f != fam], "test_family": fam}
        for fam in FAMILIES
    }
    return {"seed": seed, "per_family": per_family, "lofo": lofo}


# --------------------------------------------------------------------------------------
# Perturbation operators — each targets one rule. Always validated before use, so a
# label is only ever assigned by the verifier (operators that miss are simply discarded).
# --------------------------------------------------------------------------------------


def _move(steps: list[str], src: int, dst: int) -> list[str]:
    s = list(steps)
    item = s.pop(src)
    s.insert(dst - 1 if dst > src else dst, item)
    return s


def _idx(steps: list[str], pred) -> list[int]:
    return [i for i, s in enumerate(steps) if pred(s)]


def _op_del_clean_before_dep(steps, rng):
    deps = _idx(steps, lambda s: s in grammar.DEPOSITION_STEPS)
    rng.shuffle(deps)
    for d in deps:
        cleans = [j for j in range(max(0, d - 12), d) if steps[j] in grammar.CLEAN_STEPS]
        if cleans:
            drop = set(cleans)
            return [s for k, s in enumerate(steps) if k not in drop]
    return None


def _op_del_develop_before_etch(steps, rng):
    etches = _idx(steps, lambda s: s in grammar.ETCH_STEPS and s not in grammar.METAL_ETCH_STEPS)
    rng.shuffle(etches)
    for e in etches:
        devs = [j for j in range(max(0, e - 12), e) if steps[j] in DEVELOP_STEPS]
        if devs:
            drop = set(devs)
            return [s for k, s in enumerate(steps) if k not in drop]
    return None


def _op_del_litho_before_metal_etch(steps, rng):
    metals = _idx(steps, lambda s: s in grammar.METAL_ETCH_STEPS)
    rng.shuffle(metals)
    for m in metals:
        litho = [
            j
            for j in range(max(0, m - 15), m)
            if steps[j].startswith("EXPOSE LITHO LEVEL") or steps[j] in DEVELOP_STEPS
        ]
        if litho:
            drop = set(litho)
            return [s for k, s in enumerate(steps) if k not in drop]
    return None


def _op_del_opener_before_implant(steps, rng):
    implants = _idx(steps, lambda s: s in grammar.IMPLANT_STEPS)
    rng.shuffle(implants)
    for im in implants:
        openers = [j for j in range(max(0, im - 15), im) if steps[j] in grammar.IMPLANT_OPENER_STEPS]
        if openers:
            drop = set(openers)
            return [s for k, s in enumerate(steps) if k not in drop]
    return None


def _op_del_dep_before_cmp(steps, rng):
    cmps = _idx(steps, lambda s: s in grammar.CMP_STEPS)
    rng.shuffle(cmps)
    for c in cmps:
        deps = [j for j in range(max(0, c - 6), c) if steps[j] in grammar.FILL_STEPS]
        if deps:
            drop = set(deps)
            return [s for k, s in enumerate(steps) if k not in drop]
    return None


def _op_swap_litho_levels(steps, rng):
    aligns = [(i, s) for i, s in enumerate(steps) if s.startswith("ALIGN MASK LEVEL ")]
    levels = {}
    for i, s in aligns:
        try:
            levels.setdefault(int(s.rsplit(" ", 1)[1]), i)
        except ValueError:
            continue
    have = sorted(levels)
    for lo in have:
        if lo + 1 in levels:  # move the (lo+1) align to just before the lo align
            return _move(steps, levels[lo + 1], levels[lo])
    return None


def _op_move_test_before_cure(steps, rng):
    cure = next((i for i, s in enumerate(steps) if s == "CURE PASSIVATION"), None)
    if cure is None:
        return None
    tests = [i for i in range(cure + 1, len(steps)) if steps[i] in grammar.ELECTRICAL_TEST_STEPS]
    if tests:
        return _move(steps, rng.choice(tests), cure)
    return None


def _op_move_ship_before_sort(steps, rng):
    ship = next((i for i, s in enumerate(steps) if s == "SHIP LOT"), None)
    sort = next((i for i, s in enumerate(steps) if s == "WAFER SORT TEST"), None)
    if ship is not None and sort is not None and sort < ship:
        return _move(steps, ship, sort)
    return None


def _op_move_backside_before_cure(steps, rng):
    cure = next((i for i, s in enumerate(steps) if s == "CURE PASSIVATION"), None)
    bsm = next((i for i, s in enumerate(steps) if s == "DEPOSIT BACKSIDE METAL"), None)
    if cure is not None and bsm is not None and bsm > cure:
        return _move(steps, bsm, cure)
    return None


def _op_move_padopen_before_passivation(steps, rng):
    pad = next((i for i, s in enumerate(steps) if s in grammar.PAD_WINDOW_STEPS), None)
    dep = next(
        (i for i, s in enumerate(steps) if s in ("DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER")),
        None,
    )
    if pad is not None and dep is not None and pad > dep:
        return _move(steps, pad, dep)
    return None


OPERATORS = {
    "RULE_DEP_NO_CLEAN": _op_del_clean_before_dep,
    "RULE_ETCH_NO_MASK": _op_del_develop_before_etch,
    "RULE_METAL_ETCH_NO_LITHO": _op_del_litho_before_metal_etch,
    "RULE_IMPLANT_NO_MASK": _op_del_opener_before_implant,
    "RULE_CMP_NO_DEP": _op_del_dep_before_cmp,
    "RULE_LITHO_LEVEL_SKIP": _op_swap_litho_levels,
    "RULE_TEST_BEFORE_PASSIVATION": _op_move_test_before_cure,
    "RULE_SHIP_BEFORE_TEST": _op_move_ship_before_sort,
    "RULE_BACKSIDE_BEFORE_PASSIVATION": _op_move_backside_before_cure,
    "RULE_PAD_OPEN_BEFORE_DEP": _op_move_padopen_before_passivation,
}


def make_negative(steps: list[str], rng: random.Random, rule: str | None = None) -> dict | None:
    """Produce ONE verified rule-violating variant of a valid sequence.

    Returns ``{"steps", "rules", "target_rule", "repair"}`` or ``None`` if no operator
    applied. ``rules`` is whatever ``validate_sequence`` actually flags (ground truth);
    ``repair`` is the valid original (the free minimal fix the copilot is scored against).
    """
    items = [(rule, OPERATORS[rule])] if rule else list(OPERATORS.items())
    if not rule:
        rng.shuffle(items)
    for rname, op in items:
        cand = op(list(steps), rng)
        if cand is None:
            continue
        viol = grammar.validate_sequence(cand)
        if viol:
            return {
                "steps": cand,
                "rules": sorted({v.rule for v in viol}),
                "target_rule": rname,
                "repair": list(steps),
            }
    return None


# --------------------------------------------------------------------------------------
# Correct-by-construction explanations / CoT rationales
# --------------------------------------------------------------------------------------


def explain_violation(steps: list[str], descriptions: dict | None = None) -> str | None:
    """NL explanation of why a sequence is invalid — straight from the verifier. ``None`` if valid."""
    viol = grammar.validate_sequence(steps)
    if not viol:
        return None
    parts = []
    for v in viol:
        line = f"At step {v.step_index} ('{v.step_name}'), {v.rule} is violated: {v.description}"
        if descriptions and v.step_name in descriptions:
            ctx = descriptions[v.step_name].get("description")
            if ctx:
                line += f" [{v.step_name}: {ctx}]"
        parts.append(line)
    return " ".join(parts)


def justify_next_step(full_steps: list[str], i: int) -> dict:
    """Verified justification for why ``full_steps[i]`` is the next step.

    Counterfactual: remove it and re-validate. If the sequence becomes invalid, the step
    was load-bearing for process logic and we cite the exact rule(s); otherwise it's a
    routine backbone/optional step. Assumes ``full_steps`` is valid.
    """
    step = full_steps[i]
    cf = full_steps[:i] + full_steps[i + 1 :]
    viol = grammar.validate_sequence(cf)
    if viol:
        rules = sorted({v.rule for v in viol})
        return {
            "step": step,
            "rule_relevant": True,
            "rules": rules,
            "justification": (
                f"'{step}' is required here — removing it would violate "
                f"{', '.join(rules)} downstream."
            ),
        }
    return {
        "step": step,
        "rule_relevant": False,
        "rules": [],
        "justification": f"'{step}' is a routine backbone/optional step, not forced by a process-logic rule here.",
    }


# --------------------------------------------------------------------------------------
# Text framing for the LLM (prompt/completion pairs)
# --------------------------------------------------------------------------------------


def _nl_block(steps, descriptions, nl: bool) -> str:
    if not nl or not descriptions:
        return ""
    seen, lines = set(), []
    for s in steps:
        if s in seen or s not in descriptions:
            continue
        seen.add(s)
        d = descriptions[s].get("description")
        if d:
            lines.append(f"- {s}: {d}")
    return ("\nStep reference:\n" + "\n".join(lines) + "\n") if lines else ""


def nextstep_example(family, prefix, target, descriptions=None, nl=False) -> dict:
    p = (
        f"Product family: {family}{_nl_block(prefix[-8:], descriptions, nl)}\n"
        f"Process so far: {SEP.join(prefix)}\n\nNext process step?"
    )
    return {"prompt": p, "completion": " " + target, "family": family}


def lm_example(family, steps, descriptions=None, nl=False) -> dict:
    """ONE full-sequence SFT example — the autoregressive LM learns every next-step at once.

    This is the primary SFT corpus (one row per sequence), NOT a per-position expansion.
    Kept clean (no inline NL) so plain ``text``-field SFT doesn't learn to emit descriptions;
    do NL augmentation via the prompt/completion framings where completion-only loss is safe.
    """
    return {"text": f"Product family: {family}\nProcess sequence: {SEP.join(steps)}", "family": family}


def completion_example(family, prefix, rest, descriptions=None, nl=False) -> dict:
    p = (
        f"Product family: {family}\n"
        f"Partial process sequence: {SEP.join(prefix)}\n\n"
        f"Complete the remaining steps in order:"
    )
    return {"prompt": p, "completion": " " + SEP.join(rest), "family": family}


def anomaly_example(family, steps, is_valid, rules=None, descriptions=None, explain=False) -> dict:
    p = (
        f"Product family: {family}\n"
        f"Process sequence: {SEP.join(steps)}\n\n"
        f"Is this a valid process sequence? If not, name the violated rule."
    )
    if explain and not is_valid:
        why = explain_violation(steps, descriptions) or ""
        comp = f" INVALID. {why}"
    elif is_valid:
        comp = " VALID."
    else:
        comp = f" INVALID. Rule(s): {', '.join(rules or [])}."
    return {"prompt": p, "completion": comp, "family": family, "is_valid": int(bool(is_valid))}


# --------------------------------------------------------------------------------------
# Build the full corpus to disk
# --------------------------------------------------------------------------------------


def _write_jsonl(path: Path, rows) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def _fingerprint(out_dir: Path) -> str:
    """sha256 over the whole corpus (filename + bytes, sorted). Two machines that regenerate
    with the same code+seed get the SAME fingerprint — the cheap proof nobody drifted."""
    h = hashlib.sha256()
    for p in sorted(out_dir.glob("*.jsonl")) + [out_dir / "splits.json"]:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def build_all(
    out_dir: Path | None = None,
    seed: int = 42,
    neg_per_family: int = 400,
    data_dir: Path | None = None,
) -> dict:
    """Generate the whole corpus into ``out_dir`` (default ``generated_data_dir()``)."""
    out_dir = Path(out_dir) if out_dir else generated_data_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    splits = make_splits(seed=seed, data_dir=data_dir)
    (out_dir / "splits.json").write_text(json.dumps(splits, indent=2))

    manifest: dict = {"seed": seed, "families": list(FAMILIES), "counts": {}}

    for fam in FAMILIES:
        seqs = read_sequences(fam, data_dir)
        desc = load_descriptions(fam, data_dir)
        train_idx = splits["per_family"][fam]["train"]
        train_seqs = [seqs[i] for i in train_idx]

        # Negatives (balanced across rules) + their repairs + anomaly/CoT examples.
        negs, per_rule = [], {r: 0 for r in ALL_RULES}
        target_cycle, guard = 0, 0
        while len(negs) < neg_per_family and guard < neg_per_family * 40:
            guard += 1
            rule = ALL_RULES[target_cycle % len(ALL_RULES)]
            target_cycle += 1
            neg = make_negative(rng.choice(train_seqs), rng, rule=rule)
            if neg:
                negs.append(neg)
                for r in neg["rules"]:
                    per_rule[r] = per_rule.get(r, 0) + 1

        manifest["counts"][fam] = {
            "train_seqs": len(train_seqs),
            "negatives": len(negs),
            "per_rule": per_rule,
        }

        _write_jsonl(out_dir / f"{fam}_negatives.jsonl", negs)
        _write_jsonl(
            out_dir / f"{fam}_cot_anomaly.jsonl",
            (
                anomaly_example(fam, n["steps"], False, n["rules"], desc, explain=True)
                for n in negs
            ),
        )

        # SFT corpus: ONE full-sequence example per training sequence (the LM learns every
        # next-step at once — no wasteful per-position expansion).
        manifest["counts"][fam]["sft_lm_examples"] = _write_jsonl(
            out_dir / f"{fam}_sft_lm.jsonl",
            (lm_example(fam, s) for s in train_seqs),
        )
        # Completion SFT/eval at the same 60/80 cuts the organizers use.
        comp = (
            completion_example(fam, s[: int(len(s) * frac)], s[int(len(s) * frac) :])
            for s in train_seqs
            for frac in (0.6, 0.8)
        )
        manifest["counts"][fam]["completion_examples"] = _write_jsonl(
            out_dir / f"{fam}_sft_completion.jsonl", comp
        )
        # Sampled next-step eval from VAL sequences (internal metric; real eval = kickoff CSVs).
        val_seqs = [seqs[i] for i in splits["per_family"][fam]["val"]]
        ns_eval = [
            nextstep_example(fam, s[:i], s[i])
            for s in val_seqs
            if len(s) >= 3
            for i in rng.sample(range(1, len(s)), k=min(8, len(s) - 1))
        ]
        manifest["counts"][fam]["eval_nextstep_examples"] = _write_jsonl(
            out_dir / f"{fam}_eval_nextstep.jsonl", ns_eval
        )

    manifest["fingerprint"] = _fingerprint(out_dir)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# --------------------------------------------------------------------------------------
# Smoke test (manual): verify operators fire, explanations are correct, data is sane.
# --------------------------------------------------------------------------------------


def smoke() -> None:  # pragma: no cover - manual
    import pprint

    rng = random.Random(0)
    print("=== per-rule negative yield (200 tries each, MOSFET train) ===")
    seqs = read_sequences("MOSFET")
    for rule in ALL_RULES:
        hits = sum(
            1 for _ in range(200) if make_negative(rng.choice(seqs), rng, rule=rule) is not None
        )
        print(f"  {rule:34s} {hits/200:5.0%}")

    print("\n=== sample anomaly explanation (verified) ===")
    neg = None
    while neg is None:
        neg = make_negative(rng.choice(seqs), rng, rule="RULE_DEP_NO_CLEAN")
    desc = load_descriptions("MOSFET")
    print(" ", explain_violation(neg["steps"], desc))
    print("  repair == valid original:", grammar.validate_sequence(neg["repair"]) == [])

    print("\n=== sample next-step justification (verified) ===")
    s = rng.choice(seqs)
    for i in range(1, len(s)):
        j = justify_next_step(s, i)
        if j["rule_relevant"]:
            print(" ", j["justification"])
            break

    print("\n=== descriptions loaded ===")
    print("  MOSFET steps with description:", len(load_descriptions("MOSFET")))
    print("\n=== splits ===")
    pprint.pp({k: {kk: len(vv) for kk, vv in v.items()} for k, v in make_splits()["per_family"].items()})


if __name__ == "__main__":  # pragma: no cover
    import sys

    if "--build" in sys.argv:
        m = build_all()
        print(f"corpus → {generated_data_dir()}  fingerprint={m['fingerprint']}")
    else:
        smoke()

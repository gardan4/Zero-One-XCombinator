"""Process-logic copilot endpoints — the interactive demo surface.

Everything here is powered by the *symbolic* fab-route verifier (``validate_sequence``), so it is
instant and GPU-free: paste a recipe, get rule-violation detection + a plain-English explanation +
a one-step repair. This is the "does it respect process logic" story made tangible, and it doubles
as the ground-truth oracle the learned model is measured against.

Lazy-imports ``zo_train`` (grammar / datagen / fab) inside handlers: those modules are pure Python
(no torch), and live in the same uv-workspace venv as the backend.
"""

from __future__ import annotations

import random
from functools import lru_cache
from typing import Any

from pydantic import BaseModel

router = APIRouter(tags=["copilot"])

FAMILIES = ("MOSFET", "IGBT", "IC")


def _grammar():
    from zo_train import grammar

    return grammar


def _datagen():
    from zo_train import datagen

    return datagen


# --- recipe validation (the core demo engine) -------------------------------------------------
class ValidateRequest(BaseModel):
    steps: list[str] | None = None
    # convenience: a single string, pipe- or newline-separated (what the editor posts).
    text: str | None = None


def _parse_steps(req: ValidateRequest) -> list[str]:
    if req.steps:
        return [s.strip() for s in req.steps if s and s.strip()]
    if req.text:
        raw = req.text.replace("\r", "").replace("\n", "|")
        return [s.strip() for s in raw.split("|") if s.strip()]
    return []


@router.post("/validate")
def validate(req: ValidateRequest) -> dict[str, Any]:
    """Run the symbolic verifier on a recipe → violations + explanation (empty list == valid)."""
    g, d = _grammar(), _datagen()
    steps = _parse_steps(req)
    viols = g.validate_sequence(steps) if steps else []
    return {
        "valid": len(viols) == 0 and len(steps) > 0,
        "n_steps": len(steps),
        "n_violations": len(viols),
        "violations": [
            {
                "rule": v.rule,
                "description": v.description,
                "step_index": v.step_index,
                "step_name": v.step_name,
            }
            for v in viols
        ],
        "explanation": (d.explain_violation(steps) if viols else None),
        "steps": steps,
    }


# --- step vocabulary (the ~90-token process grammar) ------------------------------------------
@lru_cache(maxsize=1)
def _grammar_payload() -> dict[str, Any]:
    g = _grammar()
    cats = {
        a[: -len("_STEPS")].replace("_", " ").title(): sorted(getattr(g, a))
        for a in dir(g)
        if a.endswith("_STEPS") and isinstance(getattr(g, a), (set, frozenset))
    }
    allsteps = sorted({s for v in cats.values() for s in v})
    return {
        "categories": cats,
        "steps": allsteps,
        "n_steps": len(allsteps),
        "n_categories": len(cats),
    }


@router.get("/grammar")
def grammar() -> dict[str, Any]:
    return _grammar_payload()


# --- seed examples (valid recipes + canned violations to demo from) ---------------------------
@lru_cache(maxsize=8)
def _examples(family: str) -> dict[str, Any]:
    g, d = _grammar(), _datagen()
    from zo_train import fab

    rng = random.Random(42)
    seqs = fab.read_sequences(family)
    valid = [list(s) for s in seqs[:3]]
    invalid: list[dict[str, Any]] = []
    seen: set[str] = set()
    i = 0
    while len(invalid) < 6 and i < 400:
        neg = d.make_negative(seqs[(i * 7) % len(seqs)], rng)
        i += 1
        if not neg:
            continue
        rule = neg.get("target_rule")
        if rule in seen:
            continue
        seen.add(rule)
        viols = g.validate_sequence(neg["steps"])
        invalid.append(
            {
                "steps": neg["steps"],
                "target_rule": rule,
                "repair": neg.get("repair"),
                "explanation": d.explain_violation(neg["steps"]),
                "violation_index": viols[0].step_index if viols else None,
            }
        )
    return {"family": family, "valid": valid, "invalid": invalid}


@router.get("/examples")
def examples(family: str = "MOSFET") -> dict[str, Any]:
    fam = family.upper()
    return _examples(fam if fam in FAMILIES else "MOSFET")


# --- rule catalogue (the process-logic constraints the model must learn) ----------------------
@lru_cache(maxsize=1)
def _rules_catalog() -> list[dict[str, str]]:
    g, d = _grammar(), _datagen()
    from zo_train import fab

    rng = random.Random(7)
    cat: dict[str, str] = {}
    for family in FAMILIES:
        seqs = fab.read_sequences(family)
        for k in range(0, len(seqs), 11):
            neg = d.make_negative(seqs[k], rng)
            if not neg:
                continue
            for v in g.validate_sequence(neg["steps"]):
                cat.setdefault(v.rule, v.description)
        if len(cat) >= 12:
            break
    return [{"rule": r, "description": desc} for r, desc in sorted(cat.items())]


@router.get("/rules")
def rules() -> list[dict[str, str]]:
    return _rules_catalog()

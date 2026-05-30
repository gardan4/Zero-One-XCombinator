"""GPU-free tests for the Stream 3 diagnose/explain/repair copilot.

No model, no GPU, no server: the 3 tools are exercised directly, and the scripted episode runs with
a STUB repair brain (echoes the scenario's known-good repair). Validity is judged by the real
grammar verifier via the new ``success_type: validates`` path. These mirror the autonomous path's
trace shape, so a green run here proves the whole demo loop end to end.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from zo_agent import rollout, tools
from zo_agent.harness import ScenarioSuite, _judge, run_suite
from zo_train.datagen import make_negative
from zo_train.fab import read_sequences
from zo_train.grammar import validate_sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
REPAIR_YAML = REPO_ROOT / "packages" / "agent" / "scenarios" / "repair.yaml"
SEP = " | "


@pytest.fixture
def broken():
    """A provably-invalid recipe + its valid repair (deterministic)."""
    rng = random.Random(13)
    seqs = read_sequences("MOSFET")
    neg = None
    while neg is None:
        neg = make_negative(rng.choice(seqs), rng, rule="RULE_SHIP_BEFORE_TEST")
    # sanity on the fixture itself
    assert validate_sequence(neg["steps"])  # broken is invalid
    assert validate_sequence(neg["repair"]) == []  # repair is valid
    return neg


@pytest.fixture(autouse=True)
def _reset_repair_fn():
    """Never leak an injected repair brain across tests."""
    yield
    tools.set_repair_fn(None)


# --------------------------------------------------------------------------- tools (no model)


def test_validate_recipe_accepts_string_and_list(broken):
    pipe = SEP.join(broken["steps"])
    out_str = tools.validate_recipe(pipe)
    out_list = tools.validate_recipe(broken["steps"])
    assert out_str == out_list  # string and list inputs agree
    rows = json.loads(out_str)
    assert isinstance(rows, list) and rows
    r = rows[0]
    assert {"rule", "step_index", "step_name", "description"} <= set(r)
    assert r["rule"] in {v.rule for v in validate_sequence(broken["steps"])}


def test_validate_recipe_valid_returns_VALID(broken):
    assert tools.validate_recipe(broken["repair"]) == "VALID"
    assert tools.validate_recipe(SEP.join(broken["repair"])) == "VALID"


def test_validate_recipe_handles_tight_pipes(broken):
    # "A|B|C" (no surrounding spaces — submission separator) must parse too.
    tight = "|".join(broken["repair"])
    assert tools.validate_recipe(tight) == "VALID"


def test_explain_violation(broken):
    why = tools.explain_violation(SEP.join(broken["steps"]))
    assert "RULE_SHIP_BEFORE_TEST" in why
    assert tools.explain_violation(broken["repair"]) == "VALID — no violations."


def test_dispatch_never_raises():
    # dispatch must swallow bad args / unknown tools (rollout relies on this).
    assert tools.dispatch("validate_recipe", {}).startswith("error:")
    assert tools.dispatch("does_not_exist", {}).startswith("error:")


def test_suggest_repair_with_stub_validates(broken):
    """Stub returns the known-good repair → suggest_repair returns a valid recipe."""
    target = SEP.join(broken["repair"])
    tools.set_repair_fn(lambda steps, family: target)
    out = tools.suggest_repair(SEP.join(broken["steps"]), "MOSFET")
    assert validate_sequence(out.split("|")) == []
    # idempotent on an already-valid recipe
    assert validate_sequence(tools.suggest_repair(target, "MOSFET").split("|")) == []


def test_suggest_repair_loops_until_clean(broken):
    """First attempt still broken, second is the fix: the verifier-in-the-loop must retry."""
    good = SEP.join(broken["repair"])
    bad = SEP.join(broken["steps"])
    calls = {"n": 0}

    def flaky(steps, family):
        calls["n"] += 1
        return bad if calls["n"] == 1 else good

    tools.set_repair_fn(flaky)
    out = tools.suggest_repair(bad, "MOSFET", max_loops=3)
    assert calls["n"] >= 2  # it did not stop on the first (still-invalid) try
    assert validate_sequence(out.split("|")) == []


def test_suggest_repair_returns_concrete_sequence_when_unfixable(broken):
    """If the brain never fixes it, we still return a non-empty recipe (not a crash/empty)."""
    bad = SEP.join(broken["steps"])
    tools.set_repair_fn(lambda steps, family: bad)
    out = tools.suggest_repair(bad, "MOSFET", max_loops=2)
    assert out.strip()  # concrete sequence returned for inspection


# --------------------------------------------------------------------------- judge


def test_validates_judge(broken):
    assert _judge("validates", SEP.join(broken["repair"]), "") == 1.0
    assert _judge("validates", SEP.join(broken["steps"]), "") == 0.0
    # empty / no parseable steps must NOT count as success (validate_sequence([]) is vacuously [])
    assert _judge("validates", "", "") == 0.0
    assert _judge("validates", "lorem ipsum no steps here", "") == 0.0
    # existing judge types still work
    assert _judge("contains", "the answer is 42", "42") == 1.0
    assert _judge("numeric", "result: 103", "103") == 1.0


# --------------------------------------------------------------------------- scripted episode


def test_extract_recipe():
    recipe, fam = rollout.extract_recipe("Diagnose and fix this fab recipe (IGBT): A | B | C")
    assert recipe == "A | B | C"
    assert fam == "IGBT"
    # no family token, label prefix on the piped line
    recipe2, fam2 = rollout.extract_recipe("recipe: X | Y")
    assert recipe2 == "X | Y" and fam2 == ""


def test_run_episode_scripted_stub_repairs(broken):
    """Full scripted path with a stub repair brain → success under the validates judge."""
    target = SEP.join(broken["repair"])
    tools.set_repair_fn(lambda steps, family: target)
    task = f"Diagnose and fix this fab recipe (MOSFET): {SEP.join(broken['steps'])}"

    ep = rollout.run_episode_scripted(None, task, model="stub")
    # same trace dict shape as run_episode
    assert set(ep) == {"final", "steps", "tool_calls", "trace"}
    # trace ran validate → explain → repair → re-validate
    seq = [t["tool"] for t in ep["trace"]]
    assert seq == ["validate_recipe", "explain_violation", "suggest_repair", "validate_recipe"]
    assert ep["trace"][-1]["result"] == "VALID"  # final re-validation passes
    assert ep["tool_calls"] == 4
    # the judge agrees the final answer is a valid recipe
    assert _judge("validates", ep["final"], "") == 1.0


def test_run_episode_scripted_already_valid_short_circuits(broken):
    task = f"Diagnose and fix this fab recipe (MOSFET): {SEP.join(broken['repair'])}"
    ep = rollout.run_episode_scripted(None, task, model="stub")
    assert [t["tool"] for t in ep["trace"]] == ["validate_recipe"]
    assert ep["final"] == SEP.join(broken["repair"])
    assert _judge("validates", ep["final"], "") == 1.0


# --------------------------------------------------------------------------- suite (registry)


def test_run_suite_scripted_with_stub(tmp_path, monkeypatch):
    """run_suite over repair.yaml with a stub brain → 100% success, logged to the registry."""
    monkeypatch.setenv("ZO_EXPERIMENTS_DIR", str(tmp_path))
    suite = ScenarioSuite.from_yaml(REPAIR_YAML)
    assert len(suite.scenarios) >= 8
    assert all(sc.success_type == "validates" for sc in suite.scenarios)

    # Stub brain: echo each scenario's known-good repair (carried in success_value).
    by_task = {sc.task: sc.success_value for sc in suite.scenarios}

    def stub(steps, family):
        # find the scenario whose broken recipe contains these steps; fall back to any match
        joined = SEP.join(steps)
        for task, repair in by_task.items():
            if joined in task:
                return repair
        return joined

    tools.set_repair_fn(stub)
    res = run_suite(suite, model="stub", scripted=True)
    assert res["success_rate"] == 1.0, res
    assert res["n"] == len(suite.scenarios)

    # metrics landed in the registry
    from zo_common import registry

    rows = registry.read_metrics(res["run_id"])
    assert len(rows) == len(suite.scenarios)
    assert all(r["success"] == 1.0 for r in rows)


def test_repair_yaml_scenarios_are_provably_broken():
    """Every scenario's recipe really is invalid (so 'repair' is a meaningful task)."""
    suite = ScenarioSuite.from_yaml(REPAIR_YAML)
    rules_seen, fams_seen = set(), set()
    for sc in suite.scenarios:
        recipe, fam = rollout.extract_recipe(sc.task)
        steps = [s.strip() for s in recipe.split("|") if s.strip()]
        viol = validate_sequence(steps)
        assert viol, f"{sc.name}: recipe should be invalid"
        rules_seen.update(v.rule for v in viol)
        fams_seen.add(fam)
        # the bundled repair is valid
        assert validate_sequence([s.strip() for s in sc.success_value.split("|")]) == []
    assert len(rules_seen) >= 5  # spans multiple distinct rules
    assert fams_seen == {"MOSFET", "IGBT", "IC"}  # all three families represented

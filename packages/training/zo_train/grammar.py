"""Single source of truth for the track grammar + validator.

The track ships ``generate_sequences.py`` as a standalone script under the data dir
(not a Python package), so we load it by path exactly once and re-export its public
API plus the rule-vocabulary frozensets. Every stream (training / eval / agent /
data factory) imports from here so the rules can never drift between components::

    from zo_train.grammar import validate_sequence, generate_dataset, DEPOSITION_STEPS

``validate_sequence(steps) -> list[Violation]`` is a free, perfect verifier:
empty list == valid. ``Violation`` has ``.rule .description .step_index .step_name``.
"""

from __future__ import annotations

import importlib.util

from zo_common.paths import repo_root

_PATH = (
    repo_root()
    / "data"
    / "industrial-infineon"
    / "training_data"
    / "generate_sequences.py"
)

_spec = importlib.util.spec_from_file_location("zo_vendored_grammar", _PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover - defensive
    raise ImportError(f"Cannot load the vendored grammar module at {_PATH}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Re-export every public function + the UPPERCASE rule-vocabulary frozensets
# (DEPOSITION_STEPS, CLEAN_STEPS, ETCH_STEPS, IMPLANT_STEPS, CMP_STEPS, ...).
_FUNCS = {
    "validate_sequence",
    "generate_sequence",
    "generate_dataset",
    "read_csv_sequences",
    "write_csv",
    "estimate_combinatorics",
    "Violation",
}
_EXPORT = [n for n in dir(_mod) if n.isupper() or n in _FUNCS]
globals().update({n: getattr(_mod, n) for n in _EXPORT})

__all__ = _EXPORT

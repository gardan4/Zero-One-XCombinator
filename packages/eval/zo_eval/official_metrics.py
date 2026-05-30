"""Vendored organizer ``eval_metrics.py`` — exact scoring when ground truth is available.

Kickoff **inputs** live in ``data/industrial-infineon/eval/`` (600 valid + 987 anomaly rows).
Organizers hold **labels** for those IDs; we self-score locally via ``track_metrics.py`` + ``gold.json``
from held-out splits, or run this script when ground-truth CSVs are available (e.g. local proxy sets).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = _REPO_ROOT / "data" / "industrial-infineon" / "eval"
OFFICIAL_SCRIPT = EVAL_DIR / "eval_metrics.py"
KICKOFF_VALID = EVAL_DIR / "eval_input_valid.csv"
KICKOFF_ANOMALY = EVAL_DIR / "eval_input_anomaly.csv"


def run_official(
    task: str,
    ground_truth: str | Path,
    predictions: str | Path,
    *,
    valid_supplement: str | Path | None = None,
) -> int:
    """Invoke the vendored ``eval_metrics.py``; returns the subprocess exit code."""
    if not OFFICIAL_SCRIPT.is_file():
        raise FileNotFoundError(f"missing organizer script: {OFFICIAL_SCRIPT}")
    cmd = [
        sys.executable,
        str(OFFICIAL_SCRIPT),
        "--task",
        task,
        "--ground-truth",
        str(ground_truth),
        "--predictions",
        str(predictions),
    ]
    if valid_supplement:
        cmd.extend(["--valid-supplement", str(valid_supplement)])
    return subprocess.run(cmd, check=False).returncode

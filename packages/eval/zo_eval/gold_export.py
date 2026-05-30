"""Build organizer ground-truth CSVs from ``gold.json`` + eval inputs.

Used by ``zo-track self-check`` to run vendored ``eval_metrics.py`` on local labeled proxy sets
and confirm our ``track_metrics.py`` scores match the official script.
"""

from __future__ import annotations

import csv
from pathlib import Path

from zo_eval import submission as sub

STEP_SEP = sub.STEP_SEP


def export_valid_gt(gold: dict, valid_csv: str | Path, out_path: str | Path) -> Path:
    """``eval_set_valid.csv`` format: partial + ``NEXT_STEP`` + ``FULL_SEQUENCE``."""
    out_path = Path(out_path)
    inputs = {it.example_id: it for it in sub.read_valid_inputs(valid_csv)}
    rows = []
    for ex, it in inputs.items():
        if ex not in gold.get("next", {}):
            continue
        partial = it.partial_sequence
        remaining = gold["completion"][ex]
        rows.append(
            {
                "EXAMPLE_ID": ex,
                "FAMILY": it.family,
                "COMPLETION_FRACTION": it.completion_fraction,
                "PARTIAL_SEQUENCE": STEP_SEP.join(partial),
                "NEXT_STEP": gold["next"][ex],
                "FULL_SEQUENCE": STEP_SEP.join(partial + remaining),
            }
        )
    _write_dicts(
        out_path,
        [
            "EXAMPLE_ID",
            "FAMILY",
            "COMPLETION_FRACTION",
            "PARTIAL_SEQUENCE",
            "NEXT_STEP",
            "FULL_SEQUENCE",
        ],
        rows,
    )
    return out_path


def export_anomaly_gt(
    gold: dict,
    anomaly_csv: str | Path,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """Forbidden + valid-supplement CSVs for official anomaly scoring."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = {it.example_id: it for it in sub.read_anomaly_inputs(anomaly_csv)}
    forbidden, valid_sup = [], []
    for ex, info in gold.get("anomaly", {}).items():
        it = inputs.get(ex)
        if it is None:
            continue
        row = {
            "EXAMPLE_ID": ex,
            "FAMILY": it.family,
            "SEQUENCE": STEP_SEP.join(it.sequence),
        }
        if info.get("is_valid") == 0:
            row["VIOLATION_RULE"] = info.get("rule") or ""
            forbidden.append(row)
        else:
            valid_sup.append(row)
    forbidden_path = out_dir / "gt_anomaly_forbidden.csv"
    valid_path = out_dir / "gt_anomaly_valid_supplement.csv"
    _write_dicts(
        forbidden_path,
        ["EXAMPLE_ID", "FAMILY", "SEQUENCE", "VIOLATION_RULE"],
        forbidden,
    )
    _write_dicts(valid_path, ["EXAMPLE_ID", "FAMILY", "SEQUENCE"], valid_sup)
    return forbidden_path, valid_path


def export_all_ground_truth(
    gold: dict | str | Path,
    valid_csv: str | Path,
    anomaly_csv: str | Path,
    out_dir: str | Path,
) -> dict[str, Path]:
    """Export every GT file needed for ``eval_metrics.py`` on a labeled proxy set."""
    if not isinstance(gold, dict):
        gold = __import__("json").loads(Path(gold).read_text())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    valid_gt = export_valid_gt(gold, valid_csv, out_dir / "gt_valid.csv")
    forbidden, valid_sup = export_anomaly_gt(gold, anomaly_csv, out_dir)
    return {
        "valid": valid_gt,
        "anomaly_forbidden": forbidden,
        "anomaly_valid_supplement": valid_sup,
    }


def _write_dicts(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

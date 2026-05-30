"""Run vendored ``eval_metrics.py`` on exported ground truth + write captured output."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from zo_eval.gold_export import export_all_ground_truth
from zo_eval.official_metrics import run_official


def run_self_check(
    results_dir: str | Path,
    *,
    gold: dict | str | Path,
    valid_csv: str | Path,
    anomaly_csv: str | Path,
) -> tuple[Path, dict[str, str]]:
    """Export GT CSVs, score all three tasks with ``eval_metrics.py``, save transcript."""
    results_dir = Path(results_dir)
    gt_dir = results_dir / "ground_truth"
    paths = export_all_ground_truth(gold, valid_csv, anomaly_csv, gt_dir)

    lines: list[str] = []
    task_specs = [
        ("next-step", paths["valid"], results_dir / "nextstep.csv", None),
        ("completion", paths["valid"], results_dir / "completion.csv", None),
        (
            "anomaly",
            paths["anomaly_forbidden"],
            results_dir / "anomaly.csv",
            paths["anomaly_valid_supplement"],
        ),
    ]
    exit_codes: dict[str, int] = {}
    for task, gt_path, pred_path, supplement in task_specs:
        if not pred_path.exists():
            lines.append(f"=== {task}: SKIPPED (missing {pred_path.name}) ===\n")
            continue
        lines.append(f"=== {task} ===")
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            code = run_official(task, gt_path, pred_path, valid_supplement=supplement)
        exit_codes[task] = code
        lines.append(buf_out.getvalue().rstrip())
        if buf_err.getvalue().strip():
            lines.append("STDERR:")
            lines.append(buf_err.getvalue().rstrip())
        lines.append("")

    transcript = "\n".join(lines).strip() + "\n"
    out_path = results_dir / "official_scores.txt"
    out_path.write_text(transcript, encoding="utf-8")
    return out_path, {k: ("ok" if v == 0 else f"exit_{v}") for k, v in exit_codes.items()}


def run_self_check_cli(
    results_dir: str | Path,
    gold_path: str | Path,
    valid_csv: str | Path,
    anomaly_csv: str | Path,
) -> Path:
    """CLI helper: load gold.json and run self-check."""
    gold = __import__("json").loads(Path(gold_path).read_text())
    path, _ = run_self_check(
        results_dir,
        gold=gold,
        valid_csv=valid_csv,
        anomaly_csv=anomaly_csv,
    )
    return path

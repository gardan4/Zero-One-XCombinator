"""Load, promote, and manifest track-eval result directories."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from zo_eval import submission as sub

_REPO_ROOT = Path(__file__).resolve().parents[3]
EXTRAS_RESULTS = _REPO_ROOT / "extras" / "results"

ARTIFACT_NAMES = (
    "nextstep.csv",
    "completion.csv",
    "anomaly.csv",
    "metrics_report.json",
    "metrics_report.md",
    "proxy_report.json",
    "official_scores.txt",
    "manifest.json",
    "ground_truth",
)


def load_predictions_from_dir(results_dir: str | Path) -> dict:
    """Read submission CSVs back into predictor-shaped dicts (for rescore / self-check)."""
    results_dir = Path(results_dir)
    out: dict = {"nextstep": {}, "completion": {}, "anomaly": {}}
    ns_path = results_dir / "nextstep.csv"
    if ns_path.exists():
        for row in sub._norm_rows(ns_path):
            ex = row["EXAMPLE_ID"]
            ranks = [row.get(f"RANK_{k}", "").strip() for k in range(1, 6)]
            out["nextstep"][ex] = [r for r in ranks if r]
    cp_path = results_dir / "completion.csv"
    if cp_path.exists():
        for row in sub._norm_rows(cp_path):
            out["completion"][row["EXAMPLE_ID"]] = sub._split(row.get("PREDICTED_SEQUENCE", ""))
    an_path = results_dir / "anomaly.csv"
    if an_path.exists():
        for row in sub._norm_rows(an_path):
            ex = row["EXAMPLE_ID"]
            try:
                iv = int(float(row.get("IS_VALID", "1")))
            except (ValueError, TypeError):
                iv = 1
            sc_raw = row.get("SCORE", "").strip()
            try:
                sc = float(sc_raw) if sc_raw else None
            except ValueError:
                sc = None
            rule = row.get("PREDICTED_RULE", "").strip() or None
            out["anomaly"][ex] = {"is_valid": iv, "score": sc, "rule": rule}
    return out


def write_manifest(
    out_dir: Path,
    *,
    run_id: str,
    version: str,
    predictor: str,
    model_ref: str | None,
    eval_set: str | None,
    tags: list[str],
    valid_csv: str | None,
    anomaly_csv: str | None,
    gold: str | None,
    artifacts: list[str],
    train_run_id: str | None = None,
    notes: str | None = None,
    train_run: dict | None = None,
) -> Path:
    """Write ``manifest.json`` beside result CSVs for promote / audit."""
    manifest = {
        "run_id": run_id,
        "version": version,
        "predictor": predictor,
        "model_ref": model_ref,
        "eval_set": eval_set,
        "tags": tags,
        "train_run_id": train_run_id,
        "train_run": train_run,
        "notes": notes,
        "valid_csv": valid_csv,
        "anomaly_csv": anomaly_csv,
        "gold": gold,
        "artifacts": artifacts,
        "created_at": datetime.now(UTC).isoformat(),
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def promote_results(
    src: str | Path,
    slug: str,
    *,
    extras_root: Path | None = None,
) -> Path:
    """Copy graded outputs to ``extras/results/<slug>/`` and update the index."""
    src = Path(src)
    if not src.is_dir():
        raise FileNotFoundError(f"results dir not found: {src}")
    dest = (extras_root or EXTRAS_RESULTS) / slug
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ARTIFACT_NAMES:
        if name == "ground_truth":
            gt_src = src / "ground_truth"
            if gt_src.is_dir():
                gt_dest = dest / "ground_truth"
                if gt_dest.exists():
                    shutil.rmtree(gt_dest)
                shutil.copytree(gt_src, gt_dest)
                copied.append("ground_truth/")
            continue
        p = src / name
        if p.exists():
            shutil.copy2(p, dest / name)
            copied.append(name)
    index_path = (extras_root or EXTRAS_RESULTS) / "INDEX.json"
    index: dict = {}
    if index_path.exists():
        index = json.loads(index_path.read_text())
    manifest = {}
    if (dest / "manifest.json").exists():
        manifest = json.loads((dest / "manifest.json").read_text())
    try:
        rel_path = str(dest.relative_to(_REPO_ROOT)).replace("\\", "/")
    except ValueError:
        rel_path = str(dest)
    index[slug] = {
        "path": rel_path,
        "run_id": manifest.get("run_id"),
        "version": manifest.get("version"),
        "predictor": manifest.get("predictor"),
        "model_ref": manifest.get("model_ref"),
        "eval_set": manifest.get("eval_set"),
        "tags": manifest.get("tags", []),
        "train_run_id": manifest.get("train_run_id"),
        "notes": manifest.get("notes"),
        "artifacts": copied,
        "promoted_at": datetime.now(UTC).isoformat(),
    }
    index_path.write_text(json.dumps(index, indent=2))
    return dest

#!/usr/bin/env bash
# Pull a cluster-promoted scaling result down to the laptop and re-promote locally so the
# dashboard INDEX.json + extras/results/<slug> are updated. Usage: pull_promote_scale.sh <N>
# (N = training-size bucket: 100|300|800|2000). Run from the bold-banzai worktree root.
set -euo pipefail
N="${1:?usage: pull_promote_scale.sh <N>}"
SLUG="hf-sft-scale-${N}"
HOST="a08trd0f@login01-ext.leonardo.cineca.it"
RREPO="/leonardo/home/usertrain/a08trd0f/Zero-One-Philyr"
PW="$(python3 -c "print([l.split('=',1)[1].strip().strip(chr(34)).strip(chr(39)) for l in open('.env') if l.startswith('ZO_CLUSTER_PASSWORD=')][0])")"
STAGE="/tmp/scale-pull-${N}"
rm -rf "$STAGE" && mkdir -p "$STAGE"
echo "[pull] $RREPO/extras/results/$SLUG -> $STAGE"
pscp -batch -pw "$PW" -r "${HOST}:${RREPO}/extras/results/${SLUG}/*" "$STAGE/" 2>&1 | tail -2
echo "[check] metrics in staged results:"
python3 - "$STAGE" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
rep = json.loads((d / "metrics_report.json").read_text())
tasks = rep.get("tasks", {})
def pick(t):
    o = tasks.get(t, {})
    bf = o.get("by_family", {})
    return bf.get("overall") or next(iter(bf.values()), {})
ns, comp = pick("nextstep"), pick("completion")
print("  nextstep.top1   =", ns.get("top1"))
print("  completion.block_acc =", comp.get("block_acc"))
PY
echo "[promote] local zo-track promote -> extras/results/$SLUG + INDEX.json"
uv run zo-track promote -r "$STAGE" -s "$SLUG" 2>&1 | tail -3

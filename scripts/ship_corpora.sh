#!/usr/bin/env bash
# Stage regenerated corpora onto the cluster. The cluster sync (sync_code_to_cluster) only ships
# code paths — NOT data/ ("assumes data/models already staged") — so after any datagen/scaling
# regenerate, run this to push the JSON corpora the training jobs read. Run from worktree root.
set -euo pipefail
HOST="a08trd0f@login01-ext.leonardo.cineca.it"
R="/leonardo/home/usertrain/a08trd0f/Zero-One-Philyr"
PW="$(python3 -c "print([l.split('=',1)[1].strip().strip(chr(34)).strip(chr(39)) for l in open('.env') if l.startswith('ZO_CLUSTER_PASSWORD=')][0])")"

echo "[ship] canonical data/generated/instruct_all.jsonl"
plink -batch -ssh -pw "$PW" "$HOST" "mkdir -p ${R}/data/generated" >/dev/null 2>&1 || true
pscp -batch -pw "$PW" data/generated/instruct_all.jsonl "${HOST}:${R}/data/generated/instruct_all.jsonl" 2>&1 | tail -1

for N in 100 300 800 2000; do
  f="data/generated_scale/${N}/instruct_all.jsonl"
  [ -f "$f" ] || { echo "[ship] skip ${N} (no local corpus)"; continue; }
  echo "[ship] scale ${N}"
  plink -batch -ssh -pw "$PW" "$HOST" "mkdir -p ${R}/data/generated_scale/${N}" >/dev/null 2>&1 || true
  pscp -batch -pw "$PW" "$f" "${HOST}:${R}/data/generated_scale/${N}/instruct_all.jsonl" 2>&1 | tail -1
done

echo "[ship] verify cluster canonical format:"
plink -batch -ssh -pw "$PW" "$HOST" "wc -l ${R}/data/generated/instruct_all.jsonl; head -c 80 ${R}/data/generated/instruct_all.jsonl; echo"

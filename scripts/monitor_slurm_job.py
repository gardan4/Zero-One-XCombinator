#!/usr/bin/env python3
"""Poll a Leonardo SLURM job and print log tail every N seconds."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

from zo_train.cluster._remote import ssh_capture, ssh_target, cluster_repo_dir
from zo_train.cluster._slurm import ensure_cluster_env


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("job_id")
    p.add_argument("--run-id", default="")
    p.add_argument("--interval", type=float, default=10.0)
    p.add_argument("--max-polls", type=int, default=120)
    args = p.parse_args()

    ensure_cluster_env()
    target = ssh_target()
    if not target:
        sys.exit("Set ZO_CLUSTER_HOST and ZO_CLUSTER_USER in .env")
    repo = cluster_repo_dir()
    log_glob = f"{repo}/slurm_logs/zo-infer-*-{args.job_id}.out"
    results = ""
    if args.run_id:
        scratch = "/leonardo_scratch/large/usertrain/a08trd0f/zo-experiments"
        results = f"{scratch}/{args.run_id}/results"

    last_log = ""
    for i in range(args.max_polls):
        ts = datetime.now().strftime("%H:%M:%S")
        q = ssh_capture(target, f"squeue -j {args.job_id} -h -o '%T %M %R' 2>/dev/null || echo DONE")
        log_cmd = (
            f"log=$(ls {log_glob} 2>/dev/null | head -1); "
            'if [ -n "$log" ]; then tail -n 12 "$log"; else echo "(no log yet)"; fi'
        )
        log = ssh_capture(target, log_cmd)
        files = ""
        if results:
            files = ssh_capture(
                target,
                f"ls -1 {results} 2>/dev/null | paste -sd, - || echo '(no results yet)'",
            )
        print(f"[{ts}] poll {i + 1} | queue: {q} | files: {files}", flush=True)
        if log != last_log:
            print(log, flush=True)
            print("---", flush=True)
            last_log = log
        done = "DONE" in q
        if done and (
            "judge eval complete" in log
            or "CUDA out of memory" in log
            or "OutOfMemoryError" in log
            or "FAILED" in log
            or "Traceback" in log
        ):
            break
        if done and i >= 2:
            time.sleep(5)
            log = ssh_capture(target, f"log=$(ls {log_glob} 2>/dev/null | head -1); tail -n 25 \"$log\"")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] FINAL\n{log}\n---", flush=True)
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

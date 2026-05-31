"""Pull promoted hf-sft-scale-2000 from Leonardo and re-promote locally."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from zo_train.cluster._remote import load_dotenv, scp_download, ssh_target

load_dotenv()
repo = Path(__file__).resolve().parents[1]
target = ssh_target()
remote_dir = "/leonardo/home/usertrain/a08trd0f/Zero-One-XCombinator/extras/results/hf-sft-scale-2000"
stage = Path(tempfile.mkdtemp(prefix="scale-2000-pull-"))
print(f"Pulling {remote_dir} -> {stage}")
# pscp -r for directory
import os

hostkey = os.environ.get("ZO_CLUSTER_HOSTKEY", "")
pw = os.environ.get("ZO_CLUSTER_PASSWORD", "")
args = ["pscp", "-batch", "-r"]
if hostkey:
    args += ["-hostkey", hostkey]
args += ["-pw", pw, f"{target}:{remote_dir}/*", str(stage)]
subprocess.run(args, check=True)
print("Staged:", list(stage.iterdir()))
subprocess.run(
    ["uv", "run", "zo-track", "promote", "-r", str(stage), "-s", "hf-sft-scale-2000"],
    cwd=repo,
    check=True,
)
shutil.rmtree(stage, ignore_errors=True)
print("Done — INDEX + extras/results/hf-sft-scale-2000 updated")

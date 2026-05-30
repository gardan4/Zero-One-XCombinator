from zo_train.cluster._remote import ssh_capture, ssh_target
from zo_train.cluster._slurm import ensure_cluster_env

ensure_cluster_env()
target = ssh_target()
paths = [
    "/leonardo_scratch/large/usertrain/a08trd0f/hf-local/Qwen2.5-1.5B-Instruct",
    "/leonardo_scratch/large/usertrain/a08trd0f/zo-models/Qwen--Qwen2.5-1.5B-Instruct",
    "/leonardo_scratch/large/usertrain/a08trd0f/hf-local",
    "/leonardo_scratch/large/usertrain/a08trd0f/zo-models",
]
for p in paths:
    print("===", p, "===")
    print(ssh_capture(target, f"ls -la {p} 2>/dev/null | head -8 || echo MISSING"))

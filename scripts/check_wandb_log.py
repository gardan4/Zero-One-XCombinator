from zo_train.cluster._remote import ssh_capture, ssh_target
from zo_train.cluster._slurm import ensure_cluster_env

ensure_cluster_env()
target = ssh_target()
job = "43116477"
repo = "/leonardo/home/usertrain/a08trd0f/Zero-One-XCombinator"
cmd = (
    "log=$(ls " + repo + "/slurm_logs/*" + job + "*.out 2>/dev/null | head -1); "
    "grep -A15 'publish failed' \"$log\" 2>/dev/null || echo NO_WANDB_LINES"
)
print(ssh_capture(target, cmd))

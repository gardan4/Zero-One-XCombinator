from zo_train.cluster._remote import ssh_capture, ssh_target
from zo_train.cluster._slurm import ensure_cluster_env

ensure_cluster_env()
target = ssh_target()
job = "43120833"
run = "20260530_184839_eval_judge-eval-zeroshot-rules-v3_f6ed56"
repo = "/leonardo/home/usertrain/a08trd0f/Zero-One-XCombinator"
results = f"/leonardo_scratch/large/usertrain/a08trd0f/zo-experiments/{run}/results"

q = ssh_capture(target, f"squeue -j {job} -h -o '%T %M' 2>/dev/null || echo DONE")
print("QUEUE:", q or "DONE")
log_cmd = (
    f"log=$(ls {repo}/slurm_logs/*{job}*.out 2>/dev/null | head -1); "
    "grep -E 'zo-track|hub_inference|publish|wandb|judge eval complete|Traceback|OOM' \"$log\" 2>/dev/null | tail -20"
)
print("KEY LOG:")
print(ssh_capture(target, log_cmd) or "(waiting)")
print("RESULTS:")
print(ssh_capture(target, f"ls -1 {results} 2>/dev/null | paste -sd, - || echo none"))

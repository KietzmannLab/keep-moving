"""
split training into multiple jobs
"""
import time
import subprocess
import wandb

nruns = 10
nparallel = 6

submit_script = "submit_sweep_agent.sh"
slurm_options = [
    "-p",
    "klab-gpu",
    "-c",
    "9",
    "--mem",
    "48G",
    "--gres",
    "gpu:H100.20gb:1",
    "--time=05:00:00",
]
script_options = ["cgk0id73"]


cmd = ["sbatch", "--array", f"1-{nruns}%{nparallel}", *slurm_options]
cmd.append(submit_script)
cmd += script_options
print("command:")
print(" ".join(cmd))
res = subprocess.run(cmd, stdout=subprocess.PIPE)
outstr = res.stdout.decode("utf-8")[:-1]  # string format and remove newline
jobid = outstr.split(" ")[-1]  # last word in submission output is jobid
print(f"submitted {jobid}")

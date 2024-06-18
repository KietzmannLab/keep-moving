"""
split training into multiple jobs
"""
import time
import subprocess
import wandb

nruns = 50

submit_script = "submit_train.sh"
slurm_options = [
    "-p",
    "klab-gpu",
    "-c",
    "9",
    "--mem",
    "50G",
    "--gres",
    "gpu:H100.10gb:1",
    "--time=05:00:00",
]
script_options = ["bzbvg79h"]

for i in range(nruns):
    cmd = ["sbatch", *slurm_options]
    cmd.append(submit_script)
    cmd += script_options
    res = subprocess.run(cmd, stdout=subprocess.PIPE)
    outstr = res.stdout.decode("utf-8")[:-1]  # string format and remove newline
    jobid = outstr.split(" ")[-1]  # last word in submission output is jobid
    print(f"submitted {jobid}")

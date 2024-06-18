"""
split training into multiple jobs
"""
import time
import subprocess
import wandb

runid = "l5odo31l"
tstamp = "1689624742.7842183"
nblocks = 100
chunksize = 20
last_checkpoint = 6

submit_script = "submit_train.sh"
slurm_options = [
    "-p",
    "klab-gpu",
    "-c",
    "9",
    "--mem",
    "250G",
    "--gres",
    "gpu:H100.20gb:1",
    "--time=12:00:00",
]
script_options = [
    "environment=klab",
    "experiment=split_ecoset_continual_three_block_replay",
    f"environment.timestamp={tstamp}",
    f"environment.wandb_id={runid}",
]

jobid = ""

for i in range(last_checkpoint + chunksize, nblocks, chunksize):
    scrpt_iter_opt = f"experiment.exit_early_after={i}"  # stop after ith phase
    cmd = ["sbatch", *slurm_options]
    if not jobid == "":
        cmd.append(f"--dependency=afterok:{jobid}")
    cmd.append(submit_script)
    cmd += script_options
    cmd.append(scrpt_iter_opt)
    res = subprocess.run(cmd, stdout=subprocess.PIPE)
    outstr = res.stdout.decode("utf-8")[:-1]  # string format and remove newline
    jobid = outstr.split(" ")[-1]  # last word in submission output is jobid

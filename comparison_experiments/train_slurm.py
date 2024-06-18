"""
split training into multiple jobs
"""
import time
import subprocess
import wandb

runid = wandb.util.generate_id()
tstamp = time.time()
nblocks = 11
chunksize = 11

submit_script = "submit_train.sh"
slurm_options = [
    "-p",
    "klab-gpu",
    "-c",
    "9",
    "--mem",
    "32G",
    "--gres",
    "gpu:H100.20gb:1",
    "--time=02:00:00",
]

script_options = [
    "environment=klab",
    "experiment=cifar110_test",
    f"environment.timestamp={tstamp}",
    f"environment.wandb_id={runid}",
]

jobid = ""

iterator = list(range(chunksize, nblocks + 1, chunksize))
print(iterator)

# get iterator counting to nblocks in chunks of chunksize, make sure to count exactly to nblocks
for i in iterator:
    scrpt_iter_opt = f"experiment.exit_early_after={i}"  # stop after ith phase
    cmd = ["sbatch", *slurm_options]
    if len(jobid) > 0:
        cmd.append(f"--dependency=afterok:{jobid}")
    cmd.append(submit_script)
    cmd += script_options
    cmd.append(scrpt_iter_opt)
    res = subprocess.run(cmd, stdout=subprocess.PIPE)
    outstr = res.stdout.decode("utf-8")[:-1]  # string format and remove newline
    jobid = outstr.split(" ")[-1]  # last word in submission output is jobid

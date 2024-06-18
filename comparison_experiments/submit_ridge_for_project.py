"""
split training into multiple jobs
"""
import subprocess
import os

submit_script = "submit_ridge.sh"

# list all runs in project dir

project = "plasticity"
logdir = "logs"
project_dir = os.path.join(logdir, project)
runs = os.listdir(project_dir)

# check whether run is a directory
runs = [run for run in runs if os.path.isdir(os.path.join(project_dir, run))]

embedding_layer = "dense_dropout"

for run in runs:
    cmd = ["sbatch"]
    cmd.append(submit_script)
    opt = [
        f"project={project}",
        f"run_name={run}",
        f"embedding_layer={embedding_layer}",
    ]
    cmd += opt
    res = subprocess.run(cmd, stdout=subprocess.PIPE)
    outstr = res.stdout.decode("utf-8")[:-1]  # string format and remove newline
    jobid = outstr.split(" ")[-1]  # last word in submission output is jobid
    print(f"submitted {jobid}")

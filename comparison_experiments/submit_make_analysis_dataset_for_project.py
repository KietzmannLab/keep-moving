"""
split training into multiple jobs
"""
import subprocess
import os
from time import time

submit_script = "submit_make_analysis_datasets.sh"

# list all runs in project dir

project = "plasticity"
logdir = "logs"
embedding_layer = "relu_fc1"
analysis_tasks = 0
tstamp = int(time())
cfgfile = f"tmp-cfg-{tstamp}.txt"
nparrallel = 0
project_dir = os.path.join(logdir, project)

runs = os.listdir(project_dir)
# check whether run is a directory
runs = [run for run in runs if os.path.isdir(os.path.join(project_dir, run))]

for run in runs:
    for analysis_task in analysis_tasks:
        opt = [
            f"project={project}",
            f"run_name={run}",
            f"embedding_layer={embedding_layer}",
            f"analysis_task={analysis_task}",
        ]
        # if cfg file does not exist, create. Then add opt on a new line
        if not os.path.exists(cfgfile):
            with open(cfgfile, "w") as f:
                f.write(" ".join(opt))
        else:
            with open(cfgfile, "a") as f:
                f.write("\n" + " ".join(opt))

cmd = ["sbatch"]
cmd.extend([f"--array", f"1-{len(runs) * len(analysis_tasks)}%{nparrallel}"])
cmd.append(submit_script)
cmd.append(cfgfile)
print(cmd)
res = subprocess.run(cmd, stdout=subprocess.PIPE)
outstr = res.stdout.decode("utf-8")[:-1]  # string format and remove newline
jobid = outstr.split(" ")[-1]  # last word in submission output is jobid
print(f"submitted {jobid}")

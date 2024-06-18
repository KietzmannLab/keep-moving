import subprocess
import os
from time import time

submit_script = "submit_analysis_filtered_activation_trajectory_single_run.sh"
project = "naive_plasticiy_cifar110"
logdir = "logs"
embedding_layers = ["relu_fc1", "readout"]
filter_layer = "classifier.classifiers.0.classifier.weight"
task = 0
max_phase = 11


nparrallel = 100

cfg = f"tmp-cfg-{int(time())}.txt"

project_dir = os.path.join(logdir, project)
runs = os.listdir(project_dir)

# filter for folders
runs = [run for run in runs if os.path.isdir(os.path.join(project_dir, run))]

# submit a job for each run
for run in runs:
    opt = (
        ["--project", project, "--run", run, "--layers"]
        + embedding_layers
        + [
            "--filter_layer",
            filter_layer,
            "--task",
            str(task),
            "--max_phase",
            str(max_phase),
        ]
    )

    # if cfg file does not exist, create. Then add opt on a new line
    if not os.path.exists(cfg):
        with open(cfg, "w") as f:
            f.write(" ".join(opt))
    else:
        with open(cfg, "a") as f:
            f.write("\n" + " ".join(opt))

cmd = ["sbatch"]
cmd.extend([f"--array", f"1-{len(runs)}%{nparrallel}"])
cmd.append(submit_script)
cmd.append(cfg)
print(cmd)

# run cmd
res = subprocess.run(cmd, stdout=subprocess.PIPE)
print(res)

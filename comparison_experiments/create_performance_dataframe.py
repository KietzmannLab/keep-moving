import pandas as pd
import numpy as np
import os
import yaml
import argparse

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, help="name of the project")
parser.add_argument("--logdir", type=str, help="path to logdir", default="logs")
# parse multiple algorithm fields
parser.add_argument(
    "--af", nargs="+", help="algorithm fields to include in dataframe", default=[]
)
# parse multiple model fields
parser.add_argument(
    "--mf", nargs="+", help="model fields to include in dataframe", default=[]
)
args = parser.parse_args()
logdir, project = args.logdir, args.project
algorithm_fields = args.af
model_fields = args.mf

project_dir = os.path.join(logdir, project)
runs = os.listdir(project_dir)
# get folders only
runs = [run for run in runs if os.path.isdir(os.path.join(project_dir, run))]
print("found runs:")
for r in runs:
    print(r)

cfgs = []
for run in runs:
    cfg_path = os.path.join(project_dir, run, "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfgs.append(cfg)

algo_cfgs = [cfg["experiment"]["algorithm"] for cfg in cfgs]

# %%

# get all performance results
cl_results = []
ridge_results = []

for run in runs:
    run_dir = os.path.join(project_dir, run)
    cl_path = os.path.join(run_dir, "results.yaml")
    ridge_path = os.path.join(run_dir, "ridge_aligned_readout_results.yaml")

    with open(cl_path) as f:
        cl_res = yaml.load(f, Loader=yaml.FullLoader)
    with open(ridge_path) as f:
        ridge_res = yaml.load(f, Loader=yaml.FullLoader)

    cl_results.append(cl_res)
    ridge_results.append(ridge_res)

# %%


def results_to_pandas(results_dict, result_type):
    tasks = list(results_dict.keys())
    nphases = len(results_dict[tasks[0]])
    ntasks = len(tasks)
    rows = []
    for task in tasks:
        for phase in range(nphases):
            result = results_dict[task][phase]
            row = {"task": task, "phase": phase, f"{result_type}_result": result}
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


# get ridge and cl frames
cl_frames = [results_to_pandas(res, "cl") for res in cl_results]
ridge_frames = [results_to_pandas(res, "ridge") for res in ridge_results]

# %% add info to each frame


for i, frame in enumerate(cl_frames):
    frame["run"] = cfgs[i]["experiment"]["name"]
    frame["tstamp"] = cfgs[i]["environment"]["timestamp"]
    frame["algorithm"] = algo_cfgs[i]["name"]
    frame["manager"] = cfgs[i]["experiment"]["use_manager"]
    for field in algorithm_fields:
        frame[field] = algo_cfgs[i][field]
    for field in model_fields:
        frame[field] = cfgs[i]["experiment"]["model"][field]

for i, frame in enumerate(ridge_frames):
    frame["run"] = cfgs[i]["experiment"]["name"]
    frame["tstamp"] = cfgs[i]["environment"]["timestamp"]

cl_frame = pd.concat(cl_frames, ignore_index=True).reset_index()
ridge_frame = pd.concat(ridge_frames, ignore_index=True).reset_index()

# %%

# set index of ridge frame
ridge_frame.set_index(["task", "phase", "run", "tstamp"], inplace=True)
frame = cl_frame.join(
    ridge_frame,
    how="outer",
    on=["task", "phase", "run", "tstamp"],
    lsuffix="_cl",
    rsuffix="_ridge",
)
# drop index_cl and index_ridge, reset index
frame.drop(columns=["index_cl", "index_ridge"], inplace=True)
frame.reset_index(inplace=True)

# %%

# save to csv in project folder
frame.to_csv(os.path.join(project_dir, "performance.csv"), index=False)

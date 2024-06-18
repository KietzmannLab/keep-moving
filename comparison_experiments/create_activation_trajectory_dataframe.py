# %%
import numpy as np
import pandas
import os

project = "gem_plasticity_cifar110"
logdir = "logs"
task = 0
layers = ["relu_fc1", "readout"]
# join list with _
layer_str = "_".join(layers)


# %%

# list all runs
runs = os.listdir(os.path.join(logdir, project))
runs = [r for r in runs if os.path.isdir(os.path.join(logdir, project, r))]

# for each run, assemble path to analysis folder for task
paths = [
    os.path.join(logdir, project, r, "analysis", "task_{}".format(task)) for r in runs
]

distances_to_first = []
distances_cumulative = []

# for each run, load trajectories
for path in paths:
    distances_to_first.append(
        np.load(
            os.path.join(path, f"distances_to_first_{layer_str}.npy"), allow_pickle=True
        ).item()
    )
    distances_cumulative.append(
        np.load(
            os.path.join(path, f"distances_to_previous_{layer_str}.npy"),
            allow_pickle=True,
        ).item()
    )

# %%


def dataframe_for_run(name, distances_to_first, distances_cumulative):
    # create dataframe
    df = pandas.DataFrame(
        columns=["run", "layer", "phase", "distance_to_first", "distance_cumulative"]
    )
    # get all layer names
    layers = list(distances_to_first.keys())
    phases = []
    dist_to_first = []
    dist_cumulative = []
    layer_names = []
    # for each layer
    for layer in layers:
        # create vector of phases
        layer_phases = np.arange(1, len(distances_to_first[layer]) + 1, 1)
        print(layer_phases)
        layername = [layer] * len(layer_phases)
        # get results
        layer_dist_to_first = distances_to_first[layer]
        print(layer_dist_to_first)
        layer_dist_cumulative = distances_cumulative[layer]
        # append to lists
        phases.extend(layer_phases)
        dist_to_first.extend(layer_dist_to_first)
        dist_cumulative.extend(layer_dist_cumulative)
        layer_names.extend(layername)

    name = [name] * len(phases)

    # create dataframe
    df = pandas.DataFrame(
        {
            "run": name,
            "layer": layer_names,
            "phase": phases,
            "distance_to_first": dist_to_first,
            "distance_cumulative": dist_cumulative,
        }
    )
    return df


# %%

# create dataframe for each run
dfs = []
for run, dist_to_first, dist_cumulative in zip(
    runs, distances_to_first, distances_cumulative
):
    dfs.append(dataframe_for_run(run, dist_to_first, dist_cumulative))

# stack
df = pandas.concat(dfs)

# save
df.to_pickle(
    os.path.join(
        "analysis_data",
        "trajectory_dfs",
        f"trajectories_{project}_task_{task}_{layer_str}.pkl",
    )
)

# %%

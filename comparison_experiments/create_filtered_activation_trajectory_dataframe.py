# %%
import numpy as np
import pandas
import os

project = "gem_plasticity_cifar110"
logdir = "logs"
task = 0
layers = ["relu_fc1"]
# join list with _
layer_str = "_".join(layers)


# %%

# list all runs
runs = os.listdir(os.path.join(logdir, project))
runs = [r for r in runs if os.path.isdir(os.path.join(logdir, project, r))]
print(f"found {len(runs)} runs for project")

# for each run, assemble path to analysis folder for task
paths = [
    os.path.join(logdir, project, r, "analysis", "task_{}".format(task)) for r in runs
]

distances_to_first_in_plane = []
distances_to_first_null_space = []
distances_cumulative_in_plane = []
distances_cumulative_null_space = []

# for each run, load trajectories
for path in paths:
    distances_to_first_in_plane.append(
        np.load(
            os.path.join(path, f"filtered_in_plane_distances_to_first_{layer_str}.npy"),
            allow_pickle=True,
        ).item()
    )
    distances_cumulative_in_plane.append(
        np.load(
            os.path.join(
                path, f"filtered_in_plane_distances_to_previous_{layer_str}.npy"
            ),
            allow_pickle=True,
        ).item()
    )
    distances_to_first_null_space.append(
        np.load(
            os.path.join(
                path, f"filtered_null_space_distances_to_first_{layer_str}.npy"
            ),
            allow_pickle=True,
        ).item()
    )
    distances_cumulative_null_space.append(
        np.load(
            os.path.join(
                path, f"filtered_null_space_distances_to_previous_{layer_str}.npy"
            ),
            allow_pickle=True,
        ).item()
    )


print("to first in,out")
print(len(distances_to_first_in_plane), len(distances_to_first_null_space))

print("cumulative in,out")
print(len(distances_cumulative_in_plane), len(distances_cumulative_null_space))

print(distances_cumulative_null_space)

# %%


def dataframe_for_run(
    name,
    distances_to_first_in,
    distances_to_first_out,
    distances_cumulative_in,
    distances_cumulative_out,
):
    # create dataframe
    df = pandas.DataFrame(
        columns=[
            "run",
            "layer",
            "phase",
            "distance_to_first_in_plane",
            "distance_to_first_null_space",
            "distance_cumulative_in_plane",
            "dinstance_cumulative_null_space",
        ]
    )
    print(run)
    # get all layer names
    layers = list(distances_to_first_in.keys())
    print("layers", layers)
    phases = []
    dist_to_first_in = []
    dist_to_first_out = []
    dist_cumulative_in = []
    dist_cumulative_out = []
    layer_names = []
    # for each layer
    for layer in layers:
        # create vector of phases
        layer_phases = np.arange(1, len(distances_to_first_in[layer]) + 1, 1)
        print(layer_phases)
        layername = [layer] * len(layer_phases)
        # get results
        layer_dist_to_first_in = distances_to_first_in[layer]
        layer_dist_cumulative_in = distances_cumulative_in[layer]
        layer_dist_to_first_out = distances_to_first_out[layer]
        layer_dist_cumulative_out = distances_cumulative_out[layer]

        # append to lists
        phases.extend(layer_phases)
        dist_to_first_in.extend(layer_dist_to_first_in)
        dist_to_first_out.extend(layer_dist_to_first_out)
        dist_cumulative_in.extend(layer_dist_cumulative_in)
        dist_cumulative_out.extend(layer_dist_cumulative_out)
        layer_names.extend(layername)

    name = [name] * len(phases)

    # create dataframe
    df = pandas.DataFrame(
        {
            "run": name,
            "layer": layer_names,
            "phase": phases,
            "distance_to_first_in_plane": dist_to_first_in,
            "distance_to_first_null_space": dist_to_first_out,
            "distance_cumulative_in_plane": dist_cumulative_in,
            "distance_cumulative_null_space": dist_cumulative_out,
        }
    )
    return df


# %%

# create dataframe for each run
dfs = []
for (
    run,
    dist_to_first_in,
    dist_to_first_out,
    dist_cumulative_in,
    dist_cumulative_out,
) in zip(
    runs,
    distances_to_first_in_plane,
    distances_to_first_null_space,
    distances_cumulative_in_plane,
    distances_cumulative_null_space,
):
    dfs.append(
        dataframe_for_run(
            run,
            dist_to_first_in,
            dist_to_first_out,
            dist_cumulative_in,
            dist_cumulative_out,
        )
    )

# stack
df = pandas.concat(dfs)

# save
df.to_pickle(
    os.path.join(
        "analysis_data",
        "trajectory_dfs",
        f"filtered_trajectories_{project}_task_{task}_{layer_str}.pkl",
    )
)

# %%

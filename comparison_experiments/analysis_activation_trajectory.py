# %%
import numpy as np
import matplotlib.pyplot as plt
import os

# %%
project = "naive_plasticiy_cifar110"
layers = ["relu_fc1", "readout"]
layer_string = "_".join(layers)
task = 0
max_phase = 10

project_path = os.path.join("logs", project)

# list all runs
runs = os.listdir(project_path)

# filter for folders
runs = [run for run in runs if os.path.isdir(os.path.join(project_path, run))]
run_names = [run.split("_")[-1] for run in runs]

# for each run, load embeddings through time for each layer
# store as a dictionary
run_embeddings = {}
for run, run_name in zip(runs, run_names):
    analysis_dir = os.path.join(project_path, run, "analysis", f"task_{str(task)}")
    # get all phase folders
    phases = os.listdir(analysis_dir)
    # filter for folders starting with phase
    phases = [phase for phase in phases if phase.startswith("phase")]
    # for each phase, load embeddings
    current_run_embeddings = dict()
    for phase in phases:
        phase_dir = os.path.join(analysis_dir, phase)
        # for each layer name construct the path to the embeddings
        layer_embeddings = dict()
        for layer in layers:
            layer_path = os.path.join(phase_dir, f"embeddings_layer_{layer}.npy")
            layer_embeddings[layer] = np.load(layer_path)
        current_run_embeddings[phase] = layer_embeddings
    run_embeddings[run] = current_run_embeddings
    run_embeddings[run]["name"] = run_name


# %%
def get_embeddings(run_embeddings, run, layer):
    embeddings = run_embeddings[run]
    # get all phases
    phases = list(embeddings.keys())
    # filter for phase folders
    phases = [phase for phase in phases if phase.startswith("phase")]
    nphases = len(phases)
    # get all embeddings for layer
    embed_list = []
    for phase in range(min(nphases, max_phase)):
        phase_name = f"phase_{str(phase)}"
        embed_list.append(embeddings[phase_name][layer])
    return embed_list


def compute_distance_to_first(embeddings):
    distances = []
    for i in range(1, len(embeddings), 1):
        dists = np.linalg.norm(embeddings[0] - embeddings[i], axis=1)
        dist = np.mean(dists)
        distances.append(dist)
    return distances


def compute_distance_to_previous(embeddings):
    distances = []
    for i in range(1, len(embeddings), 1):
        dists = np.linalg.norm(embeddings[i - 1] - embeddings[i], axis=1)
        dist = np.mean(dists)
        distances.append(dist)
    return distances


# %%

# compute distance to first for all runs and layers
distances_to_first = dict()
for run in run_embeddings.keys():
    distances_to_first[run] = dict()
    for layer in layers:
        embeddings = get_embeddings(run_embeddings, run, layer)
        distances_to_first[run][layer] = compute_distance_to_first(embeddings)
    # save to run directory in subdir analysis, as pkl
    save_dir = os.path.join(project_path, run, "analysis", f"task_{str(task)}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(
        os.path.join(save_dir, f"distances_to_first_{layer_string}.npy"),
        distances_to_first[run],
    )


# compute distances to previous for all runs and layers
distances_to_previous = dict()
for run in run_embeddings.keys():
    distances_to_previous[run] = dict()
    for layer in layers:
        embeddings = get_embeddings(run_embeddings, run, layer)
        distances_to_previous[run][layer] = compute_distance_to_previous(embeddings)
    # save to run directory in subdir analysis, as pkl
    save_dir = os.path.join(project_path, run, "analysis", f"task_{str(task)}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(
        os.path.join(save_dir, f"distances_to_previous_{layer_string}.npy"),
        distances_to_previous[run],
    )


# %%
# get tab10 colormap
colors = plt.cm.get_cmap("tab10").colors
# map colors to run names
colors = dict(zip(np.unique(run_names), colors))
print(colors)
print(run_names)
styles = ["-", "--"]
styles = dict(zip(layers, styles))
# %%

# plot
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
for i, run in enumerate(distances_to_first.keys()):
    for layer in distances_to_first[run].keys():
        label = run.split("_")[-1] + "_" + layer
        ax.plot(
            distances_to_first[run][layer],
            label=label,
            marker="o",
            color=colors[run.split("_")[-1]],
            linestyle=styles[layer],
        )
ax.legend()
ax.set_xlabel("Phase")
ax.set_ylabel("Distance to first")
ax.set_title("Distance to first embedding")

# remove duplicate legend entries
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())


# save
fig.savefig(
    os.path.join(
        "logs", project, f"distance_to_first_layers_{layer_string}_task_{task}.pdf"
    )
)
# print savedir
print(
    f"Saved to {os.path.join('logs', project, f'distance_to_first_layers_{layer_string}_task_{task}.pdf')}"
)

# %% TODO This is wrong! Do not use distance to first, but distance to previous

# plot cumulative distance integrating over trajectory
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
for i, run in enumerate(distances_to_previous.keys()):
    for layer in distances_to_previous[run].keys():
        label = run.split("_")[-1] + "_" + layer
        ax.plot(
            np.cumsum(distances_to_previous[run][layer]),
            label=label,
            marker="o",
            color=colors[run.split("_")[-1]],
            linestyle=styles[layer],
        )
ax.legend()
ax.set_xlabel("Phase")
ax.set_ylabel("Cumulative distance")
ax.set_title("Path length in activation space")

# remove duplicate legend entries
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

fig.savefig(
    os.path.join(
        "logs", project, f"cumulative_distance_layers_{layer_string}_task_{task}.pdf"
    )
)

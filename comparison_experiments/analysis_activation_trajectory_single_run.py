# %%
import numpy as np
import os
import argparse

# %%

# get project layers and run from argparse instead
parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str)
parser.add_argument("--run", type=str)
parser.add_argument("--layers", nargs="+", type=str)
parser.add_argument("--task", type=int, default=0)
parser.add_argument("--max_phase", type=int, default=11)

args = parser.parse_args()
project = args.project
run = args.run
layers = args.layers
task = args.task
max_phase = args.max_phase

# pretty print this info
print(f"project: {project}")
print(f"run: {run}")
print(f"layers: {layers}")
print(f"task: {task}")
print(f"max_phase: {max_phase}")


layer_string = "_".join(layers)
project_path = os.path.join("logs", project)
run_name = run.split("_")[-1]

# for each run, load embeddings through time for each layer
# store as a dictionary
run_embeddings = {}
analysis_dir = os.path.join(project_path, run, "analysis", f"task_{str(task)}")
# get all phase folders
phases = os.listdir(analysis_dir)
# filter for folders starting with phase
phases = [phase for phase in phases if phase.startswith("phase")]
# for each phase, load embeddings
for phase in phases:
    phase_dir = os.path.join(analysis_dir, phase)
    # for each layer name construct the path to the embeddings
    layer_embeddings = dict()
    for layer in layers:
        layer_path = os.path.join(phase_dir, f"embeddings_layer_{layer}.npy")
        layer_embeddings[layer] = np.load(layer_path)
    run_embeddings[phase] = layer_embeddings
run_embeddings["name"] = run_name


# %%
def get_embeddings(embeddings, layer):
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
for layer in layers:
    embeddings = get_embeddings(run_embeddings, layer)
    distances_to_first[layer] = compute_distance_to_first(embeddings)
# save to run directory in subdir analysis, as pkl
save_dir = os.path.join(project_path, run, "analysis", f"task_{str(task)}")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
np.save(
    os.path.join(save_dir, f"distances_to_first_{layer_string}.npy"), distances_to_first
)
print(f"saved distances to first for {run} {layer_string} to {save_dir}")

# compute distances to previous for all runs and layers
distances_to_previous = dict()
for layer in layers:
    embeddings = get_embeddings(run_embeddings, layer)
    distances_to_previous[layer] = compute_distance_to_previous(embeddings)
# save to run directory in subdir analysis, as pkl
save_dir = os.path.join(project_path, run, "analysis", f"task_{str(task)}")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
np.save(
    os.path.join(save_dir, f"distances_to_previous_{layer_string}.npy"),
    distances_to_previous,
)
print(f"saved distances to previous for {run} {layer_string} to {save_dir}")

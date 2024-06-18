# %%
import numpy as np
import os
import argparse
import torch

from scipy.linalg import orth, null_space

# %%

# get project layers and run from argparse instead
parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str)
parser.add_argument("--run", type=str)
parser.add_argument("--layers", nargs="+", type=str)
parser.add_argument(
    "--filter_layer", type=str, default="classifier.classifiers.0.classifier.weight"
)
parser.add_argument("--task", type=int, default=0)
parser.add_argument("--max_phase", type=int, default=11)

args = parser.parse_args()
project = args.project
run = args.run
layers = args.layers
task = args.task
max_phase = args.max_phase
filter_layer = args.filter_layer

assert len(layers) == 1, "This analysis needs exactly one layer"

# %%
# pretty print this info
print(f"project: {project}")
print(f"run: {run}")
print(f"layers: {layers}")
print(f"task: {task}")
print(f"max_phase: {max_phase}")
print(f"filter_name: {filter_layer}")


layer_string = "_".join(layers)
project_path = os.path.join("logs", project)
run_name = run.split("_")[-1]

# load model for task
model_path = os.path.join(project_path, run, f"model_phase_{task}.pt")  # phase of task
state_dict = torch.load(model_path, map_location="cpu")

# %%

weight_mat = state_dict[filter_layer].numpy()
"""
# get orthonormal basis of range of weight matrix
C = orth(weight_mat.T)
print("RANGE OF WEIGHT MATRIX USED FOR FILTERING", C.shape)
in_plane_filter = np.matmul(C, C.T)
print("IN PLANE FILTER", in_plane_filter.shape)

N = null_space(weight_mat)
print("NULLSPACE OF WEIGHT MATRIX USED FOR FILTERING", N.shape)
out_of_plane_filter = np.matmul(N, N.T)
print("OUT OF PLANE FILTER", out_of_plane_filter.shape)

"""

# THIS IS THE EXACT DECOMPOSITION FROM THE LINEAR ANALYSIS SCRIPTS
C = orth(weight_mat.T)
CC_T = C @ C.T
N = null_space(weight_mat)
NN_T = N @ N.T

in_plane_filter = CC_T
out_of_plane_filter = NN_T

# %%

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
def get_embeddings(embeddings, layer, filter):
    # get all phases
    phases = list(embeddings.keys())
    # filter for phase folders
    phases = [phase for phase in phases if phase.startswith("phase")]
    nphases = len(phases)
    # get all embeddings for layer
    embed_list = []
    for phase in range(min(nphases, max_phase)):
        phase_name = f"phase_{str(phase)}"
        embedding = embeddings[phase_name][layer]
        # filter embedding
        embedding = np.matmul(embedding, filter)
        embed_list.append(embedding)
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
distances_to_first_in_plane = dict()
for layer in layers:  # exclude readout layer
    embeddings = get_embeddings(run_embeddings, layer, in_plane_filter)
    distances_to_first_in_plane[layer] = compute_distance_to_first(embeddings)
# save to run directory in subdir analysis, as pkl
save_dir = os.path.join(project_path, run, "analysis", f"task_{str(task)}")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
np.save(
    os.path.join(save_dir, f"filtered_in_plane_distances_to_first_{layer_string}.npy"),
    distances_to_first_in_plane,
)
print(f"saved distances to first for {run} {layer_string} to {save_dir}")

# compute distances to previous for all runs and layers
distances_to_previous_in_plane = dict()
for layer in layers:
    embeddings = get_embeddings(run_embeddings, layer, in_plane_filter)
    distances_to_previous_in_plane[layer] = compute_distance_to_previous(embeddings)
# save to run directory in subdir analysis, as pkl
save_dir = os.path.join(project_path, run, "analysis", f"task_{str(task)}")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
np.save(
    os.path.join(
        save_dir, f"filtered_in_plane_distances_to_previous_{layer_string}.npy"
    ),
    distances_to_previous_in_plane,
)
print(f"saved distances to previous for {run} {layer_string} to {save_dir}")


# compute distance to first for all runs and layers
distances_to_first_null_space = dict()
for layer in layers:  # exclude readout layer
    embeddings = get_embeddings(run_embeddings, layer, out_of_plane_filter)
    distances_to_first_null_space[layer] = compute_distance_to_first(embeddings)
# save to run directory in subdir analysis, as pkl
save_dir = os.path.join(project_path, run, "analysis", f"task_{str(task)}")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
np.save(
    os.path.join(
        save_dir, f"filtered_null_space_distances_to_first_{layer_string}.npy"
    ),
    distances_to_first_null_space,
)
print(f"saved distances to first for {run} {layer_string} to {save_dir}")

# compute distances to previous for all runs and layers
distances_to_previous_null_space = dict()
for layer in layers:
    embeddings = get_embeddings(run_embeddings, layer, out_of_plane_filter)
    distances_to_previous_null_space[layer] = compute_distance_to_previous(embeddings)
# save to run directory in subdir analysis, as pkl
save_dir = os.path.join(project_path, run, "analysis", f"task_{str(task)}")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
np.save(
    os.path.join(
        save_dir, f"filtered_null_space_distances_to_previous_{layer_string}.npy"
    ),
    distances_to_previous_null_space,
)
print(f"saved distances to previous for {run} {layer_string} to {save_dir}")

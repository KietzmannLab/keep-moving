# %% imports
import os
import torch
import numpy as np
import yaml
from pprint import pprint
import argparse

from scipy.linalg import orth, null_space

from load_data import load_data, map_classes_for_task
from model import Model

DEVICE = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--rundir", type=str, help="path to the run directory")
args = parser.parse_args()
rundir = args.rundir

# %% load cfg.yaml as dict

with open(os.path.join(rundir, "cfg.yaml"), "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

pprint(cfg)


# %% make sure all seeds are set to the same values as at training time

seed = cfg["seed"]
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# %% load data

data, classes = load_data(
    batchsize=cfg["batchsize"],
    nworkers=0,
    nsplits=cfg["nsplits"],
    shuffle_classes=cfg["shuffle_classes"],
    seed=cfg["seed"],
)

test_loaders = [test_loader for train_loader, test_loader in data]
test_loader_t0 = test_loaders[0]
test_loader_t1 = test_loaders[1]

print("loaded data.")

# %% load model

try:
    model = Model(
        device=DEVICE,
        with_bias=cfg["with_bias"],
        l1=cfg["l1"],
        channels_per_block=cfg["channels_per_block"],
        dense_size=cfg["dense_size"],
    )
except (
    KeyError
):  # for backwards compatibility, if not all parameters are present in cfg
    model = Model(
        device=DEVICE,
        with_bias=cfg["with_bias"],
        l1=cfg["l1"],
    )
model.load_state_dict(
    torch.load(os.path.join(rundir, "model_task_0.pt"), map_location=DEVICE)
)

print("loaded model.")

# %% collect embeddings after training on task 0

model.eval()

embeddings = []


def activation_hook(self, input, output):
    embeddings.append(output)


handle = model.blocks[-1].register_forward_hook(activation_hook)

with torch.no_grad():
    for batch in test_loader_t0:
        x, y = batch
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        model(x, 0)

embeddings_t0 = torch.cat(embeddings, dim=0).numpy()

print(f"collected embeddings {embeddings_t0.shape} after training on task 0")

# compute accuracies

# t0
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader_t0:
        x, y = batch
        y = map_classes_for_task(y, classes[0])
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        outputs = model(x, 0)
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

acc_after_t0_t0 = correct / total

# t1
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader_t1:
        x, y = batch
        y = map_classes_for_task(y, classes[1])
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        outputs = model(x, 1)
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

acc_after_t0_t1 = correct / total

# %% collect embeddings after training on task 1

model.load_state_dict(
    torch.load(os.path.join(rundir, "model_task_1.pt"), map_location=DEVICE)
)

embeddings = []

with torch.no_grad():
    for batch in test_loader_t0:
        x, y = batch
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        model(x, 0)

embeddings_t1 = torch.cat(embeddings, dim=0).numpy()

print(f"collected embeddings {embeddings_t1.shape} after training on task 1")

# compute accuracy for both tasks

# t0
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader_t0:
        x, y = batch
        y = map_classes_for_task(y, classes[0])
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        outputs = model(x, 0)
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

acc_after_t1_t0 = correct / total

# t1
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader_t1:
        x, y = batch
        y = map_classes_for_task(y, classes[1])
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        outputs = model(x, 1)
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

acc_after_t1_t1 = correct / total

# %%

weight_mat = model.readouts[0].weight.detach().clone().data.numpy()
print(weight_mat.shape)

C = orth(weight_mat.T)
CC_T = C @ C.T
N = null_space(weight_mat)
NN_T = N @ N.T

in_plane_filter = CC_T
out_of_plane_filter = NN_T

distances = np.linalg.norm(embeddings_t0 - embeddings_t1, axis=1)

embeddings_t0_null = np.matmul(embeddings_t0, out_of_plane_filter)
embeddings_t1_null = np.matmul(embeddings_t1, out_of_plane_filter)

distances_null = np.linalg.norm(embeddings_t0_null - embeddings_t1_null, axis=1)

embeddings_t0_range = np.matmul(embeddings_t0, in_plane_filter)
embeddings_t1_range = np.matmul(embeddings_t1, in_plane_filter)

distances_range = np.linalg.norm(embeddings_t0_range - embeddings_t1_range, axis=1)

# %% print all results

print(f"mean distance: {np.mean(distances)}")
print(f"mean distance null: {np.mean(distances_null)}")
print(f"mean distance range: {np.mean(distances_range)}")

print(f"acc after t0 t0: {acc_after_t0_t0}")
print(f"acc after t0 t1: {acc_after_t0_t1}")
print(f"acc after t1 t0: {acc_after_t1_t0}")
print(f"acc after t1 t1: {acc_after_t1_t1}")

results = {
    "mean_distance": float(np.mean(distances)),
    "mean_distance_null": float(np.mean(distances_null)),
    "mean_distance_range": float(np.mean(distances_range)),
    "acc_after_t0_t0": float(acc_after_t0_t0),
    "acc_after_t0_t1": float(acc_after_t0_t1),
    "acc_after_t1_t0": float(acc_after_t1_t0),
    "acc_after_t1_t1": float(acc_after_t1_t1),
}

# save in rundir as results.yaml

with open(os.path.join(rundir, "results.yaml"), "w") as f:
    yaml.dump(results, f)

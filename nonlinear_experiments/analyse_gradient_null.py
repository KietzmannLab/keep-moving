# %% IMPORTS
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
import numpy as np
import yaml
from pprint import pprint
import argparse
from scipy.linalg import orth, null_space
from load_data import load_data, map_classes_for_task
from model import Model

# %%
# automatically determine DEVICE. Order of preference: cuda, mps, cpu
# set globally here so it is consistent across the whole script

if torch.cuda.is_available():
    DEVICE = "cuda"
# else if mps is available
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# %%

# set default in case of interactive session (skip next cell)
rundir = "logs/baseline/baseline_warmup_5_params_64_4_2_True_42_30_False_0.0"

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--rundir", type=str, help="path to the run directory")
args = parser.parse_args()
rundir = args.rundir

# %% load cfg.yaml as dict

with open(os.path.join(rundir, "cfg.yaml"), "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

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

train_loaders = [train_loader for train_loader, test_loader in data]
train_loader_t0 = train_loaders[0]
train_loader_t1 = train_loaders[1]

print("loaded data.")

# %% load model

# for backwards compatibility, potentially add missing keys in cfg
if "channels_per_block" not in cfg:
    cfg["channels_per_block"] = [8, 8, 16, 16, 32]
if "dense_size" not in cfg:
    cfg["dense_size"] = 64

model = Model(
    device=DEVICE,
    with_bias=cfg["with_bias"],
    l1=cfg["l1"],
    channels_per_block=cfg["channels_per_block"],
    dense_size=cfg["dense_size"],
)

model.to(DEVICE)

# load model state dict after training on task 1
model.load_state_dict(
    torch.load(os.path.join(rundir, "model_task_0.pt"), map_location=DEVICE)
)


def register_hook_retain_grad(module):

    outputs = []

    def hook_retain_grad(module, input, output):
        output.retain_grad()
        outputs.append(output)

    # register hook to dense layer preactivations
    handle = model.blocks[-1].register_forward_hook(hook_retain_grad)
    return outputs, handle


outputs, handle = register_hook_retain_grad(model.blocks[-1])
outputs_preact, handle_preact = register_hook_retain_grad(model.blocks[-2])
gradients, gradients_pre = [], []

# pass the training set through the model once
for x, y in train_loader_t0:
    model.zero_grad()
    y = map_classes_for_task(y, classes[0])
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    ypred = model(x, 0)
    loss = F.cross_entropy(ypred, y)
    loss.backward()
    gradients.append(outputs[-1].grad.data.cpu())
    gradients_pre.append(outputs_preact[-1].grad.data.cpu())
    outputs[-1] = outputs[-1].data.cpu()
    outputs_preact[-1] = outputs_preact[-1].data.cpu()

# remove hook
handle.remove()
handle_preact.remove()

# %% concatenate all outputs and gradients

outputs = torch.cat(outputs, dim=0)
outputs_preact = torch.cat(outputs_preact, dim=0)
gradients = torch.cat(gradients, dim=0)
gradients_pre = torch.cat(gradients_pre, dim=0)

# %% perform svd

u_act, s_act, v_act = torch.svd(outputs.T, some=True)
u_act_pre, s_act_pre, v_act_pre = torch.svd(outputs_preact.T, some=True)
u_grad, s_grad, v_grad = torch.svd(gradients.T, some=True)
u_grad_pre, s_grad_pre, v_grad_pre = torch.svd(gradients_pre.T, some=True)

# %% compute explained variance

cum_var_act = np.cumsum(s_act.detach().numpy() ** 2)
cum_var_act_pre = np.cumsum(s_act_pre.detach().numpy() ** 2)
cum_var_grad = np.cumsum(s_grad.detach().numpy() ** 2)
cum_var_grad_pre = np.cumsum(s_grad_pre.detach().numpy() ** 2)

exp_var_act = cum_var_act / cum_var_act[-1]
exp_var_act_pre = cum_var_act_pre / cum_var_act_pre[-1]
exp_var_grad = cum_var_grad / cum_var_grad[-1]
exp_var_grad_pre = cum_var_grad_pre / cum_var_grad_pre[-1]

# %%

# perform decomposition based onreadout weight matrix

weight_mat = model.readouts[0].weight.detach().cpu().clone().data.numpy()
print(weight_mat.shape)

C = orth(weight_mat.T)
CC_T = C @ C.T
N = null_space(weight_mat)
NN_T = N @ N.T


# %%

# print number of components needed to explain 99.9% of the variance for each matrix
print(
    np.where(exp_var_act > 0.999)[0][0],
    "components needed to explain 99.9% of the variance for activation matrix",
)
print(
    np.where(exp_var_act_pre > 0.999)[0][0],
    "components needed to explain 99.9% of the variance for activation pre matrix",
)
print(
    np.where(exp_var_grad > 0.999)[0][0],
    "components needed to explain 99.9% of the variance for gradient matrix",
)
print(
    np.where(exp_var_grad_pre > 0.999)[0][0],
    "components needed to explain 99.9% of the variance for gradient pre matrix",
)
print(C.shape[1], "dimensions in the range of the readout matrix")

dimensionalities = {
    "act": int(np.where(exp_var_act > 0.999)[0][0]),
    "act_pre": int(np.where(exp_var_act_pre > 0.999)[0][0]),
    "grad": int(np.where(exp_var_grad > 0.999)[0][0]),
    "grad_pre": int(np.where(exp_var_grad_pre > 0.999)[0][0]),
    "readout": int(C.shape[1]),
}

# %%

# get projection matrices into activation and gradient range
M_act = u_act[:, : np.where(exp_var_act > 0.999)[0][0]]
M_grad = u_grad[:, : np.where(exp_var_grad > 0.999)[0][0]]

# get projection matrices
P_act = M_act @ M_act.T
P_grad = M_grad @ M_grad.T

# %%

# after Task 0
model.load_state_dict(
    torch.load(os.path.join(rundir, "model_task_0.pt"), map_location=DEVICE)
)

# register forward hook to dense layer activations
embeddings, handle = register_hook_retain_grad(model.blocks[-1])

for batch in test_loader_t0:
    x, y = batch
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    model(x, 0)

embeddings_t0 = torch.cat(embeddings, dim=0).detach().cpu()
handle.remove()

print(f"collected embeddings {embeddings_t0.shape} after training on task 1")


# for the test set of the first task, comput embeddings after training on task 1 and task 2
model.load_state_dict(
    torch.load(os.path.join(rundir, "model_task_1.pt"), map_location=DEVICE)
)

# register forward hook to dense layer activations
embeddings, handle = register_hook_retain_grad(model.blocks[-1])

for batch in test_loader_t0:
    x, y = batch
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    model(x, 0)

embeddings_t1 = torch.cat(embeddings, dim=0).detach().cpu()

print(f"collected embeddings {embeddings_t1.shape} after training on task 1")

# %%

# project embeddings into subspaces and compute how far they moved

distances = torch.norm(embeddings_t0 - embeddings_t1, dim=1)
print("mean distance between embeddings after training on task 1:", distances.mean())

# %%

# project embeddings into range of gradients and activations
emb_t0_grad_range = embeddings_t0 @ P_grad
emb_t1_grad_range = embeddings_t1 @ P_grad
emb_t0_act_range = embeddings_t0 @ P_act
emb_t1_act_range = embeddings_t1 @ P_act

emb_t0_grad_null = embeddings_t0 - emb_t0_grad_range
emb_t1_grad_null = embeddings_t1 - emb_t1_grad_range
emb_t0_act_null = embeddings_t0 - emb_t0_act_range
emb_t1_act_null = embeddings_t1 - emb_t1_act_range

# compute distances
distances_grad_range = torch.norm(emb_t0_grad_range - emb_t1_grad_range, dim=1)
distances_grad_null = torch.norm(emb_t0_grad_null - emb_t1_grad_null, dim=1)
distances_act_range = torch.norm(emb_t0_act_range - emb_t1_act_range, dim=1)
distances_act_null = torch.norm(emb_t0_act_null - emb_t1_act_null, dim=1)

print("mean distance between embeddings after training on task 1:", distances.mean())
print(
    "mean distance between embeddings after training on task 1 in gradient range:",
    distances_grad_range.mean(),
)
print(
    "mean distance between embeddings after training on task 1 in gradient null:",
    distances_grad_null.mean(),
)
print(
    "mean distance between embeddings after training on task 1 in activation range:",
    distances_act_range.mean(),
)
print(
    "mean distance between embeddings after training on task 1 in activation null:",
    distances_act_null.mean(),
)

# make sure results are floats
results = {
    "distances": float(distances.mean()),
    "distances_grad_range": float(distances_grad_range.mean()),
    "distances_grad_null": float(distances_grad_null.mean()),
    "distances_act_range": float(distances_act_range.mean()),
    "distances_act_null": float(distances_act_null.mean()),
}

# save to rundir as yaml
with open(os.path.join(rundir, "activation_gradient_results.yaml"), "w") as f:
    yaml.dump(results, f)

# save dimensionalities to rundir as yaml
with open(os.path.join(rundir, "dimensionalities.yaml"), "w") as f:
    yaml.dump(dimensionalities, f)

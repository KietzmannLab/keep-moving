from tqdm import tqdm
import torch
from load_data import map_classes_for_task
from scipy.linalg import svd

import numpy as np
from sklearn.decomposition import PCA

import wandb


def compute_subspace(
    model, criterion, train_loader, task, classes, epochs, var_exp=0.999, log_wandb=True
):
    # generate gradients and save as numpy array
    grads = [[] for _ in model.blocks.parameters()]
    for i in range(epochs):
        for x, y in tqdm(train_loader):
            y = map_classes_for_task(y, classes)
            x, y = x.to(model.device), y.to(model.device)
            model.train()
            model.zero_grad()
            y_hat = model(x, task)
            loss = criterion(y_hat, y)
            g = torch.autograd.grad(loss, model.blocks.parameters())
            g = [x.detach().cpu().numpy().flatten() for x in g]
            for i, grad in enumerate(g):
                grads[i].append(grad)
    grads = [np.array(gs) for gs in grads]
    # pca on gradients, keeping var_exp components
    grad_subspaces = []
    for i, grad_dset in enumerate(grads):
        print(f"computing subspace for layer {i+1}/{len(grads)}")
        grad_dset = grad_dset.T
        U, V, T = svd(grad_dset)
        # get variance explained
        var = np.cumsum(V**2) / np.sum(V**2)
        n_components = np.where(var > var_exp)[0][0]
        print(f"keeping {n_components} / {grad_dset.shape[0]} components")
        # log n components
        if log_wandb:
            wandb.log({f"subspaces/subspace_{i}": n_components}, commit=False)

        U = U[:, :n_components]
        grad_subspaces.append(torch.tensor(U))

    return grad_subspaces


@torch.no_grad()
def project_gradients(subspaces, grads, knobs=None):
    # get grad shape
    grads_projected = []
    for grad, subspace in zip(grads, subspaces):
        grad = grad.clone()  # make sure no in place modification is happening
        grad_shape = grad.shape
        grad_device = grad.device
        subspace = subspace.to(grad_device)
        grad = grad.reshape(-1)
        projection = (subspace @ subspace.T) @ grad[:, None]
        projection = projection.reshape(grad_shape)
        grad = grad.reshape(grad_shape)
        grad -= projection
        grads_projected.append(grad)
    grads_projected = maybe_use_knobs(grads, grads_projected, knobs)
    return grads_projected


@torch.no_grad()
def project_gradients_on_vector(grads, target_grads, knobs=None, t0_bias=None):
    projected_grads = []
    for g, tg in zip(grads, target_grads):
        gshape = g.shape
        g = g.detach().clone().view(-1)
        tg = tg.detach().clone().view(-1)
        tg_norm = tg / torch.norm(tg)
        g_range = (g @ tg_norm) * tg_norm
        g_null = g - g_range
        projected_grads.append(g_null.view(gshape))
    projected_grads = maybe_use_knobs(grads, projected_grads, knobs)
    if t0_bias is not None:
        biased_grads = []
        for g, tg in zip(projected_grads, target_grads):
            g = (1 - t0_bias) * g + t0_bias * tg
            biased_grads.append(g)
        projected_grads = biased_grads
    return projected_grads


@torch.no_grad()
def maybe_use_knobs(grads, projected_grads, knobs):
    if knobs is None:
        return projected_grads
    else:
        frac_null, frac_range = knobs
        grads_knobs = []
        for task_grad, proj_grad in zip(grads, projected_grads):
            grad_null = proj_grad
            grad_range = task_grad - grad_null
            grad_knob = grad_null * frac_null + grad_range * frac_range
            grads_knobs.append(grad_knob)
        return grads_knobs

import torch
from sklearn.decomposition import PCA
import numpy as np


def project_gradient(subspaces, grad):
    """
    repeatedly filter out the component of the gradient that is in the subspace
    subspaces is a list of spaces to be rejected
    """
    for i, subspace in enumerate(subspaces):
        print(f"projection {i}")
        subspace = subspace.to(grad.device)
        projection = ((subspace @ grad).T @ subspace).T
        grad -= projection
    return grad

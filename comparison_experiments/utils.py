import torch
import torch.nn as nn
import numpy as np
import random
import pprint
from scipy.linalg import orthogonal_procrustes
from avalanche.training.determinism.rng_manager import RNGManager

from scipy.linalg import svd


def set_seed(seed):
    """
    make very sure we fix as many seeds as we can find
    """
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
    RNGManager.set_random_seeds(seed)


def format_model_dict(state_dict):
    print_dict = {key: state_dict[key].shape for key in state_dict}
    printstring = pprint.pformat(print_dict)
    return printstring


def procrustes_transform(mat1, mat2):
    """
    transform mat2 such that it is best aligned with mat1 using
    a Procrustes transformation.

    INPUTS:
        two matrices of same shape, where each row is an observation
    """
    # create output matrices
    mout1 = mat1.copy()
    mout2 = mat2.copy()

    # get translations for both matrices
    translate1 = np.mean(mout1, axis=0, keepdims=True)
    translate2 = np.mean(mout2, axis=0, keepdims=True)

    # get scale for both matrices
    norm1 = np.linalg.norm(mout1 - translate1)
    norm2 = np.linalg.norm(mout2 - translate2)

    # translate and norm
    mout1 = (mout1 - translate1) / norm1
    mout2 = (mout2 - translate2) / norm2

    # procrustes transform
    R, s = orthogonal_procrustes(mout1, mout2)
    mout2 = np.dot(mout2, R.T) * s

    # scale back up and translate back
    mout2 = mout2 * norm1 + translate1

    return mout2, R, s, norm1, translate1


def scaled_orth(A, rcond=None):
    u, s, vh = svd(A, full_matrices=False)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = u[:, :num]
    Q = Q * s[:num]
    return Q

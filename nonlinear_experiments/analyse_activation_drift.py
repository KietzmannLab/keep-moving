# %%  IMPORTS
from model import Model

# %% PARAMETERS

logdir = ""
run_name = ""

# %% HELPER FUNCTIONS


def decompose_readout_weights(weights):
    C = orth(weight_mat.T)
    CC_T = C @ C.T
    N = null_space(weight_mat)
    NN_T = N @ N.T

    in_plane_filter = CC_T
    out_of_plane_filter = NN_T

    return in_plane_filter, out_of_plane_filter


# %%

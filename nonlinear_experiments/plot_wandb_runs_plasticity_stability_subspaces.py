# %% IMPORTS

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from wandb_utils import *
from plotting import *

sns.set(style="ticks")

# %% SPECIFY WHICH RUNS TO LOAD

runs = [
    "kietzmannlab/gradient_subspaces_new_knobs/g22jfbtd",  # baseline
    "kietzmannlab/gradient_subspaces_new_knobs/rtctgs34",  # subspace null 1.0 range 0.0
    "kietzmannlab/gradient_subspaces_new_knobs/06arrsik",  # subspace null 1.0 range 0.05
    "kietzmannlab/gradient_subspaces_new_knobs/u2vzxngg",  # subspace null 1.0 range 0.1
    "kietzmannlab/gradient_subspaces_new_knobs/8twk325w",  # subspace null 1.0 range 0.2
    "kietzmannlab/gradient_subspaces_new_knobs/mw9rkze0",  # subspace null 1.0 range 0.3
    "kietzmannlab/gradient_subspaces_new_knobs/u1ie20dn",  # subspace null 1.0 range 0.4
    "kietzmannlab/gradient_subspaces_new_knobs/9u0bdjau",  # subspace null 0.9 range 0.0
    "kietzmannlab/gradient_subspaces_new_knobs/laph18ka",  # subspace null 0.8 range 0.0
    "kietzmannlab/gradient_subspaces_new_knobs/r7bhqiyx",  # subspace null 0.7 range 0.0
    "kietzmannlab/gradient_subspaces_new_knobs/nnseac8q",  # subspace null 0.5 range 0.0
    "kietzmannlab/gradient_subspaces_new_knobs/o9rkxjur",  # subspace null 0.4 range 0.0
    "kietzmannlab/gradient_subspaces_new_knobs/rg02rqz4",  # subspace null 0.2 range 0.0
    "kietzmannlab/gradient_subspaces_new_knobs/n0r7ehls",  # subspace null 0.1 range 0.0
    "kietzmannlab/gradient_subspaces_new_knobs/m22uilbm",  # subspace null 0.0 range 0.0
    "kietzmannlab/gradient_subspaces_new/dhjx41rd",  # ewc 0.001
    "kietzmannlab/gradient_subspaces_new/a1cdj8le",  # ewc 0.01
    "kietzmannlab/gradient_subspaces_new/nzo53dxk",  # ewc 0.1
    "kietzmannlab/gradient_subspaces_new/m0ostm8u",  # ewc 1.0
    "kietzmannlab/gradient_subspaces_new/o9iqstp7",  # ewc 10.0
    "kietzmannlab/gradient_subspaces_new/vitf4mm7",  # ewc 100.0
    "kietzmannlab/gradient_subspaces_new/apiy2tkp",  # ewc 1000.0
    "kietzmannlab/gradient_subspaces_new/m7hs1ptf",  # ewc 10000.0
    "kietzmannlab/gradient_subspaces_new/8b6x66h7",  # ewc 100000.0
    "kietzmannlab/gradient_subspaces_new_knobs/jaes08gp",  # gradient subspace recomp. 25
    "kietzmannlab/gradient_subspaces_new_knobs/jggkzzkf",  # gradient subspace recomp. 50
    "kietzmannlab/gradient_subspaces_new_knobs/cd2odc2u",  # gradient subspace recomp. 200
    "kietzmannlab/gradient_subspaces_new_knobs/slmow5bv",  # gradient subspace recomp. 400
    "kietzmannlab/gradient_subspaces_new_knobs/794i2iaj",  # subspace legal
]

projectdir = "gradient_subspaces_knobs_new_adam"

print(runs)

# %%

print("loading data")
data = [load_history(is_wandb=True, projectdir=projectdir, logpath=run) for run in runs]
print("done.")
dfs, paths, wandb_names = list(zip(*data))

# %% NAME RUNS

names = [
    "baseline",
    "subspace null 1.0 range 0.0",
    "subspace null 1.0 range 0.05",
    "subspace null 1.0 range 0.1",
    "subspace null 1.0 range 0.2",
    "subspace null 1.0 range 0.3",
    "subspace null 1.0 range 0.4",
    "subspace null 0.9 range 0.0",
    "subspace null 0.8 range 0.0",
    "subspace null 0.7 range 0.0",
    "subspace null 0.5 range 0.0",
    "subspace null 0.4 range 0.0",
    "subspace null 0.2 range 0.0",
    "subspace null 0.1 range 0.0",
    "subspace null 0.0 range 0.0",
    "ewc 0.001",
    "ewc 0.01",
    "ewc 0.1",
    "ewc 1.0",
    "ewc 10.0",
    "ewc 100.0",
    "ewc 1000.0",
    "ewc 10000.0",
    "ewc 100000.0",
    "gradient subspace recomp. 25",
    "gradient subspace recomp. 50",
    "gradient subspace recomp. 200",
    "gradient subspace recomp. 400",
    "subspace legal",
]

# %%  SET UP FIGURE SAVEDIR

figdir = os.path.join("/Users/daniel/Desktop/subspace_figs")

# %% PLOT TRAINING CURVES AS SANITY CHECK

colors = sns.color_palette("colorblind", len(dfs))
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes = axes.flatten()
for df, name, color in zip(dfs, names, colors):
    t0_acc = df["performance/test_accuracy_task_0"].dropna()
    t1_acc = df["performance/test_accuracy_task_1"].dropna()
    epoch = np.arange(len(t0_acc))
    axes[0].plot(epoch, t0_acc, marker="o", c=color, alpha=1.0)
    axes[1].plot(epoch, t1_acc, label=name, marker="o", c=color, alpha=1.0)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
sns.despine()
# legend under plot
plt.tight_layout()
plt.legend(bbox_to_anchor=(0.1, -0.12), loc="upper center", ncol=5)
axes[0].set_title("task 0 test accuracy")
axes[1].set_title("task 1 test accuracy")
plt.show()

# %%

ewc_stabilities = [
    df["performance/test_accuracy_task_0"].dropna().values[-1] for df in dfs[15:24]
]
ewc_plasticities = [
    df["performance/test_accuracy_task_1"].dropna().values[-1] for df in dfs[15:24]
]
ewc_lambdas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]

subspace_null_stabilities = [
    df["performance/test_accuracy_task_0"].dropna().values[-1] for df in dfs[1:7]
]
subspace_null_plasticities = [
    df["performance/test_accuracy_task_1"].dropna().values[-1] for df in dfs[1:7]
]
subspace_null_knob = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]

subspace_range_stabilities = [
    df["performance/test_accuracy_task_0"].dropna().values[-1] for df in dfs[7:15]
]
subspace_range_plasticities = [
    df["performance/test_accuracy_task_1"].dropna().values[-1] for df in dfs[7:15]
]
subspace_range_knob = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]

baseline_stability = dfs[0]["performance/test_accuracy_task_0"].dropna().values[-1]
baseline_plasticity = dfs[0]["performance/test_accuracy_task_1"].dropna().values[-1]

nullspace_recompute_stabilities = [
    df["performance/test_accuracy_task_0"].dropna().values[-1] for df in dfs[24:28]
]
nullspace_recompute_plasticities = [
    df["performance/test_accuracy_task_1"].dropna().values[-1] for df in dfs[24:28]
]
nullspace_recompute_freq = [25, 50, 200, 400]

subspace_legal_stability = (
    dfs[28]["performance/test_accuracy_task_0"].dropna().values[-1]
)
subspace_legal_plasticity = (
    dfs[28]["performance/test_accuracy_task_1"].dropna().values[-1]
)

# get LogNorm for cmap
import matplotlib as mpl

ewc_norm = mpl.colors.LogNorm(vmin=ewc_lambdas[0], vmax=ewc_lambdas[-1])

plt.figure(figsize=(8, 8))
plt.scatter(
    ewc_stabilities,
    ewc_plasticities,
    label="ewc",
    cmap="Greens",
    c=ewc_lambdas,
    norm=ewc_norm,
    marker="o",
    s=100,
    edgecolors="black",
    linewidth=1.5,
)  # ewc
plt.scatter(
    subspace_null_stabilities,
    subspace_null_plasticities,
    label="subspace null",
    cmap="Blues",
    c=subspace_null_knob,
    marker=">",
    s=100,
    edgecolors="black",
    linewidth=1.5,
)  # subspace null
plt.scatter(
    subspace_range_stabilities,
    subspace_range_plasticities,
    label="subspace range",
    cmap="Reds",
    c=subspace_range_knob,
    marker="^",
    s=100,
    edgecolors="black",
    linewidth=1.5,
)  # subspace range
plt.scatter(
    nullspace_recompute_stabilities,
    nullspace_recompute_plasticities,
    label="subspace recomp.",
    cmap="Purples",
    c=nullspace_recompute_freq,
    marker="s",
    s=100,
    edgecolors="black",
    linewidth=1.5,
)  # subspace recomp.
plt.scatter(
    subspace_legal_stability,
    subspace_legal_plasticity,
    label="subspace legal",
    color="orange",
    marker="*",
    s=200,
    edgecolors="black",
    linewidth=1.5,
)  # subspace legal
plt.scatter(
    baseline_stability,
    baseline_plasticity,
    label="baseline",
    color="black",
    marker="*",
    s=200,
    edgecolors="black",
    linewidth=1.5,
)  # baseline
plt.legend()
plt.xlabel("stability")
plt.ylabel("plasticity")
sns.despine()
plt.tight_layout()
plt.savefig("/Users/daniel/Desktop/subspace_performance.png", dpi=400)

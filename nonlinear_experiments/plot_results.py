# %%

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import pandas as pd

sns.set_context("paper", font_scale=1.5)
sns.set_style("ticks")

# %%

datadir = "analysis_dfs/analysis_multi_seed"
names = [
    "baseline",
    "ewc",
    "subspace_estimation_vary_range",
    "subspace_estimation_vary_null",
]

cs = {
    "ewc": "ewc_lambda",
    "subspace_estimation_vary_range": "knob_range",
    "subspace_estimation_vary_null": "knob_null",
}

plot_opts_per_df = {
    "baseline": {
        "color": "white",
        "edgecolor": "black",
        "marker": "*",
        "s": 150,
        "label": "baseline",
    },
    "ewc": {
        "cmap": "Greens",
        "norm": LogNorm(),
        "edgecolor": "black",
        "marker": "o",
        "s": 50,
        "label": "ewc",
    },
    "subspace_estimation_vary_range": {
        "cmap": "Reds",
        "edgecolor": "black",
        "marker": "s",
        "s": 50,
        "label": r"range weight $\alpha$",
    },
    "subspace_estimation_vary_null": {
        "cmap": "Blues",
        "edgecolor": "black",
        "marker": "^",
        "s": 50,
        "label": r"null space weight $\beta$",
    },
}

legend_order = None

# %%

# load all dataframes in datadir
df_list = []
for name in names:
    filename = os.path.join(datadir, f"{name}.csv")
    df = pd.read_csv(filename)
    # switch on names: for ewc, group by and mean per ewc_lambda, for baseline take mean over all
    # for subspace_estimation_vary_range, group by and mean per knob_range
    # for subspace_estimation_vary_null, group by and mean per knob_null
    if name == "ewc":
        df = df.groupby(cs[name]).mean(numeric_only=True).reset_index()
    elif name == "baseline":
        df = df.mean(numeric_only=True).to_frame().T
    elif name == "subspace_estimation_vary_range":
        df = df.groupby(cs[name]).mean(numeric_only=True).reset_index()
    elif name == "subspace_estimation_vary_null":
        df = df.groupby(cs[name]).mean(numeric_only=True).reset_index()
    df_list.append(df)


# %%  plot acc_after_t1_t0 as stability against acc_after_t1_t1 as plasticity

fig_perf = plt.figure(figsize=(8, 8))
for df, name in zip(df_list, names):
    print(name)
    if name in plot_opts_per_df.keys():
        plot_opts = plot_opts_per_df[name]
        print(plot_opts)
    else:
        plot_opts = {}
    if name in cs.keys():
        plot_opts["c"] = df[cs[name]]
    plt.scatter(df["acc_after_t1_t1"], df["acc_after_t1_t0"], **plot_opts)

plt.xlabel("Plasticity (task 1 performance)")
plt.ylabel("Stability (task 2 performance)")
sns.despine()
plt.legend()

if legend_order is not None:
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        [handles[idx] for idx in legend_order], [labels[idx] for idx in legend_order]
    )

# plot mean_distance_null vs mean_distance_range

fig_displace = plt.figure(figsize=(8, 8))
for df, name in zip(df_list, names):
    if name in plot_opts_per_df.keys():
        plot_opts = plot_opts_per_df[name]
    else:
        plot_opts = {}
    if name in cs.keys():
        plot_opts["c"] = df[cs[name]]
    plt.scatter(df["mean_distance_range"], df["mean_distance_null"], **plot_opts)
plt.xlabel("Displacement in readout range")
plt.ylabel("Displacement in readout null space")
sns.despine()
plt.legend()
if legend_order is not None:
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        [handles[idx] for idx in legend_order], [labels[idx] for idx in legend_order]
    )

# %%

# save both
figdir = os.path.join(datadir, "figures")
# ensure exists
if not os.path.exists(figdir):
    os.makedirs(figdir)
fig_perf.savefig(os.path.join(figdir, "perf.pdf"))
fig_displace.savefig(os.path.join(figdir, "displace.pdf"))

# %%

# repeat displacement plot for distances_act_range vs distances_act_null
# and distances_grad_range vs distances_grad_null

fig_displace_act = plt.figure(figsize=(8, 8))
for df, name in zip(df_list, names):
    if name in plot_opts_per_df.keys():
        plot_opts = plot_opts_per_df[name]
    else:
        plot_opts = {}
    if name in cs.keys():
        plot_opts["c"] = df[cs[name]]
    plt.scatter(df["distances_act_range"], df["distances_act_null"], **plot_opts)
plt.xlabel("Displacement in activation range")
plt.ylabel("Displacement in activation null space")
sns.despine()
plt.legend()

fig_displace_grad = plt.figure(figsize=(8, 8))
for df, name in zip(df_list, names):
    if name in plot_opts_per_df.keys():
        plot_opts = plot_opts_per_df[name]
    else:
        plot_opts = {}
    if name in cs.keys():
        plot_opts["c"] = df[cs[name]]
    plt.scatter(df["distances_grad_range"], df["distances_grad_null"], **plot_opts)
plt.xlabel("Displacement in gradient range")
plt.ylabel("Displacement in gradient null space")
sns.despine()
plt.legend()

# %%
# save both
figdir = os.path.join(datadir, "figures")
# ensure exists
if not os.path.exists(figdir):
    os.makedirs(figdir)
fig_displace_act.savefig(os.path.join(figdir, "displace_act.pdf"))
fig_displace_grad.savefig(os.path.join(figdir, "displace_grad.pdf"))


# plot all three displacements in one plot


fig_displace_all = plt.figure(figsize=(12, 8))

for df, name in zip(df_list, names):
    if name in plot_opts_per_df.keys():
        plot_opts = plot_opts_per_df[name]
    else:
        plot_opts = {}
    if name in cs.keys():
        plot_opts["c"] = df[cs[name]]

    plot_opts["marker"] = "o"
    plot_opts["label"] = f"{name} (readout)"
    plt.scatter(
        df["mean_distance_range"],
        df["mean_distance_null"],
        **plot_opts,
    )

    plot_opts["marker"] = "^"
    plot_opts["label"] = f"{name} (activation)"
    plt.scatter(
        df["distances_act_range"],
        df["distances_act_null"],
        **plot_opts,
    )

    plot_opts["marker"] = "s"
    plot_opts["label"] = f"{name} (gradient)"
    plt.scatter(
        df["distances_grad_range"],
        df["distances_grad_null"],
        **plot_opts,
    )

plt.xlabel("Displacement in range")
plt.ylabel("Displacement in null space")

sns.despine()
plt.legend()

# move legend outside
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

# equal axes
plt.gca().set_aspect("equal", "box")

# save
fig_displace_all.savefig(os.path.join(figdir, "displace_all.pdf"), bbox_inches="tight")

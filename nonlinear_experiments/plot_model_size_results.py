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

datadir = "analysis_dfs/analysis_model_size"

# %%

# load all dataframes in datadir
df_list = []
names = []
for filename in os.listdir(datadir):
    if filename.endswith(".csv"):
        df_list.append(pd.read_csv(os.path.join(datadir, filename)))
        names.append(filename[:-4])

# %%

plt.figure()
# plt.scatter(
#     df_list[0]["acc_after_t1_t0"], df_list[0]["acc_after_t1_t1"], label="baseline"
# )

plt.scatter(
    df_list[1]["acc_after_t1_t0"] - df_list[1]["acc_after_t0_t0"],
    df_list[1]["acc_after_t1_t1"],
    label="subspace lock",
)


plt.xlabel("stability")
plt.ylabel("plasticity")
plt.legend()
plt.show()

# %%

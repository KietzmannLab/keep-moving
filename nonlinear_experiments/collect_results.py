# %%
import os
import pandas as pd
import yaml

# %%

projects = [
    "baseline",
    "subspace_estimation_vary_null_50",
    "subspace_estimation_vary_null",
    "subspace_estimation_vary_range",
    "subspace_estimation_legal",
    "subspace_estimation_vary_size",
    "ewc",
]

logdir = "logs"
outdir = "analysis_dfs"

# check that folder exists, otherwise create
if not os.path.exists(outdir):
    os.makedirs(outdir)

# %%


def load_data_for_run(runpath):
    cfgpath = os.path.join(runpath, "cfg.yaml")
    resultpath = os.path.join(runpath, "results.yaml")
    activationgradientresultpath = os.path.join(
        runpath, "activation_gradient_results.yaml"
    )

    with open(cfgpath, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    with open(resultpath, "r") as f:
        results = yaml.load(f, Loader=yaml.FullLoader)

    with open(activationgradientresultpath, "r") as f:
        activation_gradient_results = yaml.load(f, Loader=yaml.FullLoader)

    return cfg, results, activation_gradient_results


def load_results_for_project(logdir, projdir):
    results = []
    activation_gradient_results = []
    cfgs = []
    names = []
    folders = os.listdir(os.path.join(logdir, projdir))
    for folder in folders:
        runpath = os.path.join(logdir, projdir, folder)
        try:
            cfg, result, activation_gradient_result = load_data_for_run(runpath)
            results.append(result)
            cfgs.append(cfg)
            names.append(folder)
            activation_gradient_results.append(activation_gradient_result)
        except FileNotFoundError:
            print(f"Skipping {runpath}")
            continue
    return results, activation_gradient_results, cfgs, names


def create_proj_df(results, activation_gradient_results, cfgs, names):
    for r, agr, c, n in zip(results, activation_gradient_results, cfgs, names):
        r.update(c)
        r.update(agr)
        r["name"] = n
    df = pd.DataFrame(results)
    return df


# create df for each project
for proj in projects:
    print(f"Creating df for {proj}")
    results, activation_gradient_results, cfgs, names = load_results_for_project(
        logdir, proj
    )
    df = create_proj_df(results, activation_gradient_results, cfgs, names)
    df.to_csv(os.path.join(outdir, f"{proj}.csv"), index=False)

print("done.")

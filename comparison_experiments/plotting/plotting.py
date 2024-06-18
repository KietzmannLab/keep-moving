import yaml
from os import path
import numpy as np
import scipy
import matplotlib.pyplot as plt


def load_performance_data(filepath):
    """
    load performance dictionary for a training run
    """
    filepath = path.join(filepath, "original_readout_results.yaml")
    with open(filepath, "r") as file:
        results = yaml.load(file, Loader=yaml.Loader)
    return results


def load_original_performance_data(filepath):
    filepath = path.join(filepath, "results.yaml")


def load_aligned_data(filepath, val=False):
    """
    load performance dictionary for a training run
    """
    if not val:
        filepath = path.join(filepath, "aligned_readout_results.yaml")
    else:
        filepath = path.join(filepath, "validation_aligned_readout_results.yaml")
    with open(filepath, "r") as file:
        results = yaml.load(file, Loader=yaml.Loader)
    return results


def load_finetuning_data(filepath, ntasks):
    """
    load finetuning data for a training run
    """
    finetuning_dict = {task: [] for task in range(ntasks)}
    for phase in range(ntasks):
        fname = path.join(filepath, f"finetuning_results_phase_{phase}.yaml")
        with open(fname, "r") as finetune_file:
            res = yaml.load(finetune_file, Loader=yaml.Loader)
            for k in res.keys():
                finetuning_dict[k].append(res[k])
    return finetuning_dict


def load_ridge_data(filepath, val=False):
    """
    load performance dictionary for a training run
    """
    if not val:
        filepath = path.join(filepath, "ridge_aligned_readout_results.yaml")
    else:
        filepath = path.join(filepath, "ridge_validation_aligned_readout_results.yaml")
    with open(filepath, "r") as file:
        results = yaml.load(file, Loader=yaml.Loader)
    return results


def load_linear_drift_estimation_data(filepath):
    """
    load performance dictionary for linear drift estimation
    """
    filepath = path.join(filepath, "linear_drift_compensation_results.yaml")
    with open(filepath, "r") as file:
        results = yaml.load(file, Loader=yaml.Loader)

    # no data recorded for t=0, so add a padding item in its place
    return results


def align_tasks(performance_dict, mode="full"):
    """
    takes a perfromance dictionary and aligns runs at task onset
    """
    keys = list(performance_dict.keys())
    nphases = len(performance_dict[keys[0]])
    ntasks = len(keys)
    relative_phase = list(range(-nphases + 1, nphases, 1))

    if mode == "after onset":
        aligned_data = np.empty(shape=(ntasks, nphases))
        aligned_data.fill(np.nan)
        for i, k in enumerate(keys):
            data = performance_dict[k][k:]
            aligned_data[i, : len(data)] = data

        pad_nans = np.empty(shape=(ntasks, nphases - 1))
        pad_nans.fill(np.nan)
        aligned_data = np.concatenate((pad_nans, aligned_data), axis=1)

    elif mode == "full":
        aligned_data = np.empty(shape=(ntasks, 2 * nphases - 1))
        aligned_data.fill(np.nan)
        for i, k in enumerate(keys):
            start = nphases - k - 1
            end = start + nphases
            data = performance_dict[k]
            aligned_data[i, start:end] = data

    elif mode == "linear_drift_estimator":
        nphases += 1
        aligned_data = np.empty(shape=(ntasks, 2 * nphases - 1))
        aligned_data.fill(np.nan)
        for i, k in enumerate(keys):
            start = nphases
            end = start + len(performance_dict[k])
            data = performance_dict[k]
            aligned_data[i, start:end] = data

    return aligned_data, relative_phase


def average_over_tasks(aligned_data):
    return np.nanmean(aligned_data, axis=0)


def average_over_runs(aligned_data):
    means = np.nanmean(aligned_data, axis=0)
    std = np.nanstd(aligned_data, axis=0)
    sem = scipy.stats.sem(aligned_data, axis=0, nan_policy="omit")
    return means, std, sem


class RunData:
    def __init__(self, run_path, pretty_name, color, ntasks):
        self.performance_data, self.relative_phase = align_tasks(
            load_performance_data(run_path)
        )
        try:
            self.finetuning_data, _ = align_tasks(
                load_finetuning_data(run_path, ntasks)
            )
        except:
            print(f"no finetuning data for {pretty_name}")
        try:
            self.aligned_data, _ = align_tasks(load_aligned_data(run_path))
        except:
            print(f"no alignment data for {pretty_name}")
        try:
            self.validation_aligned_data, _ = align_tasks(
                load_aligned_data(run_path, val=True)
            )
        except:
            print(f"no alignment data for {pretty_name}")
        try:
            self.ridge_aligned_data, _ = align_tasks(
                load_ridge_data(run_path, val=False)
            )
        except:
            print(f"no ridge data for {pretty_name}")
        try:
            self.validation_ridge_aligned_data, _ = align_tasks(
                load_ridge_data(run_path, val=True)
            )
        except:
            print(f"no ridge data for {pretty_name}")
        try:
            self.linear_drift_estimation_data, _ = align_tasks(
                load_linear_drift_estimation_data(run_path),
                mode="linear_drift_estimator",
            )
            self.linear_drift_estimation_data[
                :, self.linear_drift_estimation_data.shape[1] // 2
            ] = self.performance_data[
                :, self.linear_drift_estimation_data.shape[1] // 2
            ]
        except:
            print(f"no linear drift estimation data for {pretty_name}")

        self.color = color
        self.name = pretty_name


def plot_run_average(
    runs,
    errorbar="none",
    task="average",
    mode="continual",
    color="tab:blue",
    first=0,
    last=-1,
    lims=None,
    **kwargs,
):
    """
    plot average over data from runs, relative phase indices must be the same for all runs
    """
    run_arrs = []
    xs = runs[0].relative_phase
    if mode == "continual":
        run_arrs = [run.performance_data for run in runs]
    elif mode == "finetuned":
        run_arrs = [run.finetuning_data for run in runs]
    elif mode == "aligned":
        run_arrs = [run.aligned_data for run in runs]
    elif mode == "ridge":
        run_arrs = [run.ridge_aligned_data for run in runs]
    elif mode == "linear_estimation":
        run_arrs = [run.linear_drift_estimation_data for run in runs]
    if task == "average":
        run_arrs = [average_over_tasks(run) for run in run_arrs]
    else:
        run_arrs = [run[task] for run in run_arrs]

    run_arrs = run_arrs[first:last]

    means, std, sem = average_over_runs(run_arrs)

    if not lims is None:
        xs = xs[lims[0] : lims[1]]
        means = means[lims[0] : lims[1]]
        sem = sem[lims[0] : lims[1]]
        std = std[lims[0] : lims[1]]

    plt.plot(xs, means, color=color, **kwargs)

    if errorbar == "sem":
        plt.fill_between(xs, means - sem, means + sem, color=color, alpha=0.3)
    elif errorbar == "std":
        plt.fill_between(xs, means - std, means + std, color=color, alpha=0.3)

    ticks = np.array(xs)[np.logical_not(np.isnan(means))]
    plt.xticks(ticks)


def plot_task(run, task="average", mode="continual", relative=False, **kwargs):
    if mode == "continual":
        data = run.performance_data
    elif mode == "finetuned":
        data = run.finetuning_data
    elif mode == "aligned":
        data = run.aligned_data
    elif mode == "val_aligned":
        data = run.validation_aligned_data
    elif mode == "ridge":
        data = run.ridge_aligned_data
    elif mode == "val_ridge":
        data = run.validation_ridge_aligned_data
    elif mode == "linear_drift_estimation":
        data = run.linear_drift_estimation_data

    if task == "average":
        data = average_over_tasks(data)
    else:
        data = data[task]

    if relative:
        data -= data[len(data) // 2]

    plt.plot(run.relative_phase, data, **kwargs)


def dedup_legend():
    """
    generates a legend for the current axis, removing duplicate entries
    source: https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
    """
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # this removes duplicate legend handles
    plt.legend(by_label.values(), by_label.keys())

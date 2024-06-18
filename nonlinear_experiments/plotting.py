import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_cosine_with_t0(axes, history, weight_names=[], split="eval", plot_kwargs={}):
    # collect metrics
    keys = []
    for name in weight_names:
        keys.append(f"gradient cosines t0/{name}.weight t0 cosim")

    # plot
    axes = axes.flatten()
    for i, key in enumerate(keys):
        ax = axes[i]
        key_split = key + " " + split
        ys = history[key_split].dropna()
        batch = np.arange(len(ys))
        ax.plot(batch, ys, **plot_kwargs)
        ax.set_title(key)
        ax.set_xlabel("batch")
        ax.set_ylabel("cosine similarity")


def plot_t0_performance(history, split="eval", plot_kwargs={}):
    ys = history[f"performance/t0 {split} sample accuracy"].dropna()
    batch = np.arange(len(ys))
    plt.plot(batch, ys, **plot_kwargs)


def plot_test_accuracy(history):
    t0_acc = history["performance/test_accuracy_task_0"].dropna()
    t1_acc = history["performance/test_accuracy_task_1"].dropna()
    epoch = np.arange(len(t0_acc))
    plt.plot(epoch, t0_acc, label="task 0", marker="o")
    plt.plot(epoch, t1_acc, label="task 1", marker="o")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    sns.despine()

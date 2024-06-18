import torch
from avalanche.models.utils import avalanche_model_adaptation
from avalanche.evaluation.metrics import Accuracy

import hydra
from time import time
import os
from os import path
import random
import logging
from termcolor import colored
import yaml
from omegaconf import OmegaConf
import numpy as np
import re

import utils
from models.models import instantiate_model
from benchmarks.benchmarks import instantiate_benchmark
from benchmarks import dataset_utils

logger = logging.getLogger("procrustes analysis")


def log_info(msg):
    logger.info(colored(msg, "blue"))


@hydra.main(
    version_base=None, config_path="configuration/analysis_conf", config_name="default"
)
def experiment(config):
    timestamp = str(time())
    config.environment.timestamp = timestamp

    # set and track seed for reproducibility
    if not type(config.environment.seed) is int:
        seed = config.environment.seed
        utils.set_seed(seed)
    else:
        seed = random.randint(0, 99999999)
        config.environment.seed = seed
        utils.set_seed(seed)
    # also set same seed for data
    config.scenario.seed = seed

    # set compute device
    device = torch.device(
        f"cuda:{config.environment.cuda}"
        if torch.cuda.is_available() and config.environment.cuda >= 0
        else "cpu"
    )
    log_info(f"running on {device}")

    # get config from run we are finetuning for
    conf_path = path.join(
        config.environment.logdir, config.project, config.run_name, "config.yaml"
    )
    log_info(f"Loading run configuration at: {conf_path}")
    run_config = OmegaConf.load(conf_path)

    # load data with same config as during training
    config_class_order = dataset_utils.format_class_order_from_config(
        run_config.experiment.scenario
    )
    log_info(f"loading data with class order: {config_class_order}")
    run_config.experiment.scenario.fixed_class_order = (
        dataset_utils.format_class_order_from_config(run_config.experiment.scenario)
    )
    run_config.experiment.scenario.fixed_class_order = ""

    benchmark = instantiate_benchmark(run_config.experiment.scenario)
    log_info(f"number of experiences: {len(benchmark.train_stream)}")
    benchmark_class_order = dataset_utils.format_class_order_from_benchmark(
        benchmark.original_classes_in_exp, name=run_config.experiment.scenario.name
    )
    assert (
        config_class_order == benchmark_class_order
    ), f"Inconsistent class order! Expected: {config_class_order} Got: {benchmark_class_order}"

    # get same model and strategy configuration as during training
    acc_metric = Accuracy()
    target_model = instantiate_model(run_config.experiment.model)
    target_model.eval()
    # make sure fields for all classification heads exist before loading weights
    for experience in benchmark.train_stream:
        avalanche_model_adaptation(target_model, experience)

    # ------------------ ANALYSIS ------------------

    task = config.analysis_task
    embedding_layer = config.embedding_layer
    ntasks = run_config.experiment.scenario.n_experiences

    # create folder for analysis dataset. In run log directory, create an analysis folder. In this folder create a folder per task. For each task, create a folder per phase
    analysis_dir = path.join(
        config.environment.logdir, config.project, config.run_name, "analysis"
    )
    if not path.exists(analysis_dir):
        os.mkdir(analysis_dir)
    task_dir = path.join(analysis_dir, f"task_{task}")
    if not path.exists(task_dir):
        os.mkdir(task_dir)
    for t in range(ntasks):
        if not path.exists(path.join(task_dir, f"phase_{t}")):
            os.mkdir(path.join(task_dir, f"phase_{t}"))

    test_experiences = benchmark.test_stream

    # get all checkpoints files in a sorted list
    model_files = os.listdir(
        path.join(config.environment.logdir, config.project, config.run_name)
    )
    # get all files ending in pt
    model_files = [f for f in model_files if f.endswith(".pt")]
    # exclude initial model
    model_files = [f for f in model_files if not "init" in f]
    # sort by epoch
    pattern = re.compile(r"\d+")
    model_files_numbers = [int(pattern.findall(f)[0]) for f in model_files]
    idx = np.argsort(model_files_numbers)
    model_files = [model_files[i] for i in idx]

    # iterate over all checkpoints
    for i, model_file in enumerate(model_files):
        # load model
        model_path = path.join(
            config.environment.logdir, config.project, config.run_name, model_file
        )
        log_info(f"loading model from {model_path}")
        target_model.load_state_dict(torch.load(model_path, map_location=device))
        target_model.eval()
        # get dataset for test task
        experience = test_experiences[task]
        embeddings = []
        labels = []
        predictions = []
        # create dataloader
        dataloader = torch.utils.data.DataLoader(
            experience.dataset, batch_size=1024, shuffle=False
        )
        for x, y, t in dataloader:
            x = x.to(device)
            t = t.to(device)
            with torch.no_grad():
                target_model.eval()
                pred, aux = target_model.forward_with_auxiliary(x, t)
                predictions.append(np.argmax(pred.cpu().numpy(), axis=1))
                embedding = aux[embedding_layer].to("cpu").numpy()
                embeddings.append(embedding)
                labels.append(y.to("cpu").numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        # accuracy
        acc = np.sum(predictions == labels) / len(labels)
        log_info(f"accuracy: {acc}")
        # save embeddings and labels
        np.save(
            path.join(
                task_dir, f"phase_{i}", f"embeddings_layer_{embedding_layer}.npy"
            ),
            embeddings,
        )
        np.save(path.join(task_dir, f"phase_{i}", f"labels.npy"), labels)


if __name__ == "__main__":
    experiment()

"""
procrustes analysis for all tasks over time
"""
import torch
from torch.utils.data import DataLoader
from avalanche.models.utils import avalanche_model_adaptation
from avalanche.evaluation.metrics import Accuracy

import hydra
from time import time
from os import path
import random
import logging
from termcolor import colored
import yaml
from omegaconf import OmegaConf
import numpy as np

import utils
from models.models import instantiate_model
from benchmarks.benchmarks import instantiate_benchmark
from benchmarks.dataset_utils import (
    format_class_order_from_benchmark,
    format_class_order_from_config,
)

logger = logging.getLogger("procrustes analysis")


def log_info(msg):
    logger.info(colored(msg, "blue"))


@hydra.main(
    version_base=None, config_path="configuration/analysis_conf", config_name="default"
)
def experiment(config):
    timestamp = str(time())
    config.environment.timestamp = timestamp

    # set and track seed for reproducability
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
    config_class_order = format_class_order_from_config(run_config.experiment.scenario)
    log_info(f"loading data with class order: {config_class_order}")
    run_config.experiment.scenario.fixed_class_order = format_class_order_from_config(
        run_config.experiment.scenario
    )
    run_config.experiment.scenario.fixed_class_order = ""  # this seems necessary for the classifier to work as trained. I have not yet figured out why.
    # as long as the original class order was generated using the seed in the configuration dict, it will be regenerated correctly based on this seed.
    # TODO: figure out why this behaviour occurs. In the current implementation this script will fail if a task order was explicitly set while training.
    # we explicitly check whether the class order was correctly generated for now and throw an error if something is off:

    benchmark = instantiate_benchmark(run_config.experiment.scenario)
    log_info(f"number of experiences: {len(benchmark.train_stream)}")
    benchmark_class_order = format_class_order_from_benchmark(
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

    source_model = instantiate_model(run_config.experiment.model)
    source_model.eval()
    # make sure fields for all classification heads exist before loading weights
    for experience in benchmark.train_stream:
        avalanche_model_adaptation(source_model, experience)

    results_with_original_readout = {
        task: [] for task in range(run_config.experiment.scenario.n_experiences)
    }
    aligned_results = {
        task: [] for task in range(run_config.experiment.scenario.n_experiences)
    }
    val_aligned_results = {
        task: [] for task in range(run_config.experiment.scenario.n_experiences)
    }

    # iterate over tasks
    for i, (train_data, test_data) in enumerate(
        zip(benchmark.train_stream, benchmark.test_stream)
    ):
        log_info(f"Procrustes analysis for task {i}")
        # get target model and classifier
        model_weight_path = path.join(
            config.environment.logdir,
            config.project,
            config.run_name,
            f"model_phase_{i}.pt",
        )
        log_info(f"loading weights: {model_weight_path}")
        state_dict = torch.load(model_weight_path, map_location=device)
        target_model.load_state_dict(state_dict)

        target_classifier = target_model.classifier
        task_data_train = train_data.dataset
        task_loader_train = DataLoader(
            task_data_train, batch_size=config.environment.batchsize, shuffle=False
        )
        task_data_test = test_data.dataset
        task_loader_test = DataLoader(
            task_data_test, batch_size=config.environment.batchsize, shuffle=False
        )

        target_embeddings_train = []
        target_labels_train = []

        target_embeddings_test = []
        target_labels_test = []

        # make sure accuracy metric is fresh and model is in evaluation mode
        acc_metric.reset()
        target_model.eval()

        # get embeddings for training and test data. Compute performance on test set to verify loading was successful
        for Xtrain, ytrain, ttrain in task_loader_train:
            with torch.no_grad():
                _, act_train = target_model.forward_with_auxiliary(Xtrain, ttrain)
                target_embedding_train = act_train[config.embedding_layer]
            target_embeddings_train.append(target_embedding_train)
            target_labels_train.append(ytrain)

        for Xtest, ytest, ttest in task_loader_test:
            with torch.no_grad():
                _, act_test = target_model.forward_with_auxiliary(Xtest, ttest)
                target_embedding_test = act_test[config.embedding_layer]
                ypred = target_classifier.forward_single_task(target_embedding_test, i)
            ypred = torch.argmax(ypred, dim=1)
            acc_metric.update(ypred, ytest)
            target_embeddings_test.append(target_embedding_test)
            target_labels_test.append(ytest)

        target_embeddings_train = torch.cat(target_embeddings_train)
        target_embeddings_test = torch.cat(target_embeddings_test)
        target_labels_test = torch.cat(target_labels_test).numpy()

        log_info(f"accuracy with original model and classifier: {acc_metric.result()}")

        # iterate over phases. For each phase j, get the corresponding model and embeddings for task i and align them to phase i
        for j, _ in enumerate(benchmark.test_stream):
            log_info(f"generating representations from model after phase {j}")
            # get source_model
            model_weight_path = path.join(
                config.environment.logdir,
                config.project,
                config.run_name,
                f"model_phase_{j}.pt",
            )
            log_info(f"loading weights: {model_weight_path}")
            state_dict = torch.load(model_weight_path, map_location=device)
            source_model.load_state_dict(state_dict)

            # TODO currently implementing this. While we're here we can get continuous classifier performance for t < 0
            # generate embeddings
            # make sure accuracy metric is fresh and model is in evaluation mode
            acc_metric.reset()
            source_model.eval()

            # get source embeddings for training and test set
            source_embeddings_train = []
            source_embeddings_test = []
            misaligned_predictions = []
            # for ((Xtrain, ytrain, ttrain),(Xtest,ytest,ttest)) in zip(task_loader_train, task_loader_test):
            for Xtrain, ytrain, ttrain in task_loader_train:
                with torch.no_grad():
                    _, act_train = source_model.forward_with_auxiliary(Xtrain, ttrain)
                    source_embedding_train = act_train[config.embedding_layer]
                source_embeddings_train.append(source_embedding_train)

            for Xtest, ytest, ttest in task_loader_test:
                with torch.no_grad():
                    _, act_test = source_model.forward_with_auxiliary(Xtest, ttest)
                    source_embedding_test = act_test[config.embedding_layer]
                    ypred = target_classifier.forward_single_task(
                        source_embedding_test, i
                    )
                source_embeddings_test.append(source_embedding_test)
                ypred = torch.argmax(ypred, dim=1)
                acc_metric.update(ypred, ytest)

            source_embeddings_train = torch.cat(source_embeddings_train)
            source_embeddings_test = torch.cat(source_embeddings_test)
            acc = acc_metric.result()
            results_with_original_readout[i].append(acc)
            log_info(f"accuracy with model {j} and original classifier: {acc}")

            acc_metric.reset()

            # TODO in the following we never explicitly convert between cpu and 'device'. This might break when running on cuda (running this analysis on cpu only will work)
            # align with train set, test performance with aligned test data
            _, R, s, norm, translate = utils.procrustes_transform(
                target_embeddings_train.detach().numpy(),
                source_embeddings_train.detach().numpy(),
            )

            # procrustes on test data
            source_embeddings_test = source_embeddings_test.detach().numpy()
            translate_test = np.mean(source_embeddings_test, axis=0, keepdims=True)
            norm_test = np.linalg.norm(source_embeddings_test - translate_test)
            source_embeddings_test_norm = (
                source_embeddings_test - translate_test
            ) / norm_test
            embeddings_aligned = np.dot(source_embeddings_test_norm, R.T) * s

            # correct for different number of samples
            scale_correct = np.sqrt(source_embedding_test.shape[0]) / np.sqrt(
                source_embedding_train.shape[0]
            )
            embeddings_aligned *= scale_correct

            # scale back up and translate back
            embeddings_aligned = embeddings_aligned * norm + translate

            embeddings_aligned = torch.tensor(embeddings_aligned)
            pred_aligned = target_classifier.forward_single_task(
                torch.tensor(embeddings_aligned), i
            )
            ypred = torch.argmax(pred_aligned, dim=1)
            acc_metric.update(ypred, target_labels_test)
            acc = acc_metric.result()
            aligned_results[i].append(acc)
            log_info(f"aligned accuracy with model {j} and original classifier: {acc}")

            # repeat alignment on the test data
            acc_metric.reset()
            embeddings_aligned, R, s, norm, translate = utils.procrustes_transform(
                target_embeddings_test.detach().numpy(), source_embeddings_test
            )
            embeddings_aligned = torch.tensor(embeddings_aligned)
            pred_aligned = target_classifier.forward_single_task(
                torch.tensor(embeddings_aligned), i
            )
            ypred = torch.argmax(pred_aligned, dim=1)
            acc_metric.update(ypred, target_labels_test)
            acc = acc_metric.result()
            val_aligned_results[i].append(acc)
            log_info(
                f"validation aligned accuracy with model {j} and original classifier: {acc}"
            )

    # save results
    savedir = path.join(
        run_config.environment.logdir,
        run_config.experiment.project,
        f"{run_config.environment.timestamp}_{run_config.experiment.name}",
    )
    og_readout_results_file = path.join(savedir, f"original_readout_results.yaml")
    aligned_readout_results_file = path.join(savedir, f"aligned_readout_results.yaml")
    val_aligned_readout_results_file = path.join(
        savedir, f"validation_aligned_readout_results.yaml"
    )

    with open(og_readout_results_file, "w") as outfile:
        yaml.dump(results_with_original_readout, outfile, default_flow_style=False)

    with open(aligned_readout_results_file, "w") as outfile:
        yaml.dump(aligned_results, outfile, default_flow_style=False)

    with open(val_aligned_readout_results_file, "w") as outfile:
        yaml.dump(val_aligned_results, outfile, default_flow_style=False)


experiment()

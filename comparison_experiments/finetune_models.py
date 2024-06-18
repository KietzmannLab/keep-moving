import torch
import avalanche as avl
from avalanche.evaluation import metrics as metrics
import hydra
from time import time
from os import path
import random
import logging
from termcolor import colored
import yaml
from omegaconf import OmegaConf

import utils
import optimizers
from strategies.strategies import instantiate_strategy
from models.models import instantiate_model
from benchmarks.benchmarks import instantiate_benchmark
from benchmarks.dataset_utils import (
    format_class_order_from_benchmark,
    format_class_order_from_config,
)

logger = logging.getLogger("trainer")


def log_info(msg):
    logger.info(colored(msg, "blue"))


@hydra.main(
    version_base=None, config_path="configuration/finetune_conf", config_name="default"
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
    run_config = OmegaConf.load(conf_path)

    # init model with same config as during training
    model = instantiate_model(run_config.experiment.model)
    criterion = torch.nn.CrossEntropyLoss()
    interactive_logger = avl.logging.InteractiveLogger()
    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger],
    )

    # load data with same config as during training
    run_config.experiment.scenario.fixed_class_order = format_class_order_from_config(
        run_config.experiment.scenario
    )
    run_config.experiment.scenario.return_task_id = True  # we want to train separate classifiers, so the benchmark must provide task labels (irrespective of whether they were used in training)
    benchmark = instantiate_benchmark(run_config.experiment.scenario)
    n_experiences = run_config.experiment.scenario.n_experiences

    for i in range(n_experiences):
        log_info(f"Finetune model checkpoint after training on experience {i}")

        # load weights
        model_weight_path = path.join(
            run_config.environment.logdir,
            run_config.experiment.project,
            f"{run_config.environment.timestamp}_{run_config.experiment.name}",
            f"model_phase_{i}.pt",
        )
        model.load_state_dict(
            torch.load(model_weight_path, map_location=device), strict=False
        )
        model.reinitialize_classifier()  # reset readout layers
        model.set_trainable_backbone(
            False
        )  # freeze all trainable layers except for classifier
        log_info("Loaded weights, backbone is frozen.")

        optimizer = optimizers.instantiate_optimizer(
            config.optimizer, model.parameters()
        )
        early_stopping = avl.training.plugins.EarlyStoppingPlugin(
            config.algorithm.patience,
            val_stream_name="test_stream/Task000",
            margin=0.001,
            verbose=True,
        )
        strategy = instantiate_strategy(
            config,
            model,
            optimizer,
            device,
            criterion,
            evaluation_plugin,
            plugins=[early_stopping],
            eval_every=1,
        )  # evaluate every epoch for early stopping

        # finetune for all datasets in benchmark
        for j, (experience, eval_experience) in enumerate(
            zip(benchmark.train_stream, benchmark.test_stream)
        ):
            log_info(f"Stream number: {j}")
            log_info(f"Training for {strategy.train_epochs} epoch(s)")
            log_info(f"Classes in experience: {experience.classes_in_this_experience}")

            # make sure early stopping, if used tracks the metric for the current task
            early_stopping.val_stream_name = f"test_stream/Task{str(j).zfill(3)}"
            early_stopping.metric_key = f"{early_stopping.metric_name}/eval_phase/{early_stopping.val_stream_name}"
            early_stopping.before_training(strategy)

            strategy.train(
                experience,
                eval_streams=[benchmark.test_stream[j]],
                num_workers=config.environment.n_workers,
            )
        res = strategy.eval(benchmark.test_stream)  # evaluate all tasks
        log_info("Results:")
        log_info(res)

        savedir = path.join(
            run_config.environment.logdir,
            run_config.experiment.project,
            f"{run_config.environment.timestamp}_{run_config.experiment.name}",
        )
        modelsavepath = path.join(savedir, f"classifiers_phase_{i}.pt")
        resultssavepath = path.join(savedir, f"finetuning_results_phase_{i}.yaml")

        result_dict = dict()
        for t in range(n_experiences):
            tstr = str(t).zfill(3)
            key = f"Top1_Acc_Exp/eval_phase/test_stream/Task{tstr}/Exp{tstr}"
            r = res[key]
            result_dict[t] = r

        # save model and results
        torch.save(model.classifier.state_dict(), modelsavepath)
        with open(resultssavepath, "w") as outfile:
            yaml.dump(result_dict, outfile, default_flow_style=False)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
    experiment()

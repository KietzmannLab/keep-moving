import torch
import hydra
import avalanche as avl
from avalanche.evaluation import metrics as metrics
from avalanche.training.plugins.checkpoint import (
    CheckpointPlugin,
    FileSystemCheckpointStorage,
)
import wandb
import omegaconf
from time import time
import yaml
import os
from os import path
import random
import logging
from termcolor import colored
import pprint
from copy import deepcopy
from torchinfo import summary
import numpy as np
import sys

import utils
from models.models import instantiate_model
from strategies.strategies import instantiate_strategy
from benchmarks.benchmarks import instantiate_benchmark
from optimizers import instantiate_optimizer
from parameter_manager import instantiate_parameter_manager
from initializers import init_model
import plugins


# main experiment
@hydra.main(
    version_base=None, config_path="configuration/train_conf", config_name="default"
)
def experiment(config):
    logger = logging.getLogger("trainer")

    def log_info(msg):
        logger.info(colored(msg, "blue"))

    wandb.login()

    try:
        timestamp = config.environment.timestamp
        log_info(f"found existing timestamp {timestamp}")
    except:
        timestamp = str(time())
        log_info(f"generating timestamp for new run: {timestamp}")
        config.environment.timestamp = timestamp

    # set and track seed for reproducibility
    if type(config.environment.seed) is int:
        seed = config.environment.seed
        utils.set_seed(seed)
    else:
        seed = random.randint(0, 99999999)
        config.environment.seed = seed
        utils.set_seed(seed)
    # also set same seed for data
    config.experiment.scenario.seed = seed

    # set compute device
    device = torch.device(
        f"cuda:{config.environment.cuda}"
        if torch.cuda.is_available() and config.environment.cuda >= 0
        else "cpu"
    )
    log_info(f"running on {device}")

    # instantiate model and data
    model = instantiate_model(config.experiment.model)
    summary(model)

    # potentially create log folder
    stamped_name = f"{timestamp}_{config.experiment.name}"
    os.makedirs(
        path.join(config.environment.logdir, config.experiment.project, stamped_name),
        exist_ok=True,
    )

    # save a copy of the randomly initialized model
    modelsavepath = path.join(
        config.environment.logdir,
        config.experiment.project,
        stamped_name,
        f"model_phase_init.pt",
    )
    torch.save(model.state_dict(), modelsavepath)

    # initialize model, optimizer, criterion, create benchmark

    config.experiment.scenario.data_root = (
        config.environment.data_root
    )  # datasets need to know where data lives depending on machine
    init_model(config.experiment.initializer, model)
    optimizer = instantiate_optimizer(config.experiment.optimizer, model)
    benchmark = instantiate_benchmark(config.experiment.scenario)
    criterion = torch.nn.CrossEntropyLoss()
    try:
        config.experiment.scenario.fixed_class_order = [
            list(x) for x in benchmark.original_classes_in_exp
        ]
    except:
        # if benchmark does not have original_classes_in_exp, we have to find classes manually
        fixed_class_order = []
        for i in range(config.experiment.scenario.n_experiences):
            dset = benchmark.train_stream[i].dataset
            classes = [int(dset.targets[i]) for i in range(len(dset))]
            classes = np.unique(classes).tolist()
            fixed_class_order.append(classes)
        config.experiment.scenario.fixed_class_order = fixed_class_order

    try:
        runid = config.environment.wandb_id
        log_info(f"wandb id has already been set, continuing with id: {runid}")
        log_info(f"rules for resuming are set to: '{config.environment.resume_wandb}'")
    except:
        runid = wandb.util.generate_id()
        config.environment.wandb_id = runid
        log_info(f"generating new wandb id: {runid}")

    config_dict = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )
    log_info("running with config:")
    log_info(pprint.pformat(config_dict))
    # save config
    omegaconf.OmegaConf.save(
        config,
        path.join(
            config.environment.logdir,
            config.experiment.project,
            stamped_name,
            "config.yaml",
        ),
    )

    if not config.experiment.name == "default":
        log_info(f"named wandb run: {config.experiment.name}")
        wandb.init(
            project=config.experiment.project,
            name=config.experiment.name,
            notes=config.experiment.description,
            config=config_dict,
            id=runid,
            resume=config.environment.resume_wandb,
            reinit=True,
        )
    else:
        log_info(f"generating a random name for this run")
        wandb.init(
            project=config.experiment.project,
            notes=config.description,
            config=config_dict,
            id=runid,
            resume=config.environment.resume_wandb,
            reinit=True,
        )

    # checkpointing for resuming jobs
    if config.environment.checkpointing:
        ckpt_dir = path.join(
            config.environment.logdir,
            config.experiment.project,
            stamped_name,
            "checkpoints",
        )
        # make sure dir exists
        os.makedirs(ckpt_dir, exist_ok=True)
        checkpoint_plugin = CheckpointPlugin(
            FileSystemCheckpointStorage(
                directory=ckpt_dir,
            ),
            map_location=device,
        )
        strategy, initial_exp = checkpoint_plugin.load_checkpoint_if_exists()
        ckpt_cleaner = plugins.CheckpointCleaner(savedir=ckpt_dir)
        strategy_plugins = [checkpoint_plugin, ckpt_cleaner]
    else:
        strategy = None
        strategy_plugins = []
        initial_exp = 0

    # if we are not resuming, setup happens here
    if strategy is None:
        # set up logging to wandb
        interactive_logger = avl.logging.InteractiveLogger()
        wandb_logger = avl.logging.WandBLogger(project_name=config.experiment.project)
        evaluation_plugin = avl.training.plugins.EvaluationPlugin(
            metrics.accuracy_metrics(
                epoch=True, experience=True, stream=True, minibatch=True
            ),
            metrics.loss_metrics(epoch=True, minibatch=True),  # log loss as well
            loggers=[interactive_logger, wandb_logger],
        )

        log_info("Starting a new run, instantiating strategy")
        # set up plugins

        # 'manually' save weights and optimizer state after every epoch
        weight_dir = path.join(
            config.environment.logdir,
            config.experiment.project,
            stamped_name,
            "weights",
        )
        optim_dir = path.join(
            config.environment.logdir, config.experiment.project, stamped_name, "optim"
        )
        epoch_weight_saver_plugin = plugins.EpochWeightSaver(
            savedir=weight_dir, save_every=config.environment.save_every
        )
        strategy_plugins.append(epoch_weight_saver_plugin)
        epoch_optim_saver_plugin = plugins.EpochOptimStateSaver(
            savedir=optim_dir, save_every=config.environment.save_every
        )
        strategy_plugins.append(epoch_optim_saver_plugin)

        if config.experiment.optimizer.early_stopping:
            early_stopping = avl.training.plugins.EarlyStoppingPlugin(
                config.experiment.optimizer.patience,
                val_stream_name="test_stream/Task000",
                margin=config.experiment.optimizer.margin,
                verbose=True,
                metric_name="Top1_Acc_Exp",
            )
            strategy_plugins.append(early_stopping)

        # set up lr scheduler if used
        if config.experiment.use_manager:
            manager = instantiate_parameter_manager(
                config.experiment.param_manager, optimizer, model
            )
            strategy_plugins.append(manager)

        log_info(f"instantiating strategy with plugins: {strategy_plugins}")
        strategy = instantiate_strategy(
            config.experiment,
            model,
            optimizer,
            device,
            criterion,
            evaluation_plugin,
            eval_every=config.environment.eval_every,
            plugins=strategy_plugins,
        )

    results = []  # save performances over experiences
    n_experiences = len(benchmark.train_stream)

    for i in range(initial_exp, n_experiences, 1):
        experience = benchmark.train_stream[i]
        # make sure early stopping, if used tracks the metric for the current task
        if config.experiment.optimizer.early_stopping:
            early_stopping.val_stream_name = f"test_stream/Task{str(i).zfill(3)}"
            early_stopping.metric_key = f"{early_stopping.metric_name}/eval_phase/{early_stopping.val_stream_name}"
            early_stopping.before_training(strategy)

            log_info(f"Early Stopping is tracking: {early_stopping.metric_key}")
        # first experience may be pretraining
        if (config.experiment.scenario.has_pretraining == True) and (i == 0):
            strategy.train_epochs = config.experiment.scenario.pretrain_episodes
        else:
            strategy.train_epochs = config.experiment.scenario.episodes
        log_info(f"Stream number: {i}")
        log_info(f"Training for {strategy.train_epochs} epoch(s)")
        log_info(f"Classes in experience: {experience.classes_in_this_experience}")
        # set up multiprocessing for data loaders
        multiprocessing_options = dict()
        if config.environment.n_workers > 0:
            log_info(
                f"setting up training with {config.environment.n_workers} worker processes"
            )
            multiprocessing_options["num_workers"] = config.environment.n_workers
            multiprocessing_options[
                "prefetch_factor"
            ] = config.environment.prefetch_factor
            multiprocessing_options[
                "persistent_workers"
            ] = config.environment.persistent_workers
            log_info(f"running with: {multiprocessing_options}")

        # if we are performing an experiment with frozen layers, potentially freeze / unfreeze things here
        if config.experiment.freeze_experiment and (
            config.experiment.freeze_experience == i
        ):
            freeze_names = config.experiment.freeze_layers
            log_info(f"Freezing layers {freeze_names}")
            for fn in freeze_names:
                layer = model.blocks[fn]
                for parameter in layer.parameters():
                    parameter.requires_grad = False

        # train the model on the current experience
        strategy.train(
            experience,
            eval_streams=[benchmark.test_stream[i]],
            **multiprocessing_options,
        )
        log_info(f"finished training on experience {i}.")
        res = deepcopy(
            strategy.eval(benchmark.test_stream, **multiprocessing_options)
        )  # evaluate all tasks
        results.append(res)
        modelsavepath = path.join(
            config.environment.logdir,
            config.experiment.project,
            stamped_name,
            f"model_phase_{i}.pt",
        )
        torch.save(model.state_dict(), modelsavepath)
        if (
            config.experiment.exit_early_after >= 0
        ) and i >= config.experiment.exit_early_after:
            sys.exit(0)

    # collect results and log to yaml
    tasks = config.experiment.scenario.n_experiences
    result_dict = {t: [] for t in range(tasks)}
    for r in results:
        print(r)
        for t in range(tasks):
            estr = str(t).zfill(3)
            if config.experiment.scenario.return_task_id:
                tstr = str(t).zfill(3)
            else:
                tstr = "000"
            key = f"Top1_Acc_Exp/eval_phase/test_stream/Task{tstr}/Exp{estr}"
            res = r[key]
            result_dict[t].append(res)

    with open(
        path.join(
            config.environment.logdir,
            config.experiment.project,
            stamped_name,
            "results.yaml",
        ),
        "w",
    ) as outfile:
        yaml.dump(result_dict, outfile, default_flow_style=False)


# run the experiment
if __name__ == "__main__":  # for multiproc
    torch.multiprocessing.set_start_method(method="fork", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
    experiment()

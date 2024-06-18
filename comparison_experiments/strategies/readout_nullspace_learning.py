import torch
from avalanche.benchmarks.utils import concat_classification_datasets
from avalanche.benchmarks.utils.data_loader import GroupBalancedInfiniteDataLoader
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import (
    ExemplarsBuffer,
    ExperienceBalancedBuffer,
)
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins.evaluation import default_evaluator

from copy import deepcopy


class ReadoutNullspacePlugin(SupervisedPlugin):
    """
    Activity replay plugin.

    This Plugin holds an external memory filled with randomly selected
    patterns. Differently from ReplayPlugin, these patterns are used
    to constrain activity patterns of the model. This is done with an
    L2 regularization term that penalizes the distance between the
    activity patterns of data in the replay buffer at the time of training
    and during subsequent learning on additional experiences.
    """

    def __init__(
        self,
    ):
        super().__init__()
        self.readouts_trained = 0

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        self.readouts_trained += 1
        strategy.model.update_filter(readout_idx=list(range(self.readouts_trained)))


class ReadoutNullspaceLearning(SupervisedTemplate):
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins=None,
        evaluator=default_evaluator(),
        eval_every=-1,
        **base_kwargs
    ):
        """Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param mem_size: replay buffer size.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param \*\*base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """

        plugin = ReadoutNullspacePlugin()
        if plugins is None:
            plugins = [plugin]
        else:
            plugins.append(plugin)

        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )

from copy import deepcopy

import torch
from avalanche.training.plugins import EWCPlugin

from avalanche.benchmarks.utils.data_loader import GroupBalancedInfiniteDataLoader
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import (
    ExemplarsBuffer,
    ExperienceBalancedBuffer,
)
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins.evaluation import default_evaluator


from typing import Callable, Optional, List, Union
import torch

from torch import sigmoid
from torch.nn import Module
from torch.optim import Optimizer

from avalanche.training.plugins.evaluation import (
    default_evaluator,
)

from avalanche.training.plugins import (
    SupervisedPlugin,
    EWCPlugin,
    EvaluationPlugin,
)
from avalanche.training.templates import SupervisedTemplate


class NullspaceMovementPlugin(SupervisedPlugin):
    def __init__(
        self,
        layer,  # layer for which we want to regularise. Should be last layer pre readout
        nullspace_lambda=1.0,
        rcond=1e-8,
        num_workers=0,
        mem_size: int = 200,
        batch_size_mem: int = None,
        storage_policy=None,
    ):
        super().__init__()
        self.nullspace_lambda = nullspace_lambda
        self.rcond = rcond
        self.num_workers = num_workers
        self.mem_size = mem_size
        self.batch_size_mem = batch_size_mem
        self.storage_policy = storage_policy
        self.dataloader = None
        self.layer = layer
        self.model = None

        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            assert storage_policy.max_size == self.mem_size
        else:  # Default
            self.storage_policy = ExperienceBalancedBuffer(
                max_size=self.mem_size, adaptive_size=True
            )

    @property
    def ext_mem(self):
        return self.storage_policy.buffer_groups

    def before_training_exp(
        self,
        strategy,
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        **kwargs
    ):
        """
        Dataloader to build batches containing examples from both memories and
        the training dataset
        """
        if len(self.storage_policy.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return

        batch_size_mem = self.batch_size_mem
        if batch_size_mem is None:
            batch_size_mem = strategy.train_mb_size

        self.set_dataloader_from_buffer(num_workers, batch_size_mem)

    def set_dataloader_from_buffer(self, num_workers, batch_size_mem):
        self.dataloader = GroupBalancedInfiniteDataLoader(
            self.storage_policy.buffer_datasets,
            batch_size=batch_size_mem,
            num_workers=num_workers,
        ).__iter__()

    def _compute_activation_anchors(self, strategy, x, t, filter, model):
        """
        Compute activation anchors for all old tasks in replay batch
        Separate samples by task and pass each through the corresponding
        model in self.models

        anchors should never have gradients.
        """
        x, t = x.to(strategy.device), t.to(strategy.device)
        _, act = model.forward_with_auxiliary(x, t)
        anchor = act[self.layer]  # get prereadout layer
        filter = filter.to(anchor.device)
        task_filter = filter @ filter.T
        filter_anchor = anchor - (anchor @ task_filter)  # only regularize nullspace
        return filter_anchor

    def before_backward(self, strategy, *args, **kwargs):
        """
        Add the replay activity loss to the strategy loss
        """
        # get the next batch from the dataloader if it exists, if there is no dataloader
        # we are in the first experience, and we don't need to do anything
        if self.dataloader is None:
            return
        if self.model is None:
            return

        x, y, t = next(self.dataloader)

        # make sure models have forward_with_auxiliary functions
        assert hasattr(
            strategy.model, "forward_with_auxiliary"
        ), "missing forward_with_auxiliary function in model"

        # get anchors with target models, these are fixed so we don't need gradients
        filter = self.model.filter
        filter.requires_grad = False

        with torch.no_grad():
            target_embeddings = self._compute_activation_anchors(
                strategy, x, t, filter=filter, model=self.model
            )
            target_embeddings = target_embeddings.detach()

        # get anchors with current model, these need gradients
        embeddings = self._compute_activation_anchors(
            strategy,
            x,
            t,
            model=strategy.model,
            filter=filter,
        )
        # mse
        loss = 1 / (
            1
            + torch.sqrt(
                torch.functional.F.mse_loss(embeddings, target_embeddings) + 1e-8
            )
        )
        loss *= self.nullspace_lambda
        strategy.loss += loss

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        if self.model is None:
            print("FINISHED TRAINING FIRST TASK. SETTING UP ANCHOR MODEL.")
            self.storage_policy.update(strategy, **kwargs)
            strategy.model.update_filter(readout_idx=[0], rcond=self.rcond)
            self.model = deepcopy(strategy.model)

        del (
            self.dataloader
        )  # make sure to clean up the iterator, otherwise pickle will fail during checkpointing
        self.dataloader = None


class EWCNullspaceMovement(SupervisedTemplate):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion,
        ewc_lambda: float,
        layer,
        nullspace_lambda=1.0,
        rcond=1e-8,
        num_workers=0,
        mem_size=200,
        batch_size_mem=None,
        storage_policy=None,
        mode: str = "separate",
        decay_factor: Optional[float] = None,
        keep_importance_data: bool = False,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        **base_kwargs
    ):
        """Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param ewc_lambda: hyperparameter to weigh the penalty inside the total
               loss. The larger the lambda, the larger the regularization.
        :param mode: `separate` to keep a separate penalty for each previous
               experience. `onlinesum` to keep a single penalty summed over all
               previous tasks. `onlineweightedsum` to keep a single penalty
               summed with a decay factor over all previous tasks.
        :param decay_factor: used only if mode is `onlineweightedsum`.
               It specify the decay term of the importance matrix.
        :param keep_importance_data: if True, keep in memory both parameter
                values and importances for all previous task, for all modes.
                If False, keep only last parameter values and importances.
                If mode is `separate`, the value of `keep_importance_data` is
                set to be True.
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
        :param base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """
        ewc = EWCPlugin(ewc_lambda, mode, decay_factor, keep_importance_data)
        nullspace_movement = NullspaceMovementPlugin(
            layer=layer,
            nullspace_lambda=nullspace_lambda,
            rcond=rcond,
            num_workers=num_workers,
            mem_size=mem_size,
            batch_size_mem=batch_size_mem,
            storage_policy=storage_policy,
        )

        if plugins is None:
            plugins = [ewc, nullspace_movement]
        else:
            plugins.append(ewc)
            plugins.append(nullspace_movement)

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

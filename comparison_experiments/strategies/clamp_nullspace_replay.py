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


class ClampNullspaceReplayPlugin(SupervisedPlugin):
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
        layer,  # layer for which we want to regularise. Should be last layer pre readout
        nullspace_lambda=1.0,
        scaled=True,
        control=False,
        rcond=1e-6,
        num_workers=0,
        mem_size: int = 200,
        batch_size_mem: int = None,
        task_balanced_dataloader: bool = False,
        storage_policy=None,
    ):
        super().__init__()
        self.nullspace_lambda = nullspace_lambda
        self.rcond = rcond
        self.num_workers = num_workers
        self.mem_size = mem_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader
        self.storage_policy = storage_policy
        self.dataloader = None
        self.models = []
        self.filters = []
        self.layer = layer
        self.scaled = scaled
        self.control = control

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

    def _compute_activation_anchors(self, strategy, x, t, filters, models):
        """
        Compute activation anchors for all old tasks in replay batch
        Separate samples by task and pass each through the corresponding
        model in self.models

        anchors should never have gradients.
        """
        # assert that task vector is sorted
        t_resort, _ = t.sort()
        assert torch.all(t_resort == t), "Task vector and samples must be sorted"

        unique_tasks = torch.unique(t)
        task_batches = []
        for ut in unique_tasks:
            task_mask = t == ut
            task_x = x[task_mask]
            task_batches.append(task_x)

        activation_anchors = []
        for task, task_x, task_model, task_filter in zip(
            unique_tasks, task_batches, models, filters
        ):
            # create task vector like input
            t_vec = torch.ones((len(task_x),)) * task
            t_vec = t_vec.long().to(strategy.device)
            task_x = task_x.to(strategy.device)
            _, act = task_model.forward_with_auxiliary(task_x, t_vec)
            anchor = act[self.layer]  # get prereadout layer
            # print shapes
            task_filter = task_filter @ task_filter.T

            filter_anchor = anchor - (anchor @ task_filter)  # only regularize nullspace
            # filter_anchor = (anchor @ task_filter)  # only regularize nullspace
            # filter_anchor = anchor - anchor

            activation_anchors.append(filter_anchor)
        activation_anchors = torch.cat(activation_anchors, dim=0)
        return activation_anchors

    def before_backward(self, strategy, *args, **kwargs):
        """
        Add the replay activity loss to the strategy loss
        """
        # get the next batch from the dataloader if it exists, if there is no dataloader
        # we are in the first experience, and we don't need to do anything
        if self.dataloader is None:
            return
        if len(self.models) == 0:
            return

        x, y, t = next(self.dataloader)

        # sort by task to ensure correct ordering
        t, idx = t.sort()
        x = x[idx]
        y = y[idx]

        # make sure models have forward_with_auxiliary functions
        assert hasattr(
            strategy.model, "forward_with_auxiliary"
        ), "missing forward_with_auxiliary function in model"

        # get anchors with target models, these are fixed so we don't need gradients
        with torch.no_grad():
            target_embeddings = self._compute_activation_anchors(
                strategy, x, t, filters=self.filters, models=self.models
            )
            target_embeddings = target_embeddings.detach()

        # get anchors with current model, these need gradients
        embeddings = self._compute_activation_anchors(
            strategy,
            x,
            t,
            models=[strategy.model] * len(self.models),
            filters=self.filters,
        )
        # mse
        loss = torch.functional.F.mse_loss(embeddings, target_embeddings)
        loss *= self.nullspace_lambda
        strategy.loss += loss

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        self.storage_policy.update(strategy, **kwargs)
        self.models.append(
            deepcopy(strategy.model)
        )  # update the list of models we use for generating activation patterns of old tasks
        self.filters.append(
            deepcopy(
                strategy.model.update_filter(
                    readout_idx=[len(self.models) - 1],
                    rcond=self.rcond,
                    device=strategy.device,
                    control=self.control,
                    scaled=self.scaled,
                )
            )
        )  # we just added a model, so the index is the length - 1
        del (
            self.dataloader
        )  # make sure to clean up the iterator, otherwise pickle will fail during checkpointing
        self.dataloader = None


class ClampNullspaceReplay(SupervisedTemplate):
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        layer,
        nullspace_lambda=1.0,
        scaled=False,
        control=False,
        nullspace_rcond=1e-15,
        replay_batch_size=None,
        mem_size: int = 200,
        replay_num_workers=0,
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

        rp = ClampNullspaceReplayPlugin(
            mem_size=mem_size,
            nullspace_lambda=nullspace_lambda,
            scaled=scaled,
            control=control,
            rcond=nullspace_rcond,
            layer=layer,
            batch_size_mem=replay_batch_size,
            num_workers=replay_num_workers,
        )
        if plugins is None:
            plugins = [rp]
        else:
            plugins.append(rp)

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

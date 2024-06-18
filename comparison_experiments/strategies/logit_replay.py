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


class LogitReplayPlugin(SupervisedPlugin):
    def __init__(
        self,
        model,
        replay_lambda=1.0,
        num_workers=0,
        mem_size: int = 200,
        batch_size_mem: int = None,
        task_balanced_dataloader: bool = False,
        storage_policy=None,
        auxiliary_loss="mse_logits",
    ):
        super().__init__()
        self.replay_lambda = replay_lambda
        self.num_workers = num_workers
        self.mem_size = mem_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader
        self.storage_policy = storage_policy
        self.dataloader = None
        self.models = []
        self.auxiliary_loss = auxiliary_loss

        if storage_policy is not None:
            self.storage_policy = storage_policy
            assert storage_policy.max_size == self.mem_size
        else:
            self.storage_policy = ExperienceBalancedBuffer(
                max_size=self.mem_size, adaptive_size=True
            )

    @property
    def ext_mem(self):
        return self.storage_policy.buffer_groups

    def set_dataloader_from_buffer(self, num_workers, batch_size_mem):
        # this sets the dataloader field to be used for this experience.
        # we need to clean it up at the end of the experience, because
        # iterators can't be pickled
        self.dataloader = GroupBalancedInfiniteDataLoader(
            self.storage_policy.buffer_datasets,
            batch_size=batch_size_mem,
            num_workers=num_workers,
        ).__iter__()

    def before_training_exp(self, strategy, num_workers=0, **kwargs):
        if len(self.storage_policy.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return

        batch_size_mem = self.batch_size_mem
        if batch_size_mem is None:
            batch_size_mem = strategy.train_mb_size

        self.set_dataloader_from_buffer(num_workers, batch_size_mem)

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        # add new samples to memory
        self.storage_policy.update(strategy, **kwargs)

        # add the model to the list of models we use for generating logits of old tasks
        self.models.append(
            deepcopy(strategy.model)
        )  # update the list of models we use for generating activation patterns of old tasks

        # clean up the dataloader
        del (
            self.dataloader
        )  # make sure to clean up the iterator, otherwise pickle will fail during checkpointing
        self.dataloader = None

    @torch.no_grad()  # we compute targets so we never want gradients here
    def _compute_target_logits(self, x, t):
        # compute target logits for all old tasks in replay batch
        # separate samples by task and pass each through the corresponding
        # model in self.models
        # make sure to use the model in eval mode
        target_logits = []
        for task_id in torch.unique(t):
            task_mask = t == task_id
            x_task = x[task_mask]
            model = self.models[task_id]
            model.eval()
            t_vec = (
                torch.ones((len(x_task),)).to(task_id.device) * task_id
            )  # make sure to use the correct device
            t_vec = t_vec.long().to(x_task.device)
            logits = model(x_task, t_vec)
            target_logits.append(logits)
        target_logits = torch.cat(target_logits, dim=0)
        return target_logits

    def before_backward(self, strategy, *args, **kwargs):
        # make sure we dont crash if dataloader has not yet been set
        if self.dataloader is None:
            return
        x, y, t = next(self.dataloader)  # new batch from memory

        # sort by task to ensure correct ordering
        t, indices = torch.sort(t)
        x = x[indices]
        y = y[indices]

        x = x.to(strategy.device)
        y = y.to(strategy.device)
        t = t.to(strategy.device)

        logits = strategy.model(x, t)

        # compute target logits
        target_logits = self._compute_target_logits(x, t)

        if self.auxiliary_loss == "crossentropy_softmax":
            target_logits = torch.nn.functional.softmax(target_logits, dim=1)
            loss = torch.nn.functional.cross_entropy(logits, target_logits)
        elif self.auxiliary_loss == "mse_logits":
            loss = torch.nn.functional.mse_loss(logits, target_logits)
        elif self.auxiliary_loss == "full_replay":
            loss = torch.nn.functional.cross_entropy(logits, y)
        else:
            raise ValueError(f"Unknown auxiliary loss {self.auxiliary_loss}")
        loss *= self.replay_lambda

        strategy.loss += loss


class LogitReplay(SupervisedTemplate):
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        replay_batch_size=None,
        mem_size: int = 200,
        auxiliary_loss="mse_logits",
        replay_lambda: float = 1.0,
        replay_num_workers=0,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins=None,
        evaluator=default_evaluator(),
        eval_every=-1,
        **base_kwargs,
    ):
        rp = LogitReplayPlugin(
            model=model,
            mem_size=mem_size,
            replay_lambda=replay_lambda,
            batch_size_mem=replay_batch_size,
            num_workers=replay_num_workers,
            auxiliary_loss=auxiliary_loss,
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
            **base_kwargs,
        )

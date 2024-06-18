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


class NullspaceRegularisationPlugin(SupervisedPlugin):
    def __init__(
        self,
        layer,
        no_bias=True,
        nullspace_lambda=1e9,  # lambda for nullspace regularization weight
        rcond=1e-3,  # rcond value for scipy orth. Directions with singular values < rcond are discarded
        control=False,
        scaled=False,
        criterion="mse",
    ):
        super().__init__()
        self.nullspace_lambda = nullspace_lambda
        self.model = None
        self.rcond = rcond
        self.layer = layer
        self.old_readouts = 0
        self.no_bias = no_bias
        self.scaled = scaled
        self.control = control
        if criterion == "mse":
            self.criterion = torch.nn.functional.mse_loss
        elif criterion == "cosine":

            def cosine_criterion(x, y):
                return 1 - torch.nn.functional.cosine_similarity(x, y)

            self.criterion = cosine_criterion
        else:
            raise ValueError(f"Unknown criterion {criterion}")

    def before_training_exp(self, strategy, **kwargs):
        # this method may not work properly if we learn a bias for the readout.
        # Optionally, we can set the biases to zero and freeze them.
        if self.no_bias:
            biases = [
                strategy.model.classifier.classifiers[str(i)].classifier.bias
                for i in range(len(strategy.model.classifier.classifiers))
            ]
            for i, p in enumerate(biases):
                print("Disabling bias for readout", i)
                p.data.zero_()
                p.requires_grad = False

    def before_backward(self, strategy, **kwargs):
        if self.model is None:  # first task, nothing to see here
            return
        # get current batch from strategy
        x, t = strategy.mb_x, strategy.mb_task_id
        # move to device
        x, t = x.to(strategy.device), t.to(strategy.device)
        # get targets
        with torch.no_grad():
            _, aux = self.model.forward_with_auxiliary(
                x, 0
            )  # 0 is a dummy task id that should always exist
            targets = aux[self.layer]

        # get embeddings with current model
        _, aux = strategy.model.forward_with_auxiliary(x, t)
        embeddings = aux[self.layer]

        # filter with model filter
        embeddings = embeddings @ strategy.model.filter
        targets = targets @ strategy.model.filter
        loss = torch.nn.functional.mse_loss(
            embeddings, targets
        )  # TODO: maybe cosine loss?
        loss *= self.nullspace_lambda
        strategy.loss += loss

    def after_training_exp(self, strategy, *args, **kwargs):
        self.old_readouts += 1
        readout_idx = list(range(self.old_readouts))
        self.model = deepcopy(strategy.model)
        self.model.eval()
        self.model.to(strategy.device)
        strategy.model.update_filter(
            readout_idx,
            self.rcond,
            device=strategy.device,
            scaled=self.scaled,
            control=self.control,
        )


class NullspaceRegularisationLearning(SupervisedTemplate):
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        layer,
        nullspace_lambda=1e9,  # lambda for nullspace regularization weight
        scaled=False,
        control=False,
        no_bias=True,
        reg_criterion="mse",
        rcond=1e-3,  # rcond value for scipy orth. Directions with singular values < rcond are discarded
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins=None,
        evaluator=default_evaluator(),
        eval_every=-1,
        **base_kwargs,
    ):
        nrp = NullspaceRegularisationPlugin(
            layer=layer,
            nullspace_lambda=nullspace_lambda,
            rcond=rcond,
            scaled=scaled,
            control=control,
            no_bias=no_bias,
            criterion=reg_criterion,
        )
        if plugins is None:
            plugins = [nrp]
        else:
            plugins.append(nrp)

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

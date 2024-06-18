import numpy as np
from scipy.linalg import null_space
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


class NullspaceGradientsPlugin(SupervisedPlugin):
    def __init__(
        self,
        model,
    ):
        super().__init__()
        self.filter = None
        self.last_exp_idx = 0

        # assert that model has a function called 'get projection layer'
        assert hasattr(
            model, "get_projection_layer"
        ), "model must have a function called 'get_projection_layer'"
        # assert model has a field called classifier
        assert hasattr(
            model, "classifier"
        ), "model must have a field called 'classifier'"

    def after_backward(self, strategy, **kwargs):
        if self.filter is None:
            return

        # get grads for grad_project_layer
        grad_project_layer = strategy.model.get_projection_layer()
        grads = grad_project_layer.weight.grad
        # print shapes

        # project grads onto nullspace
        grads = np.matmul(self.filter, grads)
        # set grads for grad_project_layer
        grad_project_layer.weight.grad = grads

    def after_training_exp(self, strategy, **kwargs):
        # get classifier weights we just trained
        key = f"classifier.classifiers.{self.last_exp_idx}.classifier.weight"
        state = strategy.model.state_dict()
        # assert key exists
        assert key in state, f"key {key} not in state_dict"
        weights = state[key]
        # get nullspace of classifier weights
        nullspace = null_space(weights)
        nullspace_inv = np.linalg.pinv(nullspace)
        exp_filter = np.matmul(nullspace, nullspace_inv)
        # update filter
        if self.filter is None:
            self.filter = exp_filter
        else:
            self.filter = np.matmul(self.filter, exp_filter)
        # update last_exp_idx
        self.last_exp_idx += 1


class NullspaceGradients(SupervisedTemplate):
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_mb_size=1,
        train_epochs=1,
        eval_mb_size=1,
        device=None,
        plugins=None,
        evaluator=default_evaluator(),
        eval_every=-1,
        **base_kwargs,
    ):
        nullspace_grad_plugin = NullspaceGradientsPlugin(model)

        if plugins is None:
            plugins = [nullspace_grad_plugin]
        else:
            plugins.append(nullspace_grad_plugin)

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

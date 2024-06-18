from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from typing import Optional, List
from torch.nn import Module, Parameter
from torch.optim import Optimizer
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge


class LinearDriftEstimation(SupervisedTemplate):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion,
        embedding_layer,
        alpha,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator(),
        eval_every=-1,
        **base_kwargs,
    ):
        drift_estimator = LinearDriftEstimationPlugin(embedding_layer, alpha)
        if plugins is None:
            plugins = [drift_estimator]
        else:
            plugins.append(drift_estimator)

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


class LinearDriftEstimationPlugin(SupervisedPlugin):
    def __init__(self, embedding_layer, alpha):
        super().__init__()
        self.dataloader = None
        self.feature_holder = None  # register a variable that holds embeddings computed during forward pass
        self.embeddings_pre_epoch = None
        self.embeddings_post_epoch = None
        self.alpha = alpha
        self.tasks_seen_in_last_exp = []

        def hook(model, input, output):
            self.feature_holder = output

        print(f"registering output hook for layer {embedding_layer}")
        embedding_layer.register_forward_hook(
            hook
        )  # register hook. This will save the embedding computed during the forward pass
        # to the corresponding field in this plugin

    def before_training_exp(self, strategy, *args, **kwargs):
        """
        create a dataloader that does not shuffle the data
        """
        data = strategy.dataloader.data
        self.dataloader = DataLoader(batch_size=256, dataset=data, shuffle=False)

        print("saving embeddings prior to training")
        embeddings = []
        for X, y, t in self.dataloader:
            with torch.no_grad():
                strategy.model.forward(X, t)
                embeddings.append(self.feature_holder.clone())
        self.embeddings_pre_epoch = torch.cat(embeddings)

    def before_forward(self, strategy, *args, **kwargs):
        self.tasks_seen_in_last_exp.append(torch.unique(strategy.mb_task_id))

    def after_training_exp(self, strategy, *args, **kwargs):
        # get all embeddings
        print("saving embeddings post training")
        embeddings = []
        for X, y, t in self.dataloader:
            with torch.no_grad():
                strategy.model.forward(X, t)
                embeddings.append(self.feature_holder.clone())
        self.embeddings_post_epoch = torch.cat(embeddings)
        # do drift estimation here
        # TODO this should most definitely be done in torch directly, not in numpy!
        # TODO if we want to run this on GPU we will likely need to explicitly convert to cpu for sklearn RIDGE
        ridge_model = Ridge(alpha=self.alpha, fit_intercept=False, solver="svd")
        ridge_model.fit(
            self.embeddings_post_epoch.detach().numpy(),
            self.embeddings_pre_epoch.detach().numpy(),
        )
        ridge_weight = ridge_model.coef_
        ridge_weight = torch.tensor(ridge_weight)

        # update all known, inactive classifiers
        known_tasks = strategy.model.classifier.known_train_tasks_labels
        active_tasks = torch.unique(torch.cat(self.tasks_seen_in_last_exp))

        for task in known_tasks:
            if not task in active_tasks:
                print(f"applying correction to classifier: {task}")
                classifier_weight = strategy.model.classifier.classifiers[
                    str(task)
                ].classifier.weight
                classifier_weight_updated = torch.matmul(
                    classifier_weight, ridge_weight
                )
                strategy.model.classifier.classifiers[
                    str(task)
                ].classifier.weight = Parameter(classifier_weight_updated)

        print("resetting embedding variable for next experience")
        self.embeddings_pre_epoch = None
        self.embeddings_post_epoch = None
        self.dataloader = None
        self.tasks_seen_in_last_exp = []

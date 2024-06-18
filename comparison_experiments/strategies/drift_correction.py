from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from typing import Optional, List
from torch.nn import Module, MSELoss
from torch.optim import Optimizer
import torch


class DriftCorrectionPlugin(SupervisedPlugin):
    def __init__(
        self,
        embedding_layer,
        multihead_classifier,
        drift_lambda_logits,
        drift_lambda_means,
        drift_lambda_cov,
    ):
        super().__init__()
        self.feature_holder = None  # register a variable that holds embeddings computed during forward pass
        self.old_outputs = None  # register a variable that holds the outputs at inactive readouts prior to learning
        self.classifier = multihead_classifier
        self.drift_lambda_logits = drift_lambda_logits
        self.drift_lambda_means = drift_lambda_means
        self.drift_lambda_cov = drift_lambda_cov
        self.auxiliary_loss = MSELoss()

        def hook(model, input, output):
            self.feature_holder = output

        print(f"registering output hook for layer {embedding_layer}")
        embedding_layer.register_forward_hook(
            hook
        )  # register hook. This will save the embedding computed during the forward pass to the corresponding field in this plugin

    def after_forward(self, strategy, *args, **kwargs):
        """
        compute additional forward pass through inactive classifiers
        """
        embeddings = (
            self.feature_holder.detach()
        )  # make sure gradients are not propagated into earlier layers
        with torch.no_grad():
            self.old_outputs = self.classifier.forward_all_tasks(embeddings)

    def after_update(self, strategy, *args, **kwargs):
        # make sure all gradients are reset before drift correction
        strategy.optimizer.zero_grad()
        active_task = torch.unique(strategy.mb_task_id)

        # we need a new forward pass to get the new embeddings
        batch_inputs = strategy.mb_x
        batch_tasks = strategy.mb_task_id
        with torch.no_grad():
            strategy.model.forward(batch_inputs, batch_tasks)
        embeddings = self.feature_holder
        embeddings = (
            embeddings.detach()
        )  # make sure gradients are not propagated into earlier layers
        old_outputs = self.old_outputs
        new_outputs = self.classifier.forward_all_tasks(embeddings)
        known_tasks = self.classifier.known_train_tasks_labels

        # compute mse for all inactive known tasks
        cum_loss_logits = torch.tensor(0, dtype=torch.float, requires_grad=True)
        cum_loss_means = torch.tensor(0, dtype=torch.float, requires_grad=True)
        cum_loss_cov = torch.tensor(0, dtype=torch.float, requires_grad=True)

        for task in known_tasks:
            if not task in active_task:
                # logits
                task_output_old = old_outputs[
                    task
                ].detach()  # do not propagate gradients through graph with old embeddings
                task_output_new = new_outputs[task]
                loss = self.auxiliary_loss(task_output_new, task_output_old)
                cum_loss_logits = cum_loss_logits + loss

                # means
                with torch.no_grad():
                    old_means = torch.mean(task_output_old, axis=0)
                new_means = torch.mean(task_output_new, axis=0)
                loss = self.auxiliary_loss(new_means, old_means)
                cum_loss_means = cum_loss_means + loss

                # covariance matrices
                with torch.no_grad():
                    old_cov = torch.cov(task_output_old)
                new_cov = torch.cov(task_output_new)
                loss = self.auxiliary_loss(new_cov, old_cov)
                cum_loss_cov = cum_loss_cov + loss

        # weight and propagate gradients for auxiliary drift correction loss
        cum_loss_logits = cum_loss_logits * self.drift_lambda_logits
        cum_loss_means = cum_loss_means * self.drift_lambda_means
        cum_loss_cov = cum_loss_cov * self.drift_lambda_cov
        cum_loss = cum_loss_logits + cum_loss_means + cum_loss_cov
        cum_loss.backward()
        # update old readouts
        strategy.optimizer.step()
        strategy.optimizer.zero_grad()  # none of the drift correction gradients should leak into regular optimization loop


class DriftCorrection(SupervisedTemplate):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion,
        multihead_classifier,
        embedding_layer,
        drift_lambda_logits,
        drift_lambda_means,
        drift_lambda_cov,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator(),
        eval_every=-1,
        **base_kwargs,
    ):
        drift_correct = DriftCorrectionPlugin(
            embedding_layer,
            multihead_classifier,
            drift_lambda_logits,
            drift_lambda_means,
            drift_lambda_cov,
        )
        if plugins is None:
            plugins = [drift_correct]
        else:
            plugins.append(drift_correct)

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

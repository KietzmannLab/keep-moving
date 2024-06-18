import avalanche
from avalanche.training.plugins import SupervisedPlugin


class ParameterManagerPlugin(SupervisedPlugin):
    """
    This plugin manages which parameter groups are trainable over the course of each experience.
    """

    def __init__(
        self,
        optimizer,
        parameter_groups,
        unfreeze_before,
        verbose=True,
        ignore_first=True,
        freeze_previous_readouts=False,
    ):
        super().__init__()
        self.optimizer = optimizer
        self.parameter_groups = [
            list(pg["params"]) for pg in parameter_groups
        ]  # make sure we do not save generators here as this will cause pickle errors on checkpointing
        self.unfreeze_before = unfreeze_before
        self.verbose = verbose
        self.counter = 0
        self.ignore_first = ignore_first
        self.freeze_previous_readouts = freeze_previous_readouts
        self.exp_counter = 0
        assert len(self.parameter_groups) == len(
            self.unfreeze_before
        ), "Number of parameter groups must match number of unfreeze epochs"

    def freeze_previous_readout_parameters(self, i, model):
        """
        Freeze all parameters in previous readouts.
        """
        for readout in range(i):
            if self.verbose:
                print(f"Freezing readout {readout}")
            model.classifier.classifiers[
                str(readout)
            ].classifier.weight.requires_grad = False
            model.classifier.classifiers[
                str(readout)
            ].classifier.bias.requires_grad = False

    # before experience
    def before_training_exp(self, strategy, **kwargs):
        self.counter = 0

    def before_training_epoch(self, strategy, **kwargs):
        for i, (params, unfreeze_epoch) in enumerate(
            zip(self.parameter_groups, self.unfreeze_before)
        ):
            if (self.counter >= unfreeze_epoch) or (
                self.ignore_first and self.exp_counter == 0
            ):  # ignore first task if desired
                if self.verbose:
                    print(f"Unfreezing parameter group {i} at epoch {self.counter}")
                for parameter in params:
                    parameter.requires_grad = True
            else:
                if self.verbose:
                    print(f"Freezing parameter group {i} at epoch {self.counter}")
                for parameter in params:
                    parameter.requires_grad = False

            if self.freeze_previous_readouts:
                if self.verbose:
                    print("Running with frozen old readouts. Freezing now:")
                self.freeze_previous_readout_parameters(
                    self.exp_counter, strategy.model
                )
        self.counter += 1

    def after_training_exp(self, strategy, *args, **kwargs):
        self.exp_counter += 1


def instantiate_parameter_manager(cfg, optimizer, model):
    name = cfg.name
    if hasattr(model, "get_parameter_groups"):
        params = model.get_parameter_groups()
    else:
        params = model.parameters()
    if name == "step_up":
        manager = ParameterManagerPlugin(
            optimizer=optimizer,
            parameter_groups=params,
            unfreeze_before=cfg.before_epoch,
            freeze_previous_readouts=cfg.freeze_previous_readouts,
        )
    else:
        raise NotImplementedError(f"Unknown parameter manager: {name}")
    return manager


"""
        if config.experiment.freeze_previous_readouts and i > 0:
            log_info("Running with old readouts frozen")
            # freeze all parameters in old readouts
            for readout in range(i):
                log_info(f"Freezing readout {readout}")
                model.classifier.classifiers[readout].classifier.weight.requires_grad = False
                model.classifier.classifiers[readout].classifier.bias.requires_grad = False


"""

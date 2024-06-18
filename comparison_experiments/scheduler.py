import torch


def instantiate_lr_scheduler(cfg, optimizer):
    kind = cfg.name
    if kind == "stepup_backbone":
        """
        unfreeze backbone after cfg.after_epochs
        """
        # assert that optimizer has two parameter groups
        assert (
            len(optimizer.param_groups) == 2
        ), "optimizer must have two parameter groups for stepup_backbone scheduler"

        def step_up(epoch):
            if epoch < cfg.after_epochs:
                return 0.0
            else:
                return 1.0

        def const(epoch):
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [step_up, const])

    else:
        raise NotImplementedError(f"Unknown scheduler: {kind}")

    return scheduler

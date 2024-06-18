import torch.optim as optim


def instantiate_optimizer(config, model):
    name = config.name
    params = model.parameters()
    if name == "sgd":
        optimizer = optim.SGD(
            params=params,
            lr=config.lr,
            momentum=config.momentum,
            nesterov=config.nesterov,
        )
    elif name == "adam":
        optimizer = optim.Adam(
            params=params,
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )
    return optimizer

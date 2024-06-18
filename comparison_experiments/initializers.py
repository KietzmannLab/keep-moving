import torch
import torch.nn as nn


# Xavier intialisation
def weights_init_xavier(m, gain=torch.tensor(2)):
    """
    glorot / xavier init

    recommended gain for ReLU is the default
    """
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        # check m has bias
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        # check m has bias
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def weights_init_kaiming(m, nonlinearity="relu"):
    """
    he / kaiming init

    recommended gain for ReLU is the default
    """
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def weights_init(config, m):
    name = config.name
    if name == "kaiming":
        weights_init_kaiming(m)
    elif name == "xavier":
        weights_init_xavier(m)
    else:
        raise NotImplementedError(f"unknown initializer {name}")


def init_model(config, model):
    for m in model.get_layers_with_init():
        weights_init(config, m)

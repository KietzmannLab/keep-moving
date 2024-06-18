from models.variable_vgg import VariableDepthVGG
from models.mlp import MultiHeadMLP
from models.linear_mlp import MultiHeadMLP as LinearMLP
from models.slimresnet import MTSlimResNet18
from models.simplecnn import SimpleCNN


def instantiate_model(config):
    modeltype = config.name
    if modeltype == "mnist_mlp":
        model = MultiHeadMLP(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            drop_rate=config.drop_rate,
        )
    elif modeltype == "variable_vgg":
        model = VariableDepthVGG(
            block_sizes=config.block_sizes,
            dense_size=config.dense_size,
            dropouts=config.dropouts,
            input_shape=config.input_shape,
        )
    elif modeltype == "linear_mlp":
        model = LinearMLP(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            n_hidden=config.n_hidden,
        )
    elif modeltype == "slimresnet":
        model = MTSlimResNet18(
            nclasses=config.nclasses,
            nf=config.nf,
        )
    elif modeltype == "simplecnn":
        model = SimpleCNN(
            input_shape=config.input_shape,
        )
    else:
        raise NotImplementedError(f"unknown model {modeltype}")
    return model

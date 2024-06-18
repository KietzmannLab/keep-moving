import torch
import torch.nn as nn
from torch.nn import functional as F
from avalanche.models import MultiHeadClassifier, MultiTaskModule

import numpy as np
from scipy.linalg import orth
from utils import scaled_orth


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(VGGBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding="same",
        )
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool1(x)
        return x


class VariableDepthVGG(MultiTaskModule):
    def __init__(self, block_sizes, dense_size, dropouts, input_shape=(3, 64, 64)):
        super().__init__()
        self.blocks = nn.ModuleDict()
        self.nblocks = len(block_sizes)
        self.dense_size = dense_size
        for i, (bsize, drop) in enumerate(zip(block_sizes, dropouts)):
            if i == 0:
                in_channels = 3
            else:
                in_channels = block_sizes[i - 1]

            block = VGGBlock(in_channels=in_channels, out_channels=bsize)
            dropout = nn.Dropout(p=drop)
            self.blocks.update({f"block_{i}": block})
            self.blocks.update({f"dropout_{i}": dropout})

        xdim = input_shape[1] // (2 ** len(block_sizes))  # each block halves size
        ydim = input_shape[2] // (2 ** len(block_sizes))  # each block halves size
        linear_in_dim = xdim * ydim * block_sizes[-1]

        self.blocks.update({"flatten": nn.Flatten()})
        self.blocks.update({"fc1": nn.Linear(linear_in_dim, dense_size)})
        self.blocks.update({"relu_fc1": nn.ReLU()})
        self.blocks.update({"dense_dropout": nn.Dropout(p=dropouts[-1])})
        self.classifier = MultiHeadClassifier(dense_size, initial_out_features=10)
        self.filter = torch.eye(dense_size)

    def forward(self, x, task_labels):
        for name, f in self.blocks.items():
            x = f(x)
        logits = self.classifier(x, task_labels)
        return logits

    def forward_with_auxiliary(self, x, task_labels):
        activations = dict()
        for name, f in self.blocks.items():
            x = f(x)
            activations[name] = x
        logits = self.classifier(x, task_labels)
        activations["readout"] = logits
        return logits, activations

    def reinitialize_classifier(self):
        self.classifier = MultiHeadClassifier(self.dense_size, initial_out_features=10)

    def get_embedding_layer(self):
        return self.blocks["dense_dropout"]

    def get_classifier(self):
        return self.classifier

    def get_projection_layer(self):
        return self.blocks["fc1"]

    def get_layers_with_init(self):
        layers_with_init = []
        for name, b in self.blocks.items():
            if name.startswith("block"):
                layers_with_init.append(b.conv1)
                layers_with_init.append(b.conv2)
            elif name.startswith("fc1"):
                layers_with_init.append(b)
        return layers_with_init

    def set_trainable_backbone(self, is_trainable):
        for layer in self.get_layers_with_init():
            for parameter in layer.parameters():
                parameter.requires_grad = is_trainable

    def get_parameter_groups(self):
        return [
            {
                "params": torch.nn.ParameterList(
                    self.get_layers_with_init()
                ).parameters()
            },
            {"params": self.classifier.parameters()},
        ]

    def forward_single_task(self, x: torch.Tensor, task_label: int) -> torch.Tensor:
        raise NotImplementedError

    def update_filter(
        self, readout_idx=[], rcond=None, device="cpu", scaled=False, control=False
    ):
        # if control is True we are using the control filter, which is identity such that
        # all dimensions are equally affected by regularization. Dot Product with identity
        # is a no-op
        if control:
            self.filter = torch.eye(self.dense_size).type_as(self.filter).to(device)
            print("CONTROL CONDITION SET. Filter is identity.")
            return self.filter

        print("updating filter")
        if len(readout_idx) == 0:
            print("no readouts selected")
            return  # nothing to see here, move along

        readout_names = [
            f"classifier.classifiers.{i}.classifier.weight" for i in readout_idx
        ]
        ws = [self.state_dict()[name] for name in readout_names]
        ws = torch.concat(ws).detach().cpu().numpy()
        ws = ws.astype(np.float64)
        print("combined readout shape: ", ws.shape)
        if not scaled:
            C = orth(ws.T, rcond)
        else:
            C = scaled_orth(ws.T, rcond)

        self.filter = torch.from_numpy(C).type_as(self.filter).to(device)
        print("updated filter.")
        print("filter shape: ", self.filter.shape)
        return self.filter

import avalanche as avl
import torch
from torch import nn
from torch.nn import functional as F
from avalanche.models import MultiHeadClassifier, MultiTaskModule
from scipy.linalg import orth
from utils import scaled_orth
import numpy as np


class SimpleCNN(MultiTaskModule):
    def __init__(self, input_shape=(3, 32, 32)):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_shape[0],
            out_channels=16,
            kernel_size=3,
            padding="same",
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, padding="same", bias=False
        )
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding="same", bias=False
        )
        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding="same", bias=False
        )

        self.conv5 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding="same", bias=False
        )

        self.conv6 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding="same", bias=False
        )

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=8)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 128, bias=False)
        self.readout = MultiHeadClassifier(128, initial_out_features=5)

        self.filter = torch.eye(128)

    def forward(self, x, y):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.max_pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.readout(x, task_labels=y)
        return x

    def forward_with_auxiliary(self, x, y):
        activations = dict()
        x = F.relu(self.conv1(x))
        activations["conv1"] = x
        x = F.relu(self.conv2(x))
        activations["conv2"] = x
        x = self.max_pool(x)
        activations["max_pool1"] = x
        x = F.relu(self.conv3(x))
        activations["conv3"] = x
        x = F.relu(self.conv4(x))
        activations["conv4"] = x
        x = self.max_pool(x)
        activations["max_pool2"] = x
        x = F.relu(self.conv5(x))
        activations["conv5"] = x
        x = F.relu(self.conv6(x))
        activations["conv6"] = x
        x = self.avg_pool(x)
        activations["avg_pool"] = x
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        activations["fc1"] = x
        x = self.readout(x, task_labels=y)
        activations["readout"] = x
        return x, activations

    def reinitialize_classifier(self):
        self.readout = MultiHeadClassifier(128, initial_out_features=5)

    def get_embedding_layer(self):
        raise NotImplementedError(
            "make sure you know what you're doing if you use this. No nonlinearity has been applied yet!"
        )
        return self.fc1

    def get_layers_with_init(self):
        return [
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.conv6,
            self.fc1,
        ]

    def get_parameter_groups(self):
        return [
            {
                "params": torch.nn.ParameterList(
                    self.get_layers_with_init()
                ).parameters()
            },
            {"params": self.readout.parameters()},
        ]

    def set_trainable_backbone(self, is_trainable):
        params = self.get_parameter_groups()[0]["params"]
        for parameter in params:
            parameter.requires_grad = is_trainable

    def get_classifier(self):
        return self.readout

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
            f"readout.classifiers.{i}.classifier.weight" for i in readout_idx
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

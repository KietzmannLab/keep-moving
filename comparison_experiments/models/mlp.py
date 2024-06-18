import torch
import torch.nn as nn
from torch.nn import functional as F
from avalanche.models import MultiHeadClassifier, MultiTaskModule

import numpy as np
from scipy.linalg import orth


class MultiHeadMLP(MultiTaskModule):
    def __init__(self, input_size=28 * 28, hidden_size=256, drop_rate=0):
        super().__init__()
        self._input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(p=drop_rate)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(p=drop_rate)
        self.relu2 = nn.ReLU()
        self.hidden_size = hidden_size
        self.classifier = MultiHeadClassifier(
            self.hidden_size, initial_out_features=10, masking=True
        )

        self.filter = torch.eye(hidden_size)

    def reinitialize_classifier(self):
        self.classifier = MultiHeadClassifier(
            self.hidden_size, initial_out_features=10, masking=True
        )

    def forward(self, x, task_labels):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        act_fc1 = self.relu1(self.dropout1(self.fc1(x)))

        act_fc2 = self.relu2(self.dropout2(self.fc2(act_fc1)))

        x = self.classifier(act_fc2, task_labels)
        return x

    def forward_with_auxiliary(self, x, task_labels):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        act_fc1 = F.relu(self.dropout1(self.fc1(x)))

        act_fc2 = F.relu(self.dropout2(self.fc2(act_fc1)))

        x = self.classifier(act_fc2, task_labels)

        activations = {"fc1": act_fc1, "fc2": act_fc2}

        return x, activations

    def get_layers_with_init(self):
        return [self.fc1, self.fc2]

    def set_trainable_backbone(self, is_trainable):
        for layer in self.get_layers_with_init():
            for parameter in layer.parameters():
                parameter.requires_grad = is_trainable

    def get_embedding_layer(self):
        return self.relu2

    def get_classifier(self):
        return self.classifier

    def get_parameter_groups(self):
        return [
            {
                "params": torch.nn.ParameterList(
                    self.get_layers_with_init()
                ).parameters()
            },
            {"params": self.classifier.parameters()},
        ]

    def get_projection_layer(self):
        return self.fc2

    def forward_single_task(self, x: torch.Tensor, task_label: int) -> torch.Tensor:
        raise NotImplementedError

    def update_filter(self, readout_idx=[], rcond=None, device="cpu"):
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
        C = orth(ws.T, rcond)
        print("orthonormal basis 'C' shape: ", C.shape)
        CC_T = C @ C.T
        print("CC_T shape: ", CC_T.shape)
        self.filter = torch.from_numpy(CC_T).type_as(self.filter).to(device)
        print("updated filter.")
        print("filter shape: ", self.filter.shape)
        return self.filter

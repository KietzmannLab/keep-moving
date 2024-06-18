import torch
import torch.nn as nn
from torch.nn import functional as F
from avalanche.models import MultiHeadClassifier, MultiTaskModule


class MultiHeadMLP(MultiTaskModule):
    def __init__(self, input_size=28 * 28, hidden_size=256, n_hidden=1):
        super().__init__()
        self._input_size = input_size
        self.hidden_size = hidden_size
        self.layers = torch.nn.ModuleDict()
        self.layers["fc1"] = nn.Linear(input_size, hidden_size)
        for i in range(1, n_hidden):
            self.layers[f"fc{i+1}"] = nn.Linear(hidden_size, hidden_size)
        self.classifier = MultiHeadClassifier(
            self.hidden_size, initial_out_features=10, masking=True
        )
        self.n_hidden = n_hidden

    def reinitialize_classifier(self):
        self.classifier = MultiHeadClassifier(
            self.hidden_size, initial_out_features=10, masking=True
        )

    def forward(self, x, task_labels):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        for i in range(self.n_hidden):
            x = self.layers[f"fc{i+1}"](x)
        return x

    def forward_with_auxiliary(self, x, task_labels):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        activations = {}
        for i in range(self.n_hidden):
            x = self.layers[f"fc{i+1}"](x)
            activations[f"fc{i+1}"] = x
        return x, activations

    def get_layers_with_init(self):
        return list(self.layers.values())

    def set_trainable_backbone(self, is_trainable):
        for layer in self.get_layers_with_init():
            for parameter in layer.parameters():
                parameter.requires_grad = is_trainable

    def get_embedding_layer(self):
        return self.layers[f"fc{self.n_hidden}"]

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
        return self.layers[f"fc{self.n_hidden}"]

    def forward_single_task(self, x: torch.Tensor, task_label: int) -> torch.Tensor:
        raise NotImplementedError

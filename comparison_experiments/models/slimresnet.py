import torch
import torch.nn as nn
from torch.nn.functional import relu
from avalanche.models import MultiHeadClassifier, MultiTaskModule, DynamicModule
from torch.nn.parameter import Parameter


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class MTSlimResNet18(MultiTaskModule, DynamicModule):
    """MultiTask Slimmed ResNet18."""

    def __init__(self, nclasses, nf=20):
        super().__init__()
        self.in_planes = nf
        block = BasicBlock
        num_blocks = [2, 2, 2, 2]

        self.avgpool = nn.AvgPool2d(8)
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.classifier = MultiHeadClassifier(nf * 8 * BasicBlock.expansion, nclasses)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, task_labels):
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 64, 64))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out, task_labels)
        return out

    def get_layers_with_init(self):
        # return all layers that have conv in their name
        return [name for name, _ in self.named_modules() if "conv" in name]

    def get_parameter_groups(self):
        return [
            {
                "params": torch.nn.ParameterList(
                    [
                        self.conv1,
                        self.bn1,
                        self.layer1,
                        self.layer2,
                        self.layer3,
                        self.layer4,
                    ]
                ).parameters()
            },
            {"params": self.classifier.parameters()},
        ]

    def set_trainable_backbone(self, is_trainable):
        params = self.get_parameter_groups()[0]["params"]
        for parameter in params:
            parameter.requires_grad = is_trainable

    def forward_with_auxiliary(self, x, task_labels):
        activations = dict()
        out = relu(self.bn1(self.conv1(x)))
        activations["conv1"] = out
        out = self.layer1(out)
        activations["layer1"] = out
        out = self.layer2(out)
        activations["layer2"] = out
        out = self.layer3(out)
        activations["layer3"] = out
        out = self.layer4(out)
        activations["layer4"] = out
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        activations["avg_pool2d"] = out
        out = self.classifier(out, task_labels)
        activations["readout"] = out
        return out, activations

    def get_embedding_layer(self):
        return self.avgpool

    def reinitialize_classifier(self):
        self.classifier = MultiHeadClassifier(
            self.nf * 8 * BasicBlock.expansion, self.nclasses
        )

    def get_classifier(self):
        return self.classifier

    def forward_single_task(self, x, task_label):
        raise NotImplementedError

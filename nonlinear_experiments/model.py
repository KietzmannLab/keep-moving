import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from load_data import map_classes_for_task


def build_conv_block(
    in_channels, out_channels, kernel_size=3, padding=1, with_bias=True
):
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=with_bias
        ),
        nn.ReLU(),
        nn.Conv2d(
            out_channels, out_channels, kernel_size, padding=padding, bias=with_bias
        ),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
    )


class Model(nn.Module):
    def __init__(
        self,
        channels_per_block=[8, 8, 16, 16, 32],
        dense_size=64,
        readout_classes=[5, 5],
        device="cpu",
        with_bias=False,
        l1=0,
    ):
        super(Model, self).__init__()
        self.device = device
        # 32 x 32 block
        self.blocks = nn.ModuleList()
        self.blocks.append(
            build_conv_block(3, channels_per_block[0], with_bias=with_bias)
        )
        # 16 x 16 block
        self.blocks.append(
            build_conv_block(
                channels_per_block[0], channels_per_block[1], with_bias=with_bias
            )
        )
        # 8 x 8 block
        self.blocks.append(
            build_conv_block(
                channels_per_block[1], channels_per_block[2], with_bias=with_bias
            )
        )
        # 4 x 4 block
        self.blocks.append(
            build_conv_block(
                channels_per_block[2], channels_per_block[3], with_bias=with_bias
            )
        )
        # 2 x 2 block
        self.blocks.append(
            build_conv_block(
                channels_per_block[3], channels_per_block[4], with_bias=with_bias
            )
        )

        self.blocks.append(nn.Flatten())
        # linear layer
        self.blocks.append(nn.Linear(channels_per_block[4], dense_size, bias=with_bias))
        self.blocks.append(nn.ReLU())

        self.readouts = nn.ModuleList(
            modules=[
                nn.Linear(dense_size, readout_classes[i], bias=with_bias)
                for i in range(len(readout_classes))
            ]
        )

        # create a dict with layer names
        self.layer_names = {}
        conv_counter = 0
        for n, p in self.blocks.named_parameters():
            # check if conv
            idx = n.split(".")
            if len(idx) == 3:
                name = f"conv_{conv_counter}.{idx[-1]}"
                conv_counter += 1
                self.layer_names[n] = name
            else:
                self.layer_names[n] = f"linear.{idx[-1]}"

        self.l1 = l1

        def hook_with_counter(module, input, output, counter):
            def save_activation(module, input, output):
                # note: explicitly do not detach the tensor so that we can compute regularization terms on it
                self.activations[counter] = output

            return save_activation

        # register a hook for every activation layer to get activations
        self.activations = {}
        hooks = {}
        counter = 0
        for block in self.blocks:
            # if block is Sequential, go one level deeper
            if isinstance(block, nn.Sequential):
                for layer in block:
                    if isinstance(layer, nn.ReLU):
                        hooks[counter] = layer.register_forward_hook(
                            hook_with_counter(layer, layer, layer, counter)
                        )
                        counter += 1

            else:
                if isinstance(block, nn.ReLU):
                    hooks[counter] = block.register_forward_hook(
                        hook_with_counter(block, block, block, counter)
                    )
                    counter += 1

    def forward(self, x, t):
        for block in self.blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        x = self.readouts[t](x)
        return x

    def compute_importances(self, dataloader, task, classes):
        print("updating ewc importances ...")
        # initialise a copy of blocks to save current weights
        self.ewc_parameters = []
        for p in self.blocks.parameters():
            self.ewc_parameters.append(p.detach().clone().data)

        # create matrices for every parameter in ewc_blocks
        self.fisher = []
        for p in self.ewc_parameters:
            self.fisher.append(torch.zeros_like(p, device=p.device))

        # set model to eval mode
        self.eval()
        self.zero_grad()  # clear gradients
        # iterate over dataloader
        for x, y in dataloader:
            # move to device
            y = map_classes_for_task(y, classes)
            x = x.to(self.device)
            y = y.to(self.device)
            # forward pass
            out = self.forward(x, task)
            # backward pass
            loss = F.cross_entropy(out, y)
            loss.backward()
            # accumulate fisher information
            for i, p in enumerate(self.blocks.parameters()):
                self.fisher[i] += p.grad.detach() ** 2

        # divide by number of samples
        for i in range(len(self.fisher)):
            self.fisher[i] /= len(dataloader)

    def ewc_penalty(self):
        loss = 0
        for importance, p, q in zip(
            self.fisher, self.blocks.parameters(), self.ewc_parameters
        ):
            loss += (importance * (p - q) ** 2).sum()
        return loss

    def l1_penalty(self):
        loss = 0
        for a in self.activations.values():
            loss += torch.mean(torch.abs(a))
        loss *= self.l1
        return loss


if __name__ == "__main__":
    model = Model(l1=0.0, with_bias=False)
    # pass a sample through the model
    x = torch.randn(1, 3, 32, 32)
    y = model(x, 0)
    # print activations
    for k, v in model.activations.items():
        print(k, v.shape)

    # compute l1
    loss = model.l1_penalty()
    print(loss)

    print(model.layer_names)

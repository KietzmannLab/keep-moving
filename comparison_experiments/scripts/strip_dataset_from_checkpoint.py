# %%
import torch
import avalanche as avl
import os
from avalanche.training.plugins.checkpoint import (
    CheckpointPlugin,
    FileSystemCheckpointStorage,
)

# %%

plug = CheckpointPlugin(
    FileSystemCheckpointStorage(
        directory="./logs/continual_ecoset/1687957718.7130935_ecoset_pretrain_1024_three_block_vgg/checkpoints"
    ),
    map_location="cpu",
)

checkpath = "./logs/continual_ecoset/1687957718.7130935_ecoset_pretrain_1024_three_block_vgg/checkpoints/1/checkpoint.pth"
check = plug.load_checkpoint(checkpath)

try:
    del check["strategy"].dataloader
except:
    pass
try:
    del check["strategy"]._eval_stream
except:
    pass
try:
    del check["strategy"].adapted_dataset
except:
    pass

import avalanche as avl
from avalanche.training.plugins import SupervisedPlugin
from avalanche.core import BaseSGDPlugin
from avalanche.training.templates import BaseSGDTemplate
import torch
import os
from os import path
import shutil


class EpochWeightSaver(SupervisedPlugin):
    def __init__(self, savedir, save_every=1):
        super().__init__()
        self.savedir = savedir
        if not path.isdir(savedir):
            os.makedirs(savedir)
        self.counter = 0
        self.save_every = save_every

    def after_training_epoch(self, strategy, *args, **kwargs):
        self.counter += 1
        savepath = path.join(
            self.savedir, f"checkpoint-{str(self.counter).zfill(6)}.pt"
        )
        if self.counter % self.save_every == 0:
            torch.save(strategy.model.state_dict(), savepath)


class EpochOptimStateSaver(SupervisedPlugin):
    def __init__(self, savedir, save_every=1):
        super().__init__()
        self.savedir = savedir
        if not path.isdir(savedir):
            os.makedirs(savedir)
        self.counter = 0
        self.save_every = save_every

    def after_training_epoch(self, strategy, *args, **kwargs):
        self.counter += 1
        savepath = path.join(
            self.savedir, f"checkpoint-{str(self.counter).zfill(6)}.pt"
        )
        if self.counter % self.save_every == 0:
            torch.save(strategy.optimizer.state_dict(), savepath)


class CheckpointCleaner(BaseSGDPlugin[BaseSGDTemplate]):
    def __init__(self, savedir, keep=1):
        super(CheckpointCleaner, self).__init__()
        self.savedir = savedir
        self.keep = keep

    def after_training(self, strategy: BaseSGDTemplate, *args, **kwargs):
        # get all checkpoints
        checkpoints = os.listdir(self.savedir)
        checkpoints.sort()
        # remove all but the last keep checkpoints
        print(f"Removing {max(len(checkpoints) - self.keep, 0)} checkpoints ...")
        if len(checkpoints) > self.keep:
            for checkpoint in checkpoints[: -(self.keep - 1)]:
                # remove dir
                shutil.rmtree(os.path.join(self.savedir, checkpoint))

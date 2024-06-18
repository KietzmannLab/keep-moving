import random
import torch
import h5py
from os import path
from pathlib import Path
from typing import Sequence, Optional, Union, Any
from torchvision.datasets import VisionDataset
from torchvision import transforms
from avalanche.benchmarks import nc_benchmark, NCScenario


_default_miniecoset_eval_transform = transforms.Compose(
    [
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(
            mean=[0.4987, 0.4702, 0.4050], std=[0.2711, 0.2635, 0.2810]
        ),
    ]
)

_default_miniecoset_train_transform = transforms.Compose(
    [transforms.Normalize(mean=[0.4987, 0.4702, 0.4050], std=[0.2711, 0.2635, 0.2810])]
)


class MiniEcoset(VisionDataset):
    """
    dataset for efficient interfacing with MiniEcoset
    """

    def __init__(self, split, dataset_path, transform=None):
        """
        Args:
            dataset_path (string): Path to the .h5 file
            transform (callable, optional): Optional transforms to be applied
                on a sample.
        """
        super().__init__(root=dataset_path, transform=transform)
        self.root_dir = dataset_path
        self.transform = transform

        with h5py.File(dataset_path, "r") as f:
            self.data = torch.from_numpy(f[split]["data"][()]).permute(
                (0, 3, 1, 2)
            )  # to match the CHW expectation of pytorch
            self.targets = torch.from_numpy(f[split]["labels"][()])

    def __len__(self):
        return len(self.targets)

    def __getitem__(
        self, idx
    ):  # accepts ids and returns the images and labels transformed to the Dataloader
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imgs = self.data[idx]
        imgs = imgs.float() / 255.0
        labels = self.targets[idx]

        if self.transform:
            imgs = self.transform(imgs)

        return imgs, labels


def _get_MiniEcoset_dataset(dataset_root, train_transform, val_transform):
    train_set = MiniEcoset(
        dataset_path=path.join(dataset_root, "miniecoset_100obj_64px.h5"),
        split="train",
        transform=train_transform,
    )
    val_set = MiniEcoset(
        dataset_path=path.join(dataset_root, "miniecoset_100obj_64px.h5"),
        split="val",
        transform=val_transform,
    )
    return train_set, val_set


def SplitMiniEcoset(
    n_experiences: int,
    *,
    seed: Optional[int] = None,
    fixed_class_order: Optional[Sequence[int]] = None,
    dataset_root: Union[str, Path],
    train_transform: Optional[Any] = _default_miniecoset_train_transform,
    eval_transform: Optional[Any] = _default_miniecoset_eval_transform,
) -> NCScenario:
    train_data, test_data = _get_MiniEcoset_dataset(
        dataset_root, train_transform, eval_transform
    )
    if fixed_class_order is None:
        random.seed(seed)
        # random shuffling of the cifar100 classes (labels 10-109)
        class_order = random.sample(range(0, 100), 100)
    else:
        class_order = fixed_class_order

    return nc_benchmark(
        train_data,
        test_data,
        n_experiences=n_experiences,
        task_labels=True,
        class_ids_from_zero_in_each_exp=True,
        shuffle=False,
        seed=None,
        fixed_class_order=class_order,
        per_exp_classes={i: 100 // n_experiences for i in range(n_experiences)},
        train_transform=train_transform,
        eval_transform=eval_transform,
    )

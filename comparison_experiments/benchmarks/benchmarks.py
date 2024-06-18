import avalanche as avl
from torchvision import transforms

from benchmarks.cifar110 import (
    SplitCIFAR110,
    _default_cifar100_train_transform,
    _default_cifar100_eval_transform,
    _cifar100_no_aug,
)

from benchmarks.continual_ecoset import create_ecoset_benchmark


_default_mnist_train_transform = transforms.Compose(
    [
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomCrop(28, padding=6),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(
            28, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True
        ),
    ]
)

_default_mnist_eval_transform = transforms.Compose(
    [transforms.Normalize((0.1307,), (0.3081,))]
)


def instantiate_benchmark(config):
    if config.name == "cifar110":
        if config.augment:
            train_transform = _default_cifar100_train_transform
            val_transform = _default_cifar100_eval_transform
        else:
            train_transform = _cifar100_no_aug
            val_transform = _cifar100_no_aug
        benchmark = SplitCIFAR110(
            n_experiences=config.n_experiences,
            seed=config.seed,
            fixed_class_order=config.fixed_class_order
            if not len(config.fixed_class_order) == 0
            else None,
            train_transform=train_transform,
            eval_transform=val_transform,
        )
    elif config.name == "split_mnist":
        benchmark = avl.benchmarks.SplitMNIST(
            config.n_experiences,
            return_task_id=True,
            class_ids_from_zero_in_each_exp=True,
            fixed_class_order=config.fixed_class_order
            if not len(config.fixed_class_order) == 0
            else None,
            train_transform=_default_mnist_train_transform,
        )
    elif config.name == "rotated_mnist":
        benchmark = avl.benchmarks.RotatedMNIST(
            n_experiences=config.n_experiences,
            rotations_list=config.rotations_list,
            return_task_id=config.return_task_id,
            train_transform=_default_mnist_train_transform,
            eval_transform=_default_mnist_eval_transform,
        )
    elif config.name == "ecoset":
        benchmark = create_ecoset_benchmark(
            classes_per_experience=config.classes_per_experience,
            root_path=config.data_root,
            img_size=config.img_size,
            fixed_class_order=config.fixed_class_order,
        )
    elif config.name == "tinyimagenet":
        benchmark = avl.benchmarks.SplitTinyImageNet(
            n_experiences=config.n_experiences,
            return_task_id=config.return_task_id,
            class_ids_from_zero_in_each_exp=True,
            seed=config.seed,
        )
    elif config.name == "splitcifar10":
        benchmark = avl.benchmarks.SplitCIFAR10(
            n_experiences=config.n_experiences,
            return_task_id=config.return_task_id,
            seed=config.seed,
            class_ids_from_zero_in_each_exp=True,
        )
    else:
        raise NotImplementedError(f"unknown benchmark {config.name}")

    return benchmark

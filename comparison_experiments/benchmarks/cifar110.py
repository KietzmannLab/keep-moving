from torchvision.datasets import CIFAR100
from torchvision import transforms
import random
from pathlib import Path
from typing import Sequence, Optional, Union, Any
from avalanche.benchmarks.datasets import CIFAR10, default_dataset_location
from avalanche.benchmarks import nc_benchmark, NCScenario
from avalanche.benchmarks.utils import (
    concat_datasets_sequentially,
)

_default_cifar100_train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ]
)

_default_cifar100_eval_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ]
)

_cifar100_no_aug = transforms.Compose([transforms.ToTensor()])


def _get_cifar10_dataset(dataset_root):
    if dataset_root is None:
        dataset_root = default_dataset_location("cifar10")

    train_set = CIFAR10(dataset_root, train=True, download=True)
    test_set = CIFAR10(dataset_root, train=False, download=True)

    return train_set, test_set


def _get_cifar100_dataset(dataset_root):
    if dataset_root is None:
        dataset_root = default_dataset_location("cifar100")

    train_set = CIFAR100(dataset_root, train=True, download=True)
    test_set = CIFAR100(dataset_root, train=False, download=True)

    return train_set, test_set


def SplitCIFAR110(
    n_experiences: int,
    *,
    seed: Optional[int] = None,
    fixed_class_order: Optional[Sequence[int]] = None,
    train_transform: Optional[Any] = _default_cifar100_train_transform,
    eval_transform: Optional[Any] = _default_cifar100_eval_transform,
    dataset_root_cifar10: Union[str, Path] = None,
    dataset_root_cifar100: Union[str, Path] = None
) -> NCScenario:
    """
    Creates a CL benchmark using both the CIFAR100 and CIFAR10 datasets.

    If the datasets are not present in the computer, this method will
    automatically download and store them in the data folder.

    The CIFAR10 dataset is used to create the first experience, while the
    remaining `n_experiences-1` experiences will be created from CIFAR100.

    The returned benchmark will return experiences containing all patterns of a
    subset of classes, which means that each class is only seen "once".
    This is one of the most common scenarios in the Continual Learning
    literature. Common names used in literature to describe this kind of
    scenario are "Class Incremental", "New Classes", etc. By default,
    an equal amount of classes will be assigned to each experience.

    This generator will apply a task label "0" to all experiences.

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label (always "0" for this specific
    benchmark).

    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param n_experiences: The number of experiences for the entire benchmark.
        The first experience will contain the entire CIFAR10 dataset, while the
        other n-1 experiences will be obtained from CIFAR100.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order ONLY for the incremental part, which is based on cifar100. The
        classes must be in range 0-99.
        If None, value of ``seed`` will be used to define the class order for
        the incremental batches on cifar100. If non-None, ``seed`` parameter
        will be ignored. Defaults to None.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default test transformation
        will be used.
    :param dataset_root_cifar10: The root path of the CIFAR-10 dataset.
        Defaults to None, which means that the default location for
        'cifar10' will be used.
    :param dataset_root_cifar100: The root path of the CIFAR-100 dataset.
        Defaults to None, which means that the default location for
        'cifar100' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """

    cifar10_train, cifar10_test = _get_cifar10_dataset(dataset_root_cifar10)
    cifar100_train, cifar100_test = _get_cifar100_dataset(dataset_root_cifar100)

    cifar_10_100_train, cifar_10_100_test, _ = concat_datasets_sequentially(
        [cifar10_train, cifar100_train], [cifar10_test, cifar100_test]
    )
    # cifar10 classes
    class_order = [_ for _ in range(10)]
    # if a class order is defined (for cifar100) the given class labels are
    # appended to the class_order list, adding 10 to them (since the classes
    # 0-9 are the classes of cifar10).
    if fixed_class_order is not None:
        class_order.extend([c + 10 for c in fixed_class_order])
    else:
        random.seed(seed)
        # random shuffling of the cifar100 classes (labels 10-109)
        cifar_100_class_order = random.sample(range(10, 110), 100)
        class_order.extend(cifar_100_class_order)

    return nc_benchmark(
        cifar_10_100_train,
        cifar_10_100_test,
        n_experiences=n_experiences,
        task_labels=True,
        class_ids_from_zero_in_each_exp=True,
        shuffle=False,
        seed=None,
        fixed_class_order=class_order,
        per_exp_classes={i: 110 // n_experiences for i in range(n_experiences)},
        train_transform=train_transform,
        eval_transform=eval_transform,
    )

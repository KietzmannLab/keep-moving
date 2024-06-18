import torchvision
import torchvision.transforms as transforms
import torch
import random


def load_cifar10():
    # transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # load datasets
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    return trainset, testset


def split_dataset(dataset, classes):
    datasets = []
    for cls in classes:
        datasets.append(
            torch.utils.data.Subset(
                dataset, [i for i in range(len(dataset)) if dataset[i][1] in cls]
            )
        )
    return datasets


def load_data(batchsize, nsplits, nworkers, seed, shuffle_classes):
    classes = list(range(10))

    if shuffle_classes:
        if seed is not None:
            random.seed(seed)
        random.shuffle(classes)

    # split evenly into nsplits
    classes = [classes[i::nsplits] for i in range(nsplits)]
    print(classes)

    train_data, test_data = load_cifar10()
    train_sets = split_dataset(train_data, classes)
    test_sets = split_dataset(test_data, classes)

    train_loaders = [
        torch.utils.data.DataLoader(
            train_sets[i],
            batch_size=batchsize,
            shuffle=True,
            num_workers=nworkers,
            persistent_workers=True if nworkers > 0 else False,
        )
        for i in range(nsplits)
    ]
    test_loaders = [
        torch.utils.data.DataLoader(
            test_sets[i],
            batch_size=batchsize,
            shuffle=False,
            num_workers=nworkers,
            persistent_workers=True if nworkers > 0 else False,
        )
        for i in range(nsplits)
    ]

    return list(zip(train_loaders, test_loaders)), classes


def map_classes_for_task(y, classes):
    return torch.tensor([classes.index(y[i].item()) for i in range(len(y))])

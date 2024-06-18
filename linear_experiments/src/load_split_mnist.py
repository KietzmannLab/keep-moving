from torchvision import datasets, transforms


def load():
    # train transform: slightly rotate translate and scale
    train_transform = transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomCrop(28, padding=4),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
            transforms.ToTensor(),
        ]
    )

    # get mnist
    mnist = datasets.MNIST(
        "./data", train=True, download=True, transform=train_transform
    )
    mnist_test = datasets.MNIST(
        "./data", train=False, download=True, transform=transforms.ToTensor()
    )

    # separate mnist into two tasks (0-4, 5-9)

    mnist_0_4 = [(x, y) for x, y in mnist if y < 5]
    mnist_5_9 = [(x, y - 5) for x, y in mnist if y >= 5]

    # same for test
    mnist_test_0_4 = [(x, y) for x, y in mnist_test if y < 5]
    mnist_test_5_9 = [(x, y - 5) for x, y in mnist_test if y >= 5]
    return (mnist_0_4, mnist_5_9), (mnist_test_0_4, mnist_test_5_9)


def load_five_way():
    # train transform: slightly rotate translate and scale
    train_transform = transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomCrop(28, padding=4),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
            transforms.ToTensor(),
        ]
    )

    # get mnist
    mnist = datasets.MNIST(
        "./data", train=True, download=True, transform=train_transform
    )
    mnist_test = datasets.MNIST(
        "./data", train=False, download=True, transform=transforms.ToTensor()
    )

    # separate mnist into five tasks (0, 1, 2, 3, 4)

    mnist_01 = [(x, y) for x, y in mnist if y < 2]
    mnist_23 = [(x, y - 2) for x, y in mnist if y >= 2 and y < 4]
    mnist_45 = [(x, y - 4) for x, y in mnist if y >= 4 and y < 6]
    mnist_67 = [(x, y - 6) for x, y in mnist if y >= 6 and y < 8]
    mnist_89 = [(x, y - 8) for x, y in mnist if y >= 8 and y < 10]

    # same for test
    mnist_test_01 = [(x, y) for x, y in mnist_test if y < 2]
    mnist_test_23 = [(x, y - 2) for x, y in mnist_test if y >= 2 and y < 4]
    mnist_test_45 = [(x, y - 4) for x, y in mnist_test if y >= 4 and y < 6]
    mnist_test_67 = [(x, y - 6) for x, y in mnist_test if y >= 6 and y < 8]
    mnist_test_89 = [(x, y - 8) for x, y in mnist_test if y >= 8 and y < 10]

    return (mnist_01, mnist_23, mnist_45, mnist_67, mnist_89), (
        mnist_test_01,
        mnist_test_23,
        mnist_test_45,
        mnist_test_67,
        mnist_test_89,
    )

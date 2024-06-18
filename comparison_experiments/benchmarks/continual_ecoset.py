import torch
import os
import h5py
import numpy as np
from torchvision import transforms
from torch.nn.functional import interpolate
from tqdm import tqdm

from avalanche.benchmarks import nc_benchmark


def get_train_transform(size):
    ecoset_train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size, scale=(0.8, 1.0), ratio=(1.0, 1.0), antialias=True
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            ),
        ]
    )
    return ecoset_train_transform


def get_test_transform():
    # no transforms needed for test set
    return None


class EcoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_path,
        classes,
        split="train",
        transform=None,
        target_transform=None,
        img_size=256,
        format="h5",
    ):
        super().__init__()
        self.is_loaded = False
        self.datasets = None
        self.split = split
        self.classes = classes
        self.root_path = root_path
        # only save data and labels once the dataset is loaded in memory
        self.data = None
        self.targets = None
        self.transform = transform
        self.target_transform = target_transform
        self.img_size = (img_size, img_size)  # always square for now
        self.length = 0
        self.format = format
        if self.format == "npz":
            self.root_path += "/npz"
            self.compute_length_npz()
            self.load_in_memory_npz()
        elif self.format == "h5":
            self.load_in_memory_h5()
        else:
            raise NotImplementedError(f"unknown format {self.format}")

    def __len__(self):
        return self.length

    def compute_length_h5(self):
        raise NotImplementedError("not implemented yet")

    def compute_length_npz(self):
        files = [os.path.join(self.root_path, f"class_{c}.npz") for c in self.classes]
        datasets = [np.load(f, allow_pickle=True) for f in files]
        # get shapes for each class
        shapes = [d[f"{self.split}_shape"] for d in datasets]

        # get len from all datasets
        nsamples = 0
        for s in shapes:
            nsamples += s[0]
        self.length = nsamples

    def load_in_memory_npz(self, labels_only=False):
        files = [os.path.join(self.root_path, f"class_{c}.npz") for c in self.classes]
        datasets = [np.load(f, allow_pickle=True) for f in files]
        # get shapes for each class
        shapes = [d[f"{self.split}_shape"] for d in datasets]

        # get len from all datasets
        nsamples = 0
        for s in shapes:
            nsamples += s[0]
        self.length = nsamples
        # create empty arrays
        if not labels_only:
            self.data = torch.empty((nsamples, 3, *self.img_size), dtype=torch.float32)
        self.targets = torch.empty((nsamples,), dtype=torch.int64)

        # fill arrays
        offset = 0
        for d, s, c in zip(datasets, shapes, self.classes):
            print(f"Loading data for class {c} in memory...")

            if not labels_only:
                class_imgs = d[f"{self.split}_data"]
                class_imgs = torch.from_numpy(class_imgs).permute(0, 3, 1, 2)
                print("resizing ...")
                # resize images
                class_imgs = interpolate(class_imgs, size=self.img_size)
                self.data[offset : offset + s[0]] = class_imgs

            class_labels = d[f"{self.split}_labels"]
            class_labels = torch.from_numpy(class_labels.astype(np.int64))
            self.targets[offset : offset + s[0]] = class_labels

            # update offset
            offset += s[0]

        # set flag
        if not labels_only:
            self.is_loaded = True

    def load_in_memory_h5(self):
        """
        load the dataset in memory before use
        """
        files = [
            h5py.File(os.path.join(self.root_path, f"class_{c}.h5"), "r")
            for c in self.classes
        ]
        datasets = [f[self.split] for f in files]
        # get len from all datasets
        nsamples = 0
        for d in datasets:
            nsamples += len(d["labels"])
        self.length = nsamples
        # create empty arrays
        self.data = torch.empty((nsamples, 3, *self.img_size), dtype=torch.float32)
        self.targets = torch.empty((nsamples,), dtype=torch.int64)
        # fill arrays
        offset = 0
        for d in datasets:
            print(f"Loading {d} in memory...")
            # nsamples in dataset
            ndata = d["data"].shape[0]
            # chunk into 256
            for i in tqdm(range(0, ndata, 256)):
                # get chunk
                start = i
                end = min(i + 256, ndata)
                chunk = d["data"][start:end].astype(np.float32) / 255.0
                # resize chunk
                chunk = torch.tensor(chunk, dtype=torch.float32).permute(0, 3, 1, 2)
                chunk = interpolate(chunk, size=self.img_size)
                # fill data
                self.data[offset + start : offset + end] = chunk
            # fill targets
            self.targets[offset : offset + ndata] = torch.tensor(
                d["labels"][:].astype(np.int64), dtype=torch.int64
            )
            # update offset
            offset += ndata
        # close all h5files
        for f in files:
            f.close()

        self.is_loaded = True

    def __getitem__(self, idx):
        if not self.is_loaded:
            if self.format == "npz":
                self.load_in_memory_npz()
            elif self.format == "h5":
                self.load_in_memory_h5()
            else:
                raise NotImplementedError(f"unknown format {self.format}")
            self.is_loaded = True
        x, y = self.data[idx], self.targets[idx]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y


def create_ecoset_benchmark(
    classes_per_experience, root_path, img_size=256, fixed_class_order=None
):
    ntasks = len(classes_per_experience)
    # classes per experience is list of integers
    if type(classes_per_experience[0]) is int:
        # for each experience, randomly draw unique classes without replacement from list of 565
        # shuffle the list of classes
        if fixed_class_order is None:
            classes = np.arange(565)
            np.random.shuffle(classes)
        else:
            assert len(fixed_class_order) == 565, "need to specify exactly 565 classes"
            classes = fixed_class_order
        # split into experiences taking number of classes indicated by classes_per_experience
        classes_per_experience = np.split(classes, np.cumsum(classes_per_experience))
    else:
        raise NotImplementedError("classes_per_experience must be a list of integers")
    # transform
    train_transform = get_train_transform(img_size)
    test_transform = get_test_transform()

    classes_per_experience = classes_per_experience[:ntasks]
    classes = np.concatenate(classes_per_experience).tolist()
    # load dataset with all needed classes
    train_dataset = EcoDataset(
        root_path=root_path, classes=classes, split="train", img_size=img_size
    )
    test_dataset = EcoDataset(
        root_path=root_path, classes=classes, split="val", img_size=img_size
    )

    return nc_benchmark(
        train_dataset,
        test_dataset,
        n_experiences=ntasks,
        task_labels=True,
        class_ids_from_zero_in_each_exp=True,
        shuffle=False,
        seed=None,
        fixed_class_order=classes,
        per_exp_classes={
            i: len(c) for i, c in enumerate(classes_per_experience[:ntasks])
        },
        train_transform=train_transform,
        eval_transform=test_transform,
    )

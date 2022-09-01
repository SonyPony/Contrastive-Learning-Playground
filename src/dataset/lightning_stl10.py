import pytorch_lightning as pl
import torch
import numpy as np

from PIL import Image
from dataclasses import dataclass, field
from torchvision.datasets import STL10, CIFAR10, VisionDataset
from torchvision import transforms
from typing import Callable, Optional
from util import ExDict
from torch.utils.data import DataLoader


class STL10Pair(STL10):
    # Based on https://github.com/chingyaoc/DCL/blob/master/utils.py
    # Creates positive pairs

    def __init__(self, samples_per_class: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int):
        img, label = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, axes=(1, 2, 0)))

        assert self.transform
        pos_a, pos_b = self.transform(img), self.transform(img)

        return pos_a, pos_b, label


class CIFAR10Pair(CIFAR10):
    def __init__(self, samples_per_class: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        targets = np.array(self.targets)
        original_samples_per_class = (targets == 0).sum()

        if samples_per_class:
            assert 0 < samples_per_class <= original_samples_per_class

            # sort by class id
            # 0, 1, 1, 3, 2 -> 0, 1, 1, 2, 3 (class indices) -> 0, 1, 2, 5, 4 (samples indices)
            sorted_samples_indices = targets.argsort()
            data = list()
            classes_count = len(self.classes)

            # sample first N samples from each class
            for i in range(classes_count):
                start_idx = original_samples_per_class * i
                data_indices = sorted_samples_indices[start_idx:start_idx+samples_per_class]
                data.append(self.data[data_indices])
            self.targets = np.array(range(classes_count)).repeat(samples_per_class).tolist()
            self.data = np.concatenate(data)

    def __getitem__(self, index: int):
        img, label = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        assert self.transform
        pos_a, pos_b = self.transform(img), self.transform(img)

        return pos_a, pos_b, label


@dataclass
class LightningDatasetWrapper(pl.LightningDataModule):
    root_dir: str
    data_loader: ExDict
    train_samples_per_class: int

    train_transform: Optional[Callable] = None
    test_transform: Optional[Callable] = None

    train: VisionDataset = field(init=False)
    val: VisionDataset = field(init=False)
    test: Optional[VisionDataset] = None

    def __post_init__(self):
        super().__init__()

    def setup(self, stage: Optional[str] = None):
        shared_params = {"root": self.root_dir, "download": True}

        self.train = CIFAR10Pair(samples_per_class=self.train_samples_per_class, train=True, transform=self.train_transform, **shared_params)
        self.val = CIFAR10Pair(train=False, transform=self.test_transform, **shared_params)
        self.memory_bank = CIFAR10Pair(samples_per_class=50, train=True, transform=self.test_transform, **shared_params)
        #self.train = STL10Pair(split="train+unlabeled", transform=self.train_transform, **shared_params)
        #self.val = STL10Pair(split="test", transform=self.test_transform, **shared_params)
        #self.memory_bank = STL10Pair(split="train", transform=self.test_transform, **shared_params)

    def memory_bank_data_loader(self):
        return DataLoader(self.memory_bank, shuffle=False, **self.data_loader)

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, **self.data_loader)

    def val_dataloader(self):
        return DataLoader(self.val, **self.data_loader)

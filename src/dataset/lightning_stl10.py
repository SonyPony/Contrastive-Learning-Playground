import pytorch_lightning as pl
import torch
import numpy as np

from .ss_sampler import SSSampler
from PIL import Image
from dataclasses import dataclass, field
from torchvision.datasets import STL10, CIFAR10, VisionDataset
from torchvision import transforms
from typing import Callable, Optional
from util import ExDict
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler
from .tinyimagenet import TinyImageNetPair, SubsetType


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
    def __init__(self, classes_count: Optional[int] = None, samples_per_class: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        targets = np.array(self.targets)
        original_samples_per_class = (targets == 0).sum()

        if samples_per_class or classes_count:
            self.class_count = classes_count if classes_count else len(self.classes)
            samples_per_class = samples_per_class if samples_per_class else len(self) // len(self.classes)

            assert 0 < samples_per_class <= original_samples_per_class

            # sort by class id
            # 0, 1, 1, 3, 2 -> 0, 1, 1, 2, 3 (class indices) -> 0, 1, 2, 5, 4 (samples indices)
            sorted_samples_indices = targets.argsort()
            data = list()

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
class DatasetBase(pl.LightningDataModule):
    root_dir: str
    data_loader: ExDict
    train_samples_per_class: int
    num_classes: int  # TODO pass to datasets
    supervised: bool
    false_positive_perc: float

    train_transform: Optional[Callable] = None
    test_transform: Optional[Callable] = None

    train: VisionDataset = field(init=False)
    val: VisionDataset = field(init=False)
    test: Optional[VisionDataset] = None

    def __post_init__(self):
        super().__init__()


@dataclass
class LightningDatasetWrapper(DatasetBase):
    def setup(self, stage: Optional[str] = None):
        shared_params = {"root_dir": self.root_dir, "classes_count": self.num_classes}
        #shared_params = {"root": self.root_dir, "download": True, "classes_count": self.num_classes}

        self.train = TinyImageNetPair(
            samples_per_class=self.train_samples_per_class,
            subset_type=SubsetType.TRAIN,
            transform=self.train_transform,
            **shared_params
        )

        self.val = TinyImageNetPair(subset_type=SubsetType.TEST, transform=self.test_transform, **shared_params)
        self.memory_bank = TinyImageNetPair(
            samples_per_class=200,
            subset_type=SubsetType.TRAIN,
            transform=self.test_transform,
            **shared_params
        )

        """self.train = CIFAR10Pair(
            samples_per_class=self.train_samples_per_class,
            train=True,
            transform=self.train_transform,
            **shared_params
        )"""

        #self.val = CIFAR10Pair(train=False, transform=self.test_transform, **shared_params)
        #self.memory_bank = CIFAR10Pair(samples_per_class=200, train=True, transform=self.test_transform, **shared_params)
        #self.train = STL10Pair(split="train+unlabeled", transform=self.train_transform, **shared_params)
        #self.val = STL10Pair(split="test", transform=self.test_transform, **shared_params)
        #self.memory_bank = STL10Pair(split="train", transform=self.test_transform, **shared_params)

    def memory_bank_data_loader(self):
        return DataLoader(self.memory_bank, shuffle=False, **self.data_loader)

    def train_dataloader(self):
        if self.supervised or self.false_positive_perc is None:
            return DataLoader(self.train, shuffle=True, **self.data_loader)

        return DataLoader(
            self.train,
            batch_sampler=SSSampler(
                batch_size=self.data_loader.get("batch_size"),
                samples_per_class=self.train.samples_per_class,
                classes_count=self.train.classes_count,
                false_negative_perc=self.false_positive_perc,
                drop_last=self.data_loader.get("drop_last")
            ),
            pin_memory=self.data_loader.get("pin_memory"),
            num_workers=self.data_loader.get("num_workers")
        )

    def val_dataloader(self):
        return DataLoader(self.val, **self.data_loader)

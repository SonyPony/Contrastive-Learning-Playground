import pytorch_lightning as pl
import torch
import numpy as np

from PIL import Image
from dataclasses import dataclass, field
from torchvision.datasets import STL10
from torchvision import transforms
from typing import Callable, Optional
from util import ExDict
from torch.utils.data import DataLoader


class STL10Pair(STL10):
    # Based on https://github.com/chingyaoc/DCL/blob/master/utils.py
    # Creates positive pairs

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int):
        img, label = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, axes=(1, 2, 0)))

        assert self.transform
        pos_a, pos_b = self.transform(img), self.transform(img)

        return pos_a, pos_b, label


@dataclass
class LightningSTL10Pair(pl.LightningDataModule):
    root_dir: str
    data_loader: ExDict

    train_transform: Optional[Callable] = None
    test_transform: Optional[Callable] = None

    train: STL10Pair = field(init=False)
    val: STL10Pair = field(init=False)
    test: Optional[STL10Pair] = None

    def __post_init__(self):
        super().__init__()

    def setup(self, stage: Optional[str] = None):
        shared_params = {"root": self.root_dir}

        self.train = STL10Pair(split="train+unlabeled", transform=self.train_transform, **shared_params)
        self.val = STL10Pair(split="test", transform=self.test_transform, **shared_params)
        #self.test = STL10Pair(**shared_params, split="train+unlabeled")

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, **self.data_loader)

    def val_dataloader(self):
        return DataLoader(self.val, **self.data_loader)

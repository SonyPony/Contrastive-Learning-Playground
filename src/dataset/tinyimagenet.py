import numpy as np
import torchvision.datasets as datasets

from enum import Enum
from typing import Optional


class SubsetType(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"



class TinyImageNet(datasets.ImageFolder):
    def __init__(self, root_dir: str, subset_type: SubsetType, transform = None, **kwargs):
        super().__init__(
            root=f"{root_dir}/{subset_type.value}",
            transform=transform,
            **kwargs
        )


class TinyImageNetPair(TinyImageNet):
    def __init__(self, root_dir, subset_type: SubsetType, transform = None, class_count: Optional[int] = None, samples_per_class: Optional[int] = None, **kwargs):
        super().__init__(
            root_dir=root_dir,
            subset_type=subset_type,
            transform=transform,
            **kwargs
        )

        targets = np.array(self.targets)
        original_samples_per_class = (targets == 0).sum()

        if samples_per_class or class_count:
            self.class_count = class_count if class_count else len(np.unique(self.targets))
            samples_per_class = samples_per_class if samples_per_class else len(self) // self.class_count

            assert 0 < samples_per_class <= original_samples_per_class

            data = list()

            for i in range(self.class_count):
                start_idx = original_samples_per_class * i
                data.extend(self.samples[start_idx:start_idx + samples_per_class])
            self.targets = np.array(range(self.class_count)).repeat(samples_per_class).tolist()
            self.samples = data
            self.imgs = self.samples

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = self.loader(path)

        assert self.transform
        pos_a, pos_b = self.transform(img), self.transform(img)

        return pos_a, pos_b, label

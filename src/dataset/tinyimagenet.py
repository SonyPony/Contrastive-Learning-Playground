import numpy as np
import torchvision.datasets as datasets

from enum import Enum
from typing import Optional


class SubsetType(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"



class TinyImageNet(datasets.ImageFolder):
    CLASSES_COUNT = 200

    def __init__(self, root_dir: str, subset_type: SubsetType, transform = None, **kwargs):
        super().__init__(
            root=f"{root_dir}/{subset_type.value}",
            transform=transform,
            **kwargs
        )


class TinyImageNetSupportSet(TinyImageNet):
    def __init__(
            self,
            root_dir,
            subset_type: SubsetType,
            support_set_size: int = 2,
            transform = None,
            classes_count: Optional[int] = None,
            samples_per_class: Optional[int] = None,
            **kwargs
    ):
        super().__init__(
            root_dir=root_dir,
            subset_type=subset_type,
            transform=transform,
            **kwargs
        )

        targets = np.array(self.targets)
        original_samples_per_class = (targets == 0).sum()
        original_class_count = len(np.unique(self.targets))

        self.support_set_size = support_set_size
        self.classes_count = classes_count if classes_count else original_class_count
        self.samples_per_class = samples_per_class if samples_per_class else original_samples_per_class

        if self.samples_per_class or self.classes_count:
            assert 0 < self.samples_per_class <= original_samples_per_class

            data = list()

            for i in range(self.classes_count):
                start_idx = original_samples_per_class * i
                data.extend(self.samples[start_idx:start_idx + self.samples_per_class])
            self.targets = np.array(range(self.classes_count)).repeat(self.samples_per_class).tolist()
            self.samples = data
            self.imgs = self.samples

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = self.loader(path)

        assert self.transform
        return [self.transform(img) for _ in range(self.support_set_size)], label, index

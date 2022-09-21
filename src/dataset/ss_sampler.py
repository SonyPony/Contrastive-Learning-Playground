import torch
import numpy as np
import random

from typing import List, Iterator
from torch.utils.data import Sampler
from dataclasses import dataclass
from itertools import chain


@dataclass()
class SSSampler(Sampler[List[int]]):
    batch_size: int
    samples_per_class: int
    classes_count: int
    false_negative_perc: float = 0.
    drop_last: bool = False

    def __post_init__(self):
        self.samples_count = self.samples_per_class * self.classes_count

    def __iter__(self) -> Iterator[List[int]]:
        # assume sorted ordering
        structured_indices = {
            i:
                [j for j in range(i * self.samples_per_class, (i + 1) * self.samples_per_class)]
            for i in range(self.classes_count)}

        for _ in range(len(self)):
            used_classes = set()
            classes = set(range(self.classes_count))
            samples = list()

            for i in range(self.batch_size):
                empty_classes = set([i for i in structured_indices.keys() if not len(structured_indices[i])])
                unused_classes = list(classes - (used_classes | empty_classes))

                # false negative ratio cannot be holded
                # TODO check
                if len(unused_classes) <= 1:
                    #all([len(structured_indices[i]) for i in structured_indices.keys()]):
                    flattened_indices = list(chain.from_iterable(structured_indices.values()))
                    sample_idx = random.choice(flattened_indices)
                    samples.append(sample_idx)
                    continue

                if self.false_negative_perc > random.random() and len(used_classes - empty_classes):     # create a false negative
                    # false negative -> it's already in the list of used classes
                    class_idx = random.choice(list(used_classes - empty_classes))

                else:   # true negative
                    class_idx = random.choice(unused_classes)
                    used_classes.add(class_idx)

                sample_idx = random.choice(structured_indices[class_idx])
                structured_indices[class_idx].remove(sample_idx)
                samples.append(sample_idx)
            yield samples


    def __len__(self):
        if self.drop_last:
            return self.samples_count // self.batch_size
        return (self.samples_count + self.batch_size - 1) // self.batch_size

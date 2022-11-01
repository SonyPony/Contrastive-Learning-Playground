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


@dataclass()
class ImbalancedSampler(Sampler[List[int]]):
    batch_size: int
    samples_per_class: int
    classes_count: int
    class_probs: List[float]
    drop_last: bool = False

    def __post_init__(self):
        assert len(self.class_probs) <= self.classes_count
        assert sum(self.class_probs) <= 1.0

        # if probs are missing, set the class probs to a uniform distributions
        missing_probs_count = self.classes_count - len(self.class_probs)
        self.class_probs.extend([
            (1.0 - sum(self.class_probs)) / missing_probs_count
            for _ in range(missing_probs_count)
        ])

        self.samples_count = self.samples_per_class * self.classes_count

    def __iter__(self) -> Iterator[List[int]]:
        # TODO modify to some sort of dataset info
        # assume sorted ordering
        structured_indices = {
            i:
                [j for j in range(i * self.samples_per_class, (i + 1) * self.samples_per_class)]
            for i in range(self.classes_count)}
        # [0, p1, p1+p2, ...] (doesn't contain the end value 1)
        cum_probs = np.concatenate(([0, ], np.array(self.class_probs).cumsum()[:-1]))[..., None]

        for _ in range(len(self)):      # for N batches
            samples = list()

            random_threshold = np.random.random((self.batch_size, 1))
            sampled_classes = np.greater(random_threshold, cum_probs.T).sum(axis=1) - 1

            for class_idx in sampled_classes:    # for B samples in a batch
                sample_idx = random.choice(structured_indices[class_idx])
                #structured_indices[class_idx].remove(sample_idx)
                samples.append(sample_idx)
            yield samples

    def __len__(self):
        if self.drop_last:
            return self.samples_count // self.batch_size
        return (self.samples_count + self.batch_size - 1) // self.batch_size
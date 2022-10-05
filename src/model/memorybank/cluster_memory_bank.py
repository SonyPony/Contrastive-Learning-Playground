import torch


class ClusterMemoryBank:
    def __init__(self):
        self.centroid, self.radius = None, None

    def empty(self, samples_count, num_features, device="cpu"):
        self.centroid = torch.zeros((samples_count, num_features), dtype=torch.float32, device=device)
        self.radius = torch.zeros(samples_count, dtype=torch.float32, device=device)

    def __getitem__(self, key):
        return self.centroid[key], self.radius[key]

    def is_empty(self) -> bool:
        return self.centroid is None or self.radius is None
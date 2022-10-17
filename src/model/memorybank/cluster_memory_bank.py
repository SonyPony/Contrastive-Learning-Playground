import torch


class ClusterMemoryBank:
    def __init__(self):
        self.centroid, self.radius = None, None
        self.decay_rate = 0.9

    def _decay(self, old, new):
        return old * (1 - self.decay_rate) + self.decay_rate * new

    def empty(self, samples_count, num_features, device="cpu"):
        if not (self.centroid is None or self.radius is None):
            return

        self.centroid = torch.zeros((samples_count, num_features), dtype=torch.float32, device=device)
        self.radius = torch.zeros(samples_count, dtype=torch.float32, device=device)

    def update_centroids(self, centroids, keys):
        self.centroid[keys] = self._decay(self.centroid[keys], centroids)

    def update_radius(self, radiuses, keys):
        self.radius[keys] = self._decay(self.radius[keys], radiuses)

    def __getitem__(self, key):
        return self.centroid[key], self.radius[key]

    def is_empty(self) -> bool:
        return self.centroid is None or self.radius is None
import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch.optim import Adam
from termcolor import cprint
from typing import Dict
from util import ExDict


def negative_mask(batch_size: int, device) -> torch.Tensor:
    mask = torch.ones((batch_size,) * 2, dtype=torch.bool, device=device)
    # remove identity
    mask = torch.logical_xor(mask, torch.eye(batch_size, dtype=torch.bool, device=device))
    # replicate for augmented samples as well
    return mask.repeat(2, 2)


class LightningModelWrapper(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optim_parameters: Dict,
        batch_size: int,
        temperature: float = 0.5,
        tau_plus: float = 0.0,
        debiased: bool = False
    ):
        super().__init__()

        self.model = model
        self.temperature = temperature
        self.tau_plus = tau_plus
        self.optim_parameters = optim_parameters
        self.batch_size = batch_size
        self.debiased = debiased

        self.save_hyperparameters(ignore="model")

    def loss(self, projected_a: torch.Tensor, projected_b: torch.Tensor):
        samples_count = 2 * self.batch_size

        # neg score
        out = torch.cat([projected_a, projected_b], dim=0)  # all samples (2*batch_size, N)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)     # NOQA

        # mask out identities
        mask = negative_mask(self.batch_size, device=self.device)
        neg = neg[mask].view(samples_count, -1)

        # pos score
        # row-wise dot product
        pos = torch.exp(torch.sum(projected_a * projected_b, dim=-1) / self.temperature)       # NOQA
        pos = torch.cat([pos, pos], dim=0)

        # estimator g()
        if self.debiased:
            N = samples_count - 2
            neg = (-self.tau_plus * N * pos + neg.sum(dim=-1)) / (1 - self.tau_plus)
            # constrain (optional)
            neg = torch.clamp(neg, min=N * torch.e ** (-1 / self.temperature))

        else:
            neg = neg.sum(dim=-1)

        return (-torch.log(pos / (pos + neg))).mean()

    def forward(self, x):
        return self.model(x)

    def _step(self, pos_a, pos_b, label):
        _, projected_a = self.model(pos_a)
        _, projected_b = self.model(pos_b)

        return self.loss(projected_a, projected_b)

    def training_step(self, batch, batch_idx):
        pos_a, pos_b, label = batch
        loss = self._step(pos_a, pos_b, label)
        return loss

    def validation_step(self, batch, batch_idx):
        pos_a, pos_b, label = batch
        loss = self._step(pos_a, pos_b, label)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), **self.optim_parameters)

    def merge_state_dict(self, state_dict: Dict):
        model_state_dict = self.state_dict()
        model_layer_names = tuple(model_state_dict.keys())
        unloaded_layers = set(model_layer_names)

        for k, v in state_dict.items():
            if k in model_layer_names and model_state_dict[k].shape == v.shape:
                model_state_dict[k] = v
                unloaded_layers.remove(k)

        coverage = 100 * (1 - len(unloaded_layers) / len(model_layer_names))
        cprint(f"Pretrained model coverage: {coverage:.1f}%", "yellow")
        self.load_state_dict(model_state_dict)

        return unloaded_layers

    def merge_pretrained_model(self, model_path: str):
        loaded_weights = torch.load(
            model_path,
            map_location="cpu"
        )["state_dict"]

        return self.merge_state_dict(loaded_weights)
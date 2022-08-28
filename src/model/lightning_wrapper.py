import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch.optim import Adam
from termcolor import cprint
from typing import Dict


def negative_mask(batch_size: int) -> torch.Tensor:
    mask = torch.ones((batch_size,) * 2, dtype=torch.bool)
    # remove identity
    mask = torch.logical_xor(mask, torch.eye(batch_size, dtype=torch.bool))
    # replicate for augmented samples as well
    return mask.repeat(2, 2)


class LightningModelWrapper(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optim_parameters: Dict,
        temperature: float = 0.5,
        tau_plus: float = 0.0
    ):
        super().__init__()

        self.model = model
        self.temperature = temperature
        self.tau_plus = tau_plus
        self.optim_parameters = optim_parameters

        #self.save_hyperparameters(ignore=["model"])

    #def loss(self, x):
    #    pass

    def forward(self, x):
        return self.model(x)

    def _step(self, pos_a, pos_b, label):
        batch_size = 256
        feature_a, projected_a = self.model(pos_a)
        feature_b, projected_b = self.model(pos_b)

        # neg score
        out = torch.cat([projected_a, projected_b], dim=0)  # all samples (2*batch_size, N)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = negative_mask(batch_size).cuda()
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        # row-wise dot product
        pos = torch.exp(torch.sum(projected_a * projected_b, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        # estimator g()
        debiased=False
        if debiased:
            pass
            #N = batch_size * 2 - 2
            #Ng = (-tau_plus * N * pos + neg.sum(dim=-1)) / (1 - tau_plus)
            # constrain (optional)
            #Ng = torch.clamp(Ng, min=N * np.e ** (-1 / temperature))
        else:
            Ng = neg.sum(dim=-1)
        loss = (- torch.log(pos / (pos + Ng) )).mean()
        return loss

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
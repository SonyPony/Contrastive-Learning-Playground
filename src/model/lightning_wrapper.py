import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

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

        self.val_acc_t_1 = torchmetrics.Accuracy()
        self.val_acc_t_5 = torchmetrics.Accuracy(top_k=5)

        self.feature_bank = None
        self.feature_labels = None

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


    def training_step(self, batch, batch_idx):
        pos_a, pos_b, label = batch
        _, projected_a = self.model(pos_a)
        _, projected_b = self.model(pos_b)

        return self.loss(projected_a, projected_b)

    def validation_step(self, batch, batch_idx):
        sim_samples_count = 128     # number of most similar samples
        sample, _, sample_label = batch
        batch_size = sample.size(0)

        # get sample features
        sample_feature, _ = self.model(sample)

        # compute the similarity of the sample to the feature bank
        # [B, N] - B is the batch size, N is the memory bank size
        sim_matrix = torch.mm(sample_feature, self.feature_bank)

        # get 'k' most similar samples from the feature bank
        sim_weight, sim_indices = sim_matrix.topk(k=sim_samples_count, dim=-1)
        # [B, K]
        sim_labels = torch.gather(self.feature_labels.expand(batch_size, -1), dim=-1, index=sim_indices)
        sim_weight = (sim_weight / self.temperature).exp()

        # counts for each class
        one_hot_label = torch.zeros(sample.size(0) * sim_samples_count, self.class_count, device=self.device)

        # [B*K, C] Creating one hot encoding for flattened feature bank
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)

        # weighted score ---> [B, C] - weightning classes based on the similarity
        pred_scores = torch.sum(one_hot_label.view(sample.size(0), -1, self.class_count) * sim_weight[..., None], dim=1)

        # compute metrics and log them
        self.val_acc_t_1(pred_scores, sample_label)
        self.val_acc_t_5(pred_scores, sample_label)

        self.log("val/acc-1", self.val_acc_t_1)
        self.log("val/acc-5", self.val_acc_t_5)

    def on_validation_epoch_start(self):
        """
        Create validation accuracy - the predicted class is based on the label of a
        training with the most similar feature vector.
        """

        super().on_validation_epoch_start()
        feature_bank = list()

        # create features database
        cprint("Computing memory bank...", color="yellow")
        memory_dl = self.trainer.datamodule.memory_bank_data_loader()
        self.class_count = len(memory_dl.dataset.classes)

        for data, _, label in memory_dl:
            feature, _ = self.model(data.to(self.device, non_blocking=True))
            feature_bank.append(feature)

        # [D, N] - D feature dimensionality, N samples count
        self.feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N] - get all labels
        self.feature_labels = torch.tensor(memory_dl.dataset.labels, device=self.device)


    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()

        self.feature_bank = None
        self.feature_labels = None

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
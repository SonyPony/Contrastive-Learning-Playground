"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
Copied from https://github.com/HobbitLong/SupContrast/blob/master/losses.py
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConSigLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.bce_loss = nn.BCELoss()

    def forward(self, features, labels, device="cpu"):

        # neg score
        labels = labels[..., None]
        out = torch.squeeze(features)  # all samples (2*batch_size, N)
        similarities = torch.mm(out, out.t().contiguous())  # NOQA  (2*batch_size, 2*batch_size)
        equality_mask = torch.eq(labels, labels.T).float().to(device)
        mask = torch.logical_xor(torch.ones_like(similarities, dtype=bool), torch.eye(similarities.shape[0], device=device))
        mask = mask.flatten()

        similarities = similarities.flatten()[mask]
        equality_mask = equality_mask.flatten()[mask]

        similarities = (similarities + 1) / 2
        #similarities = F.sigmoid(similarities)
        return self.bce_loss(similarities, equality_mask)
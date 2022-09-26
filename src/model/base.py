import torch
import torch.nn as nn
import torch.nn.functional as F

from util.nn_module import freeze_model
from torchvision.models.resnet import resnet50, resnet18


class BaseModel(nn.Module):
    # Based on implementation https://github.com/chingyaoc/DCL/blob/master/model.py.

    def __init__(self, classes_count: int, feature_size: int = 128, supervised: bool = False, linear_eval: bool = False):
        super().__init__()

        self.supervised = supervised
        self.linear_eval = linear_eval
        self.encoder = list()
        for name, module in resnet18().named_children():
            if name == "conv1":     # replace first conv layer 7x7 with stride 2 for 3x3 stride 1
                module = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
            if type(module) in {nn.Linear, nn.MaxPool2d}:
                continue

            self.encoder.append(module)
        self.encoder = nn.Sequential(*self.encoder)

        # init projection head
        lin_features_size = 512

        # TODO Xor linear eval supervised
        if self.linear_eval:
            self.classifier = nn.Linear(in_features=lin_features_size, out_features=classes_count)
            freeze_model(self.encoder)

        if not self.linear_eval:
            self.projection_head = nn.Sequential(
                # 2048
                nn.Linear(in_features=512, out_features=lin_features_size, bias=False),
                nn.BatchNorm1d(num_features=lin_features_size),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=lin_features_size, out_features=feature_size, bias=True)
            )

    def forward(self, x):
        features = self.encoder(x)
        features = torch.flatten(features, start_dim=1)

        if not self.linear_eval:
            projected = self.projection_head(features)

        if self.supervised and self.linear_eval:
            return None, self.classifier(features)

        # unsupervised
        return F.normalize(features, dim=-1), F.normalize(projected, dim=-1)

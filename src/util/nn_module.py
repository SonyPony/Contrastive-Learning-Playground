import torch


def freeze_model(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    #model.eval()
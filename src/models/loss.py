import torch


def weighted_MSE_loss(output, target, weights):
    return torch.mean(weights * (output - target) ** 2)
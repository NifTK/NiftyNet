import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):

    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, output, label):
        probs = output.view(-1)
        mask = label.view(-1)
        smooth = 1
        intersection = torch.sum(probs * mask)
        den1 = torch.sum(probs)
        den2 = torch.sum(mask)
        soft_dice = (2 * intersection + smooth) / (den1 + den2 + smooth)
        return -soft_dice


def dice(input, target):
    epsilon = 1e-8
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 2 * intersection / (iflat.sum() + tflat.sum() + epsilon)

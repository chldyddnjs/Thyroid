import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        num = torch.sum(torch.mul(predict, target))*2 + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth

        dice = num / den
        loss = 1 - dice
        return loss

class BCEDice_loss(nn.Module):
    def __init__(self,smooth=1e-7,p=1,ratio=0.75):
        super().__init__()
        self.ratio = ratio
        self.smooth = smooth
        self.p = p
    def forward(self,pred,target):
        predict = pred.view(pred.shape[0], -1)
        target = target.view(target.shape[0], -1).float()
        
        # BCE loss
        bce_loss = nn.BCEWithLogitsLoss()(predict, target)
        
        num = torch.sum(torch.mul(predict, target))*2 + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth

        dice = num / den
        loss = (1- self.ratio) * bce_loss + self.ratio * (1 - dice)
        return loss
import torch
import torch.nn as nn

class MetricTracker(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def dice_coeff(input, target,threshold=0.5):
    num_in_target = input.size(0)
    input[input >= threshold] = float(1)
    input[input < threshold] = float(0)
    smooth = 1e-7

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    intersection = (pred * truth).sum(1)

    loss = (2.0 * intersection + smooth) / (pred.sum(1) + truth.sum(1) + smooth)

    return loss.mean().item()

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coeff(y_true, y_pred)

# def dice_coeff(input, target,threshold=0.5):
#     num_in_target = input.size(0)
#     input = input - threshold
#     input = torch.ceil(input)
#     smooth = 1e-7

#     pred = input.view(num_in_target, -1)
#     truth = target.view(num_in_target, -1)

#     intersection = (pred * truth).sum(1)

#     loss = (2.0 * intersection + smooth) / (pred.sum(1) + truth.sum(1) + smooth)

#     return loss.mean().item()

# def dice_coeff_for_batch(input, target,num_channel):
#     num_in_target = input.size(0)
#     total_acc = 0
#     for index in range(0,num_channel):
#         total_acc += dice_coeff(input[index],target[index])
#     return total_acc/num_channel

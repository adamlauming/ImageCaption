'''
Description: 
Author: Ming Liu (lauadam0730@gmail.com)
Date: 2021-06-06 14:10:29
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, list): alpha = torch.Tensor(alpha)
        self.alpha = alpha.cuda()
        self.size_average = size_average

    def forward(self, y_true, y_pred):
        pos = -y_true * ((1.0 - y_pred)**self.gamma) * torch.log(y_pred + 1e-6)
        neg = -(1.0 - y_true) * (y_pred**self.gamma) * torch.log(1.0 - y_pred + 1e-6)
        loss = self.alpha[0] * pos + self.alpha[1] * neg

        if self.size_average:
            loss_focal = loss.mean()
        else:
            loss_focal = loss.sum()

        return loss_focal


class BCELoss(nn.Module):
    def __init__(self, weight=None):
        super(BCELoss, self).__init__()
        self.weight = weight

    def forward(self, y_true, y_pred):
        bce = F.binary_cross_entropy(y_pred, y_true, weight=self.weight)

        return bce


class CELoss(nn.Module):
    def __init__(self, weight=None):
        super(CELoss, self).__init__()
        self.weight = weight

    def forward(self, y_true, y_pred):
        pos = -y_true * torch.log(y_pred + 1e-12)
        neg = -(1.0 - y_true) * torch.log(1.0 - y_pred + 1e-12)
        loss = pos + neg

        return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=0.01):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        intersect = (y_pred * y_true).sum()
        union = torch.sum(y_pred) + torch.sum(y_true)
        Dice = (2 * intersect + self.smooth) / (union + self.smooth)
        dice_loss = 1 - Dice

        return dice_loss


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_true, y_pred):
        mse = F.mse_loss(y_pred, y_true)

        return mse

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = x
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()
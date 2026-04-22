import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        probs = torch.sigmoid(inputs)
        p_t = torch.where(targets == 1, probs, 1 - probs)

        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = torch.where(targets == 1, 1 - self.alpha, self.alpha)

        loss = alpha_t * focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
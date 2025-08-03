import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        alpha: Weighting factor for class imbalance (scalar or tensor of shape [num_classes])
        gamma: Focusing parameter for hard examples
        reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: raw logits of shape [batch_size, num_classes]
        targets: ground truth labels of shape [batch_size]
        """
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).float()

        ce_loss = -targets_one_hot * log_probs
        focal_weight = (1 - probs) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss

        loss = focal_loss.sum(dim=1)  # sum over classes

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

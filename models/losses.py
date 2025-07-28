import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    """
    From: https://arxiv.org/pdf/2004.11362.pdf
    Adapted to allow embeddings and labels.
    """
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Normalize features
        features = F.normalize(features, dim=1)

        dot_product = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        # Exclude self-comparisons
        mask_fill = torch.ones_like(mask) - torch.eye(mask.shape[0], device=device)
        mask = mask * mask_fill

        exp_logits = torch.exp(logits) * mask_fill
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        loss = -mean_log_prob_pos.mean()
        return loss

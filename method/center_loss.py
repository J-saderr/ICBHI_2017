import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterLoss(nn.Module):
    """
    Center loss that encourages features of the same class to be close to a
    learned class centroid.
    Reference: Wen et al., "A Discriminative Feature Learning Approach for Deep Face Recognition".
    """

    def __init__(self, num_classes: int, feat_dim: int, lambda_center: float = 0.01):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_center = lambda_center

        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.lambda_center == 0.0:
            return features.new_zeros(1, requires_grad=False)

        labels = labels.long()
        centers_batch = self.centers.index_select(0, labels)
        loss = F.mse_loss(features, centers_batch)
        return self.lambda_center * loss


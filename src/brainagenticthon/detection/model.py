"""Small residual 3D classifier. No bounding boxes or clinical probabilities."""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(incoming, outgoing, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(4, outgoing), nn.ReLU(),
            nn.Conv3d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(4, outgoing),
        )
        self.skip = (nn.Identity() if incoming == outgoing and stride == 1 else
                     nn.Conv3d(incoming, outgoing, 1, stride=stride, bias=False))

    def forward(self, volume):
        return torch.relu(self.body(volume) + self.skip(volume))


class Tiny3DClassifier(nn.Module):
    """Single native-T1 channel -> one abnormality logit per volume."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 8), nn.ReLU(),
            ResidualBlock(8, 16, stride=2), ResidualBlock(16, 32, stride=2),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(0.2),
        )
        self.head = nn.Linear(32, 1)

    def forward(self, volume):
        if volume.ndim != 5 or volume.shape[1] != 1:
            raise ValueError("expected_batch_single_channel_3d")
        return self.head(self.features(volume)).squeeze(1)


@torch.no_grad()
def predict_scores(model, volumes):
    """Research sigmoid scores, uncalibrated; deliberately not TaskAOutput."""
    previous = model.training
    model.eval()
    try:
        scores = torch.sigmoid(model(volumes))
        if not torch.isfinite(scores).all():
            raise ValueError("nonfinite_model_scores")
        return scores
    finally:
        model.train(previous)

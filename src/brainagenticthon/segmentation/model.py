"""Compact two-level 3D U-Net with small-batch GroupNorm."""
import torch
from torch import nn
from torch.nn import functional as functional


def block(inputs, outputs):
    return nn.Sequential(nn.Conv3d(inputs, outputs, 3, padding=1), nn.GroupNorm(2, outputs),
                         nn.SiLU(), nn.Conv3d(outputs, outputs, 3, padding=1),
                         nn.GroupNorm(2, outputs), nn.SiLU())


class UNet3D(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.enc1 = block(4, width)
        self.enc2 = block(width, width * 2)
        self.bridge = block(width * 2, width * 4)
        self.dec2 = block(width * 6, width * 2)
        self.dec1 = block(width * 3, width)
        self.head = nn.Conv3d(width, 4, 1)

    def forward(self, image):
        if image.ndim != 5 or image.shape[1] != 4 or min(image.shape[2:]) < 8:
            raise ValueError('expected_B4XYZ_minimum_8')
        one = self.enc1(image)
        two = self.enc2(functional.max_pool3d(one, 2))
        bottom = self.bridge(functional.max_pool3d(two, 2))
        up = functional.interpolate(bottom, size=two.shape[2:], mode='trilinear',
                                    align_corners=False)
        up = self.dec2(torch.cat((up, two), 1))
        up = functional.interpolate(up, size=one.shape[2:], mode='trilinear', align_corners=False)
        return self.head(self.dec1(torch.cat((up, one), 1)))

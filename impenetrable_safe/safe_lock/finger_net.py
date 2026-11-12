"""The finger-counting neural network, shared by training, the live test and the UNO Q.

How pictures are prepared lives in pictures.py (shared with the UNO Q).
"""

import numpy as np
import torch
from torch import nn

from pictures import IMG_H, IMG_W, LABELS, SAVE_WIDTH, prepare  # noqa: F401  (others import these from here)


def to_tensor(pictures):
    """uint8 pictures (N, H, W, 3) -> float tensor (N, 3, H, W) with values around -1..1."""
    x = torch.as_tensor(np.asarray(pictures)).permute(0, 3, 1, 2).float()
    return x / 127.5 - 1.0


def block(c_in, c_out):
    # One "layer" of pattern finders: look at small 3x3 patches, then halve the picture size.
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class FingerNet(nn.Module):
    """Small CNN: 4 blocks of pattern finders (edges -> shapes -> fingers -> hands),
    then one final layer that votes for each of the 7 answers."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # squash each pattern map to one number
            nn.Flatten(),
            nn.Dropout(0.3),           # randomly ignore some neurons while learning, so it can't memorize
            nn.Linear(128, len(LABELS)),
        )

    def forward(self, x):
        return self.head(self.features(x))


def load(path, device="cpu"):
    net = FingerNet()
    net.load_state_dict(torch.load(path, map_location=device))
    return net.to(device).eval()

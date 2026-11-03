"""The step counter's neural network, and turning its output into a step count.
The answer key it learns from is made in labels.py.
"""
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks
from torch import nn

from labels import RATE

MODELS_DIR = Path(__file__).resolve().parent / "models"
MIN_STEP_GAP_S = 0.25  # nobody takes two steps closer together than this


class Block(nn.Module):
    """Look a bit wider (dilation), then add back what came in."""

    def __init__(self, width, dilation):
        super().__init__()
        self.conv = nn.Conv1d(width, width, 5, dilation=dilation, padding=2 * dilation)
        self.mix = nn.Conv1d(width, width, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        return x + self.mix(self.drop(torch.relu(self.conv(x))))


class StepNet(nn.Module):
    """A small 1D CNN. In: raw readings [batch, 7, time] (ax, ay, az, gx, gy, gz, dist_cm).
    Out: for every reading, how sure it is that a foot just landed (0..1).

    It slides over the readings like a magnifying glass about 1.3 s wide,
    looking for the shape of a step."""

    def __init__(self, use_distance=True, width=32):
        super().__init__()
        self.use_distance = use_distance
        self.inp = nn.Conv1d(10 if use_distance else 8, width, 7, padding=3)
        self.blocks = nn.Sequential(*[Block(width, d) for d in (1, 2, 4, 8, 16)])
        self.out = nn.Conv1d(width, 1, 1)

    def features(self, x):
        a, g, d = x[:, 0:3], x[:, 3:6] / 200, x[:, 6:7]
        parts = [a, g, a.norm(dim=1, keepdim=True) - 1, g.norm(dim=1, keepdim=True)]
        if self.use_distance:
            ok = (d >= 0).float()  # no echo -> "far away" plus a flag saying so
            parts += [torch.clamp(d, 0, 60) / 30 * ok + 2 * (1 - ok), ok]
        return torch.cat(parts, 1)

    def forward(self, x):
        h = torch.relu(self.inp(self.features(x)))
        return torch.sigmoid(self.out(self.blocks(h)))[:, 0]


def predict(model, s):
    """The network's 0..1 line for a whole trial."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(s[:, 1:8].T[None], dtype=torch.float32)
        return model(x)[0].numpy()


def count_steps(prob, threshold=0.5):
    """Each peak above the threshold is one step; its height is how sure the network is."""
    peaks, info = find_peaks(prob, height=threshold, distance=int(MIN_STEP_GAP_S * RATE))
    return peaks, info["peak_heights"]

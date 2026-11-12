"""The same finger-counting network, run with plain NumPy instead of PyTorch.

PyTorch is needed to TRAIN the network (on the laptop). To USE the trained
network you only need its numbers (weights) and the same math: multiply, add,
keep the positives, shrink. That's what this file does, so the UNO Q doesn't
need PyTorch.

  export(model.pt -> model.npz)   on the laptop (needs PyTorch), see push_to_board.py
  NumpyNet(model.npz).chances()   anywhere with NumPy (the UNO Q)

check_numpy_net.py proves this gives the same answers as the PyTorch version.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

EPS = 1e-5   # BatchNorm's small safety number (PyTorch's default)


def export(model_path, npz_path):
    """Read the trained PyTorch network and save its numbers for NumpyNet.

    Each BatchNorm step is folded into the conv before it (same math, fewer steps)."""
    import torch  # only needed here, on the laptop

    sd = torch.load(model_path, map_location="cpu")
    out = {}
    for b in range(4):                      # 4 blocks
        for c, (conv, bn) in enumerate([(0, 1), (3, 4)]):   # 2 convs per block
            w = sd[f"features.{b}.{conv}.weight"].numpy()
            g = sd[f"features.{b}.{bn}.weight"].numpy()
            beta = sd[f"features.{b}.{bn}.bias"].numpy()
            mean = sd[f"features.{b}.{bn}.running_mean"].numpy()
            var = sd[f"features.{b}.{bn}.running_var"].numpy()
            scale = g / np.sqrt(var + EPS)
            # (out, in, 3, 3) -> rows of (in*3*3) numbers, scaled; plus a bias
            out[f"b{b}c{c}_w"] = (w * scale[:, None, None, None]).reshape(w.shape[0], -1).T.astype(np.float32)
            out[f"b{b}c{c}_b"] = (beta - mean * scale).astype(np.float32)
    out["fc_w"] = sd["head.3.weight"].numpy().T.astype(np.float32)
    out["fc_b"] = sd["head.3.bias"].numpy().astype(np.float32)
    np.savez(npz_path, **out)


def conv3x3_relu(x, w, b):
    """x: picture (H, W, C). Look at every 3x3 patch, multiply by the pattern finders, keep positives."""
    h, wd, c = x.shape
    padded = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    patches = sliding_window_view(padded, (3, 3), axis=(0, 1))       # (H, W, C, 3, 3)
    y = patches.reshape(h * wd, c * 9) @ w + b
    return np.maximum(y, 0).reshape(h, wd, -1)


def maxpool2(x):
    """Keep the biggest number in each 2x2 square (halves the picture size)."""
    h, w, c = x.shape
    h2, w2 = h // 2, w // 2
    return x[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2, c).max(axis=(1, 3))


class NumpyNet:
    def __init__(self, npz_path):
        self.p = dict(np.load(npz_path))

    def chances(self, picture):
        """picture: uint8 (72, 128, 3) from pictures.prepare(). Returns 7 chances that add up to 1."""
        x = picture.astype(np.float32) / 127.5 - 1.0
        for b in range(4):
            x = conv3x3_relu(x, self.p[f"b{b}c0_w"], self.p[f"b{b}c0_b"])
            x = conv3x3_relu(x, self.p[f"b{b}c1_w"], self.p[f"b{b}c1_b"])
            x = maxpool2(x)
        scores = x.mean(axis=(0, 1)) @ self.p["fc_w"] + self.p["fc_b"]
        e = np.exp(scores - scores.max())
        return e / e.sum()

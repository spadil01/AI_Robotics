"""Prove the NumPy version of the network gives the same answers as PyTorch.

Runs both on every picture in data/ and compares.

Run:  python check_numpy_net.py
"""

import time
from pathlib import Path

import cv2
import numpy as np
import torch

from finger_net import LABELS, load, prepare, to_tensor
from numpy_net import NumpyNet, export

HERE = Path(__file__).parent


def main():
    export(HERE / "model.pt", HERE / "model.npz")
    torch_net = load(HERE / "model.pt")
    numpy_net = NumpyNet(HERE / "model.npz")

    files = sorted((HERE / "data").glob("*/*/*.jpg"))
    same, biggest_gap, numpy_time = 0, 0.0, 0.0
    for f in files:
        pic = prepare(cv2.imread(str(f)))
        with torch.no_grad():
            a = torch.softmax(torch_net(to_tensor([pic])), 1)[0].numpy()
        start = time.time()
        b = numpy_net.chances(pic)
        numpy_time += time.time() - start
        same += int(a.argmax() == b.argmax())
        biggest_gap = max(biggest_gap, float(np.abs(a - b).max()))

    print(f"{len(files)} pictures: same answer {same}/{len(files)}")
    print(f"biggest difference in any 'how sure' number: {100 * biggest_gap:.5f}%")
    print(f"NumPy speed on this computer: {1000 * numpy_time / len(files):.1f} ms per picture")
    print("labels:", LABELS)


if __name__ == "__main__":
    main()

"""Teach the finger-counting network from the pictures in data/.

Two ways to run it:

    python train.py --quiz-on victor
        Learns from everyone EXCEPT victor, then quizzes it on victor's
        pictures. Victor's hand is brand new to it, so the score shows how
        well it will work on strangers. Try each person in turn.

    python train.py
        Learns from everyone (the final version, for the safe).

Either way, the result is saved to model.pt. In quiz mode, the pictures it
got wrong are saved to mistakes.jpg so you can see what confused it.
"""

import argparse
import math
import random
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from finger_net import IMG_H, IMG_W, LABELS, FingerNet, prepare, to_tensor

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"


def load_pictures(people):
    pics, labels, owners = [], [], []
    for person in people:
        for i, lab in enumerate(LABELS):
            for f in sorted((DATA_DIR / person / lab).glob("*.jpg")):
                pics.append(prepare(cv2.imread(str(f))))
                labels.append(i)
                owners.append(person)
    return np.stack(pics), torch.tensor(labels), owners


def shake_up(x):
    """Make new practice pictures by slightly changing the real ones (augmentation):
    mirror, tilt, zoom, slide, and change brightness/colors. The network then can't
    just memorize exact pictures; it has to learn what fingers look like."""
    n, dev = x.shape[0], x.device
    flip = torch.rand(n, device=dev) < 0.5                       # left hand <-> right hand
    x = torch.where(flip[:, None, None, None], x.flip(3), x)

    angle = (torch.rand(n, device=dev) * 2 - 1) * math.radians(15)
    zoom = 1 / (0.85 + torch.rand(n, device=dev) * 0.3)
    c, s = torch.cos(angle) * zoom, torch.sin(angle) * zoom
    aspect = IMG_W / IMG_H
    theta = torch.stack([
        torch.stack([c, -s / aspect, (torch.rand(n, device=dev) * 2 - 1) * 0.12], 1),
        torch.stack([s * aspect, c, (torch.rand(n, device=dev) * 2 - 1) * 0.12], 1),
    ], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)

    contrast = 0.7 + torch.rand(n, 1, 1, 1, device=dev) * 0.6
    brightness = (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * 0.25
    tint = 1 + (torch.rand(n, 3, 1, 1, device=dev) * 2 - 1) * 0.1
    return (x * contrast * tint + brightness).clamp(-1, 1)


@torch.no_grad()
def guess(net, pics, device, batch=256):
    net.eval()
    out = [net(to_tensor(pics[i:i + batch]).to(device)).argmax(1).cpu() for i in range(0, len(pics), batch)]
    return torch.cat(out)


def print_table(truth, guesses):
    print("\nRows = real answer, columns = network's guess")
    print("      " + "".join(f"{lab:>6}" for lab in LABELS))
    for i, lab in enumerate(LABELS):
        row = [int(((truth == i) & (guesses == j)).sum()) for j in range(len(LABELS))]
        print(f"{lab:>6}" + "".join(f"{v:>6}" for v in row))


def save_mistakes(pics, truth, guesses, path, limit=48):
    wrong = (truth != guesses).nonzero().flatten().tolist()
    if not wrong:
        return
    random.shuffle(wrong)
    tiles = []
    for i in wrong[:limit]:
        tile = cv2.resize(pics[i], (IMG_W * 2, IMG_H * 2))
        cv2.putText(tile, f"real {LABELS[truth[i]]} / guess {LABELS[guesses[i]]}", (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(tile)
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
    cv2.imwrite(str(path), np.vstack(rows))
    print(f"Saved {min(len(wrong), limit)} of the {len(wrong)} wrong pictures to {path.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiz-on", help="person to leave out of learning and quiz on instead")
    ap.add_argument("--epochs", type=int, default=25, help="how many times to study all the pictures")
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    people, skipped = [], []
    for folder in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        complete = all(any((folder / lab).glob("*.jpg")) for lab in LABELS)
        (people if complete else skipped).append(folder.name)
    if skipped:
        print(f"Skipping {', '.join(skipped)}: no pictures for some answers")
    quiz_person = args.quiz_on.lower() if args.quiz_on else None
    if quiz_person and quiz_person not in people:
        raise SystemExit(f"No pictures for '{quiz_person}'. People: {', '.join(people)}")
    learn_people = [p for p in people if p != quiz_person]

    pics, labels, owners = load_pictures(learn_people)
    print(f"Learning from {len(pics)} pictures of {', '.join(learn_people)}  (on {device})")

    # Groups with fewer pictures get picked more often, so every person's version of
    # every answer counts equally (even if someone recorded only some answers).
    groups = list(zip(owners, labels.tolist()))
    group_size = Counter(groups)
    weights = torch.tensor([1 / group_size[g] for g in groups])

    x_all = to_tensor(pics).to(device)
    y_all = labels.to(device)

    net = FingerNet().to(device)
    batch = 64
    steps_per_epoch = len(pics) // batch
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=args.epochs * steps_per_epoch)

    if quiz_person:
        q_pics, q_labels, _ = load_pictures([quiz_person])
        print(f"Quiz: {len(q_pics)} pictures of {quiz_person} (never seen while learning)")

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        net.train()
        order = torch.multinomial(weights, steps_per_epoch * batch, replacement=True).to(device)
        right = total_loss = 0
        for step in range(steps_per_epoch):
            idx = order[step * batch:(step + 1) * batch]
            x, y = shake_up(x_all[idx]), y_all[idx]
            out = net(x)
            loss = F.cross_entropy(out, y, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total_loss += loss.item()
            right += (out.argmax(1) == y).sum().item()
        if epoch % 5 == 0 or epoch == 1:
            msg = f"round {epoch:3d}: practice score {100 * right / (steps_per_epoch * batch):5.1f}%"
            if quiz_person:
                q_score = (guess(net, q_pics, device) == q_labels).float().mean().item()
                msg += f"   quiz score on {quiz_person} {100 * q_score:5.1f}%"
            print(msg + f"   ({time.time() - start:.0f}s)")

    torch.save(net.state_dict(), HERE / "model.pt")
    print("Saved the network to model.pt")

    if quiz_person:
        g = guess(net, q_pics, device)
        print(f"\nFinal quiz score on {quiz_person}: {100 * (g == q_labels).float().mean():.1f}%")
        print_table(q_labels, g)
        save_mistakes(q_pics, q_labels, g, HERE / "mistakes.jpg")


if __name__ == "__main__":
    main()

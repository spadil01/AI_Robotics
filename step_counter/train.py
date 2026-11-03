"""Trains the step counter's neural network on the trials in data/.

    python step_counter/train.py

1. Makes the answer key for every trial (labels.find_steps) and checks it
   against the count you typed in (it only needs to be in the ballpark; on
   stairs and course walks your count leaves out turning steps, so the answer
   key should come out higher there).
2. Fair test: trains on every trial but one, counts the one left out, and
   repeats for each trial. Does this for the network, the network without
   the ultrasonic, and a no-AI counter, so we can see what each part adds.
3. Trains a final network on everything and saves:
     models/stepnet.onnx   the network, in a form the board can run
     models/report.json    the fair-test scores (the ± error the page shows)
     models/report.png     a picture of each trial: answer key vs the network
     models/loss.png       how the final network's score (loss) dropped as it trained
     models/stepnet.npz    the network's numbers, for the board (it runs them with numpy)
   and copies stepnet.npz + report.json into the board app
   (ArduinoApps/step-counter/python/model/) when that folder is next door.
"""
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

import labels as lb
import stepnet as sn

STEPS = 1500        # training rounds
BATCH = 32
CROP = 400          # each practice piece is 4 s of readings
SEED = 0
MISS_WEIGHT = 3     # a missed step costs this many times a false alarm (10 made it trigger-happy)
BOARD_MODEL_DIR = Path(__file__).resolve().parents[2] / "ArduinoApps" / "step-counter" / "python" / "model"


# ---------------- extra practice data ----------------

def random_rotation(rng, max_deg=60):
    """A random tilt, as if the strap sat a little differently."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.radians(rng.uniform(-max_deg, max_deg))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def practice_piece(rng, x, y):
    """A random 4 s piece of one trial, changed a little: faster or slower,
    stronger or weaker, tilted, mirrored (the other ankle), with the
    ultrasonic a bit higher or lower and sometimes dropping out."""
    stretch = rng.uniform(0.8, 1.25)  # >1 = slower walking
    need = int(CROP / stretch) + 2
    start = rng.integers(0, max(1, len(y) - need))
    src = start + np.arange(CROP) / stretch
    idx = np.arange(len(y))
    xs = np.stack([np.interp(src, idx, ch) for ch in x])
    ys = np.interp(src, idx, y)
    # Distance: hold the nearest reading (it's 20 a second), don't blend "no echo" (-1) in.
    xs[6] = x[6][np.clip(np.round(src).astype(int), 0, len(y) - 1)]

    rot = random_rotation(rng)
    a, g = rot @ xs[0:3], rot @ xs[3:6]
    if rng.random() < 0.5:  # mirror image: the sensor on the other leg
        a[0], g[1], g[2] = -a[0], -g[1], -g[2]
    amp = rng.uniform(0.8, 1.2)
    a = a.mean(1, keepdims=True) + (a - a.mean(1, keepdims=True)) * amp  # gravity stays 1 g
    g = g * amp / stretch
    a += rng.normal(0, 0.02, a.shape)
    g += rng.normal(0, 3, g.shape)

    d = xs[6].copy()
    ok = d >= 0
    d[ok] = np.clip(d[ok] * rng.uniform(0.7, 1.4) + rng.uniform(-3, 3), 2, None)
    if rng.random() < 0.3:  # a stretch with no echo
        n = rng.integers(20, 100)
        s0 = rng.integers(0, CROP - n)
        d[s0:s0 + n] = -1
    return np.vstack([a, g, d[None]]).astype(np.float32), ys.astype(np.float32)


# ---------------- training ----------------

def train(trials, use_distance, steps=STEPS, seed=SEED):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    data = [(s[:, 1:8].T.astype(np.float32), lb.answer_key(s, k)) for (_, s), k in trials]
    weights = np.array([len(y) for _, y in data], float)
    weights /= weights.sum()  # longer trials get picked more often
    # The Mac's graphics chip (MPS) learns ~18x faster than the main chip here.
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = sn.StepNet(use_distance).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    losses = []
    for _ in range(steps):
        pieces = [practice_piece(rng, *data[rng.choice(len(data), p=weights)]) for _ in range(BATCH)]
        x = torch.tensor(np.stack([p[0] for p in pieces])).to(dev)
        y = torch.tensor(np.stack([p[1] for p in pieces])).to(dev)
        model.train()
        p = model(x).clamp(1e-6, 1 - 1e-6)
        # Steps are rare moments: count a missed step worse than a false alarm.
        loss = -(MISS_WEIGHT * y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.detach())  # kept on the graphics chip: reading it each round would slow training
    model.losses = torch.stack(losses).cpu().numpy()
    return model.cpu()  # predict() and the ONNX export expect it on the main chip


def fair_test(trials):
    """Leave one trial out at a time; returns one row per trial.
    Trials whose answer key isn't trusted are tested but never learned from."""
    rows = []
    for i, ((meta, s), key) in enumerate(trials):
        rest = [t for j, t in enumerate(trials) if j != i and lb.trusted(t[0][0], t[1])]
        t0 = time.time()
        full = train(rest, use_distance=True)
        no_dist = train(rest, use_distance=False)
        threshold = best_simple_threshold(rest)
        net_th, no_dist_th = best_network_threshold(full, rest), best_network_threshold(no_dist, rest)
        prob = sn.predict(full, s)
        peaks, sure = sn.count_steps(prob, net_th)
        rows.append({
            "file": meta["file"], "activity": meta["activity"], "person": meta["person"],
            "typed": meta["steps"], "answer_key": len(key),
            # what to grade against: your count, unless it leaves out turning steps
            "true": meta["steps"] if lb.typed_counts_every_step(meta) else len(key),
            "network": len(peaks), "network_no_distance": len(sn.count_steps(sn.predict(no_dist, s), no_dist_th)[0]),
            "threshold": net_th,
            "no_ai": len(lb.simple_count(s, threshold)),
            "sureness": float(np.mean(sure)) if len(sure) else None,
            "prob": prob, "peaks": peaks,
        })
        print(f"  left out {meta['file']}: typed {meta['steps']}, answer key {len(key)}, "
              f"network {rows[-1]['network']}, no ultrasonic {rows[-1]['network_no_distance']}, "
              f"no-AI {rows[-1]['no_ai']}  (cutoff {net_th:.2f}, {time.time() - t0:.0f} s)")
    return rows


def best_simple_threshold(trials):
    """Tune the no-AI counter on the training trials only (to be fair to it)."""
    options = np.arange(1.1, 2.01, 0.05)
    err = [sum(abs(len(lb.simple_count(s, th)) - (meta["steps"] if lb.typed_counts_every_step(meta) else len(key)))
               for (meta, s), key in trials) for th in options]
    return float(options[int(np.argmin(err))])


def best_network_threshold(model, trials):
    """The cutoff (how sure counts as a step) that counts the training trials best."""
    options = np.arange(0.3, 0.96, 0.05)
    probs = [sn.predict(model, s) for (_, s), _ in trials]
    err = [sum(abs(len(sn.count_steps(p, th)[0]) - (meta["steps"] if lb.typed_counts_every_step(meta) else len(key)))
               for p, ((meta, _), key) in zip(probs, trials)) for th in options]
    return round(float(options[int(np.argmin(err))]), 2)


def step_length(trials):
    """Meters per step from the measured course walks: meters / the steps you
    counted (which leave out the turns), so it's the length of a straight step.
    The ± is how much it changed from walk to walk."""
    lengths = [meta["meters"] / meta["steps"] for (meta, _), _ in trials
               if meta.get("meters") and meta["steps"]]
    if len(lengths) < 2:
        return {"meters": 0.7, "percent": 20.0, "walks": len(lengths)}  # a typical adult's step, roughly
    return {"meters": round(float(np.mean(lengths)), 3),
            "percent": round(100 * float(np.std(lengths, ddof=1) / np.mean(lengths)), 1),
            "walks": len(lengths)}


def save_for_board(model, report):
    """The board runs the network with numpy (no PyTorch there): save its numbers."""
    weights = {k: v.numpy() for k, v in model.state_dict().items()}
    np.savez(sn.MODELS_DIR / "stepnet.npz", **weights)
    if BOARD_MODEL_DIR.parent.exists():
        BOARD_MODEL_DIR.mkdir(exist_ok=True)
        shutil.copy(sn.MODELS_DIR / "stepnet.npz", BOARD_MODEL_DIR)
        shutil.copy(sn.MODELS_DIR / "report.json", BOARD_MODEL_DIR)
        print(f"copied the network to {BOARD_MODEL_DIR} -- deploy the step-counter app to use it")


def error_summary(rows, name):
    """± error in steps: on trials with steps, the average miss as a % of the
    true count; on no-step trials, the average number of false steps."""
    walk = [r for r in rows if r["true"] > 0]
    fake = [r for r in rows if r["true"] == 0]
    return {
        "percent": round(100 * float(np.mean([abs(r[name] - r["true"]) / r["true"] for r in walk])), 1) if walk else None,
        "false_steps_when_not_walking": round(float(np.mean([r[name] for r in fake])), 1) if fake else None,
    }


def save_loss_picture(losses, path):
    """The loss after every training round (faint), and smoothed over 50 rounds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    smooth = np.convolve(losses, np.ones(50) / 50, mode="valid")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(losses, color="#2a7de1", alpha=0.25, lw=1, label="each round (4 s pieces x 32)")
    ax.plot(np.arange(len(smooth)) + 25, smooth, color="#2a7de1", lw=2, label="average of 50 rounds")
    ax.set_xlabel("training round")
    ax.set_ylabel("loss (lower = closer to the answer key)")
    ax.set_title("Final network: loss while training")
    ax.set_ylim(0, None)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=100)


def save_picture(trials, rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rows), 1, figsize=(14, 2.6 * len(rows)), squeeze=False)
    for ax, ((meta, s), key), r in zip(axes[:, 0], trials, rows):
        t = np.arange(len(s)) / lb.RATE
        ax.plot(t, lb.answer_key(s, key), color="#bbb", lw=3, label="answer key")
        ax.plot(t, r["prob"], color="#2a7de1", lw=1, label="network (never saw this trial)")
        ax.plot(r["peaks"] / lb.RATE, r["prob"][r["peaks"]], "o", color="#e8590c", ms=4, label="counted step")
        ax.set_title(f"{meta['activity']} — typed {r['typed']}, answer key {r['answer_key']}, network {r['network']}", fontsize=10)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlim(0, t[-1])
    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("seconds")
    plt.tight_layout()
    plt.savefig(path, dpi=80)


def main():
    trials = [(t, lb.find_steps(t[1], t[0])) for t in lb.load_trials()]
    print(f"{len(trials)} trials. Answer key vs your count:")
    for (meta, _), key in trials:
        note = "" if lb.typed_counts_every_step(meta) else "  (your count leaves out turns)"
        if not lb.trusted(meta, key):
            note += "  ** too far off: only used for testing"
        print(f"  {meta['file']}: you typed {meta['steps']}, the step finder found {len(key)}{note}")
    if len(trials) < 3:
        raise SystemExit("Need at least 3 trials with step counts to train and test fairly.")

    print("\nFair test (each trial counted by a network that never saw it):")
    rows = fair_test(trials)
    report = {
        "trials": len(trials), "people": len({m["person"] for (m, _), _ in trials}),
        "steps_typed": int(sum(m["steps"] for (m, _), _ in trials)),
        "fair_test": [{k: v for k, v in r.items() if k not in ("prob", "peaks")} for r in rows],
        "error": {name: error_summary(rows, name) for name in ("network", "network_no_distance", "no_ai")},
        "trained": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print("\n± error (steps) — lower is better:")
    for name, e in report["error"].items():
        print(f"  {name:20s} {e['percent']}% off on walks, {e['false_steps_when_not_walking']} false steps on no-step trials")

    print("\nTraining the final network on every trial...")
    trusted = [t for t in trials if lb.trusted(t[0][0], t[1])]
    model = train(trusted, use_distance=True)
    report["threshold"] = best_network_threshold(model, trusted)  # the board counts with this cutoff
    report["step_length"] = step_length(trials)
    print(f"cutoff: {report['threshold']}, step length: {report['step_length']['meters']} m "
          f"± {report['step_length']['percent']}%")
    sn.MODELS_DIR.mkdir(exist_ok=True)
    model.eval()
    torch.onnx.export(model, torch.zeros(1, 7, 300), sn.MODELS_DIR / "stepnet.onnx",
                      input_names=["readings"], output_names=["step_prob"],
                      dynamic_axes={"readings": {2: "time"}, "step_prob": {1: "time"}}, opset_version=17, dynamo=False)
    torch.save(model.state_dict(), sn.MODELS_DIR / "stepnet.pt")
    (sn.MODELS_DIR / "report.json").write_text(json.dumps(report, indent=2))
    save_picture([t for t in trials], rows, sn.MODELS_DIR / "report.png")
    save_loss_picture(model.losses, sn.MODELS_DIR / "loss.png")
    save_for_board(model, report)
    print(f"saved {sn.MODELS_DIR / 'stepnet.onnx'}, report.json, report.png and loss.png")


if __name__ == "__main__":
    main()

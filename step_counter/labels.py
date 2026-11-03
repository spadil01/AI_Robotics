"""The answer key for training: an automatic step finder that works out when
each foot landed from the ankle sensors, steered by the count you typed in.

Kept apart from stepnet.py so it loads fast (no PyTorch):

    python step_counter/labels.py      # check the answer key against your counts

Trials are the CSV files get_data.py copies into data/: one "# {...}" info
line, then rows of  ms, ax, ay, az (g), gx, gy, gz (deg/s), dist_cm
(-1 = no echo, -2 = sensor not answering), 100 rows a second.
"""
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RATE = 100            # readings a second
BUMP_S = 0.04         # each step in the answer key is a little bump this wide
BOUT_GAP_S = 2.2      # a longer pause between strides = stopped walking
SENSITIVITY = 0.35    # how big a swing must be to count (share of the trial's biggest)
SENSITIVITIES = np.arange(0.04, 0.61, 0.02)  # what the typed count may steer it between
TRUSTED_OFF = 0.25    # an answer key more than 25% off a complete typed count is too unsure to learn from


def load_trials(data_dir=DATA_DIR):
    """Every trial that has a step count typed in: [(info, readings), ...]."""
    trials = []
    for path in sorted(Path(data_dir).glob("*.csv")):
        with open(path) as f:
            meta = json.loads(f.readline()[2:])
        if meta.get("steps") is None:
            print(f"skipping {path.name}: no step count typed in yet")
            continue
        trials.append((meta, np.loadtxt(path, delimiter=",", comments="#", ndmin=2)))
    return trials


def typed_counts_every_step(meta):
    """On stairs and measured course walks, the typed count leaves out the
    steps taken while turning around at the ends; elsewhere it's every step."""
    return meta["activity"] != "stairs" and not meta.get("meters")


def lowpass(x, hz):
    b, a = butter(2, hz / (RATE / 2))
    return filtfilt(b, a, x, axis=0)


def swing_signal(s):
    """How fast the leg swings forward and back, whichever way the sensor is strapped on.

    Turning around spins the leg around the up-down line; swinging it while
    walking spins it around a sideways line. So: find "down" (where gravity
    points on average), throw away turning around it, and keep the direction
    the leg spins most in what's left."""
    a, g = s[:, 1:4], s[:, 4:7]
    down = a.mean(0) / np.linalg.norm(a.mean(0))
    g = g - np.median(g, 0)                 # remove the gyro's small constant drift
    g = g - np.outer(g @ down, down)        # remove turning around the up-down line
    _, _, vt = np.linalg.svd(g, full_matrices=False)
    w = lowpass(g @ vt[0], 3)
    # The forward swing is the biggest lobe of each stride: make it point up.
    return -w if np.percentile(-w, 99) > np.percentile(w, 99) else w


def steps_at(s, w, sensitivity):
    """Every foot landing for one sensitivity setting.

    1. Each big enough forward swing of the sensor leg is one stride.
    2. Its foot lands when the swing stops (the swing signal crosses zero);
       the exact moment is the sharpest jolt in the accelerometer near there.
    3. The other foot lands halfway between two landings of the sensor foot,
       plus one closing step when you stop.
    """
    jolt = np.linalg.norm(s[:, 1:4], axis=1)
    top = np.percentile(w, 99)
    swings, _ = find_peaks(w, height=sensitivity * top, prominence=0.85 * sensitivity * top,
                           distance=int(0.6 * RATE))
    mine = []
    for p in swings:
        after = np.nonzero(w[p:p + RATE // 2] < 0)[0]
        if not len(after):
            continue
        stop = p + after[0]
        lo, hi = max(0, stop - RATE // 10), min(len(s), stop + RATE * 15 // 100)
        mine.append(lo + int(np.argmax(jolt[lo:hi])))

    steps = list(mine)
    for k in range(len(mine)):
        gap_next = mine[k + 1] - mine[k] if k + 1 < len(mine) else None
        if gap_next is not None and gap_next < BOUT_GAP_S * RATE:
            steps.append(mine[k] + gap_next // 2)        # other foot, mid-stride
        else:
            prev = mine[k] - mine[k - 1] if k and mine[k] - mine[k - 1] < BOUT_GAP_S * RATE else RATE
            steps.append(min(len(s) - 1, mine[k] + prev // 2))  # closing step when stopping
    return np.array(sorted(steps), dtype=int)


def find_steps(s, meta=None):
    """Times (row numbers) of every foot landing in a trial.

    Your typed count steers it:
      - Typed 0: no steps at all (whatever the leg was doing).
      - "No steps" trials: only the last <typed> steps are real (walking back
        to stop the recording); everything before is faking, so not a step.
      - Trials where your count includes every step: pick the sensitivity
        whose count comes closest to yours (small shuffling steps need a
        more sensitive setting than walking).
      - Stairs and course walks (your count leaves out turning): the usual setting.
    """
    w = swing_signal(s)
    if meta is None:
        return steps_at(s, w, SENSITIVITY)
    typed = meta["steps"]
    if typed == 0:
        return np.array([], dtype=int)
    if meta["activity"] == "fake":
        steps = steps_at(s, w, SENSITIVITY)
        return steps[len(steps) - typed:] if typed else steps[:0]
    if typed_counts_every_step(meta):
        options = [steps_at(s, w, x) for x in SENSITIVITIES]
        # closest count wins; on a tie, the setting nearest the usual one
        best = min(range(len(options)), key=lambda i: (abs(len(options[i]) - typed),
                                                        abs(SENSITIVITIES[i] - SENSITIVITY)))
        return options[best]
    return steps_at(s, w, SENSITIVITY)


def trusted(meta, steps):
    """Is this answer key good enough to learn from? Only checkable when your
    count includes every step: then it must be within TRUSTED_OFF of it."""
    if not typed_counts_every_step(meta) or meta["steps"] == 0:
        return True
    return abs(len(steps) - meta["steps"]) <= TRUSTED_OFF * meta["steps"]


def answer_key(s, steps):
    """A 0..1 line, one value per reading: 1 right at each landing, a small bump around it."""
    if not len(steps):
        return np.zeros(len(s), np.float32)
    t = np.arange(len(s))[:, None]
    sigma = BUMP_S * RATE
    return np.exp(-0.5 * ((t - steps[None, :]) / sigma) ** 2).max(1).astype(np.float32)


def simple_count(s, threshold=1.3):
    """A no-AI counter to compare against: the classic pedometer trick of
    counting jolts in the accelerometer above a threshold."""
    jolt = lowpass(np.linalg.norm(s[:, 1:4], axis=1), 5)
    peaks, _ = find_peaks(jolt, height=threshold, distance=int(0.3 * RATE))
    return peaks


if __name__ == "__main__":
    for meta, s in load_trials():
        found = find_steps(s, meta)
        note = "" if typed_counts_every_step(meta) else "  (your count leaves out turns)"
        if not trusted(meta, found):
            note += "  ** too far off: only used for testing, not learning"
        print(f"{meta['file']:34s} {meta['activity']:7s} typed {meta['steps']:3d}   answer key {len(found):3d}{note}")

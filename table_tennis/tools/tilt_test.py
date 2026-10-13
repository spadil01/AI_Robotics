"""
Finds how the mounted motor reports the paddle's tilt (paddle face angle).

Asks you to hold the paddle in a few positions (straight, face tilted up,
face tilted down, for forehand and backhand) and prints the motor's angles
and gravity direction for each, so the game can turn tilt into high/low shots.

Run from the repo root:  .venv/bin/python table_tennis/tools/tilt_test.py
"""

import statistics
import threading
import time
import tomllib
from pathlib import Path

import legoeducation as le

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"
HOLD_SECONDS = 3
POSES = [
    ("forehand STRAIGHT", "Hold the paddle like you're about to hit a forehand, face straight up and down"),
    ("forehand UP", "Same forehand position, but tilt the face UP toward the ceiling (open)"),
    ("forehand DOWN", "Same forehand position, but tilt the face DOWN toward the floor (closed)"),
    ("backhand STRAIGHT", "Now the backhand position, face straight up and down"),
    ("backhand UP", "Backhand position, face tilted UP toward the ceiling"),
    ("backhand DOWN", "Backhand position, face tilted DOWN toward the floor"),
]
FIELDS = ("pitch", "roll", "yaw", "ax", "ay", "az")


class ImuLog:
    def __init__(self):
        self._lock = threading.Lock()
        self.samples = []

    def callback(self, data):
        for item in le.device_notification_parser(data):
            if isinstance(item, le.ImuDeviceNotification):
                with self._lock:
                    self.samples.append((item.pitch, item.roll, item.yaw, item.accelerometerX,
                                         item.accelerometerY, item.accelerometerZ))

    def take(self):
        with self._lock:
            samples, self.samples = self.samples, []
        return samples


def main():
    with open(CONFIG_PATH, "rb") as f:
        p = tomllib.load(f)["paddle"]
    log = ImuLog()
    motor = le.DoubleMotor()
    print("Connecting to the motor...")
    motor.connect(card_color=getattr(le, f"LEGO_COLOR_{p['card_color']}"), card_serial=p["card_serial"],
                  device_notification_delay=p["notification_delay_ms"])
    if not motor.connected:
        print("Couldn't connect. Is the motor on?")
        return
    motor.set_notification_callback(log.callback)

    results = []
    try:
        for name, instruction in POSES:
            input(f"\n{instruction}.\n  Press ENTER when you're holding it still...")
            log.take()
            time.sleep(HOLD_SECONDS)
            samples = log.take()
            if not samples:
                print("  No data from the motor!")
                continue
            medians = [statistics.median(col) for col in zip(*samples)]
            results.append((name, medians))
            print("  got it: " + "  ".join(f"{k}={v:.0f}" for k, v in zip(FIELDS, medians)))
    except KeyboardInterrupt:
        pass
    finally:
        motor.disconnect()

    print("\n=== Copy everything below and send it ===")
    print(f"{'pose':<20}" + "".join(f"{k:>8}" for k in FIELDS))
    for name, medians in results:
        print(f"{name:<20}" + "".join(f"{v:>8.0f}" for v in medians))


if __name__ == "__main__":
    main()

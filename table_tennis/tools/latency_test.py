"""
Day-1 measurement for the paddle: connects to the Double Motor and reports

  1. how often IMU updates actually arrive (rate and worst gaps),
  2. gyro noise while holding still vs. gyro peaks while shaking,
  3. how long beep and light commands take to complete.

Run from the repo root:  .venv/bin/python table_tennis/tools/latency_test.py
"""

import statistics
import threading
import time
import tomllib
from pathlib import Path

import legoeducation as le

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"

STILL_SECONDS = 5
SHAKE_SECONDS = 5
BEEP_TRIALS = 10


class ImuLog:
    """Collects IMU samples from the notification callback (BLE thread)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.samples = []  # (arrival_time, gx, gy, gz, yaw, pitch, roll)

    def callback(self, data):
        now = time.perf_counter()
        for item in le.device_notification_parser(data):
            if isinstance(item, le.ImuDeviceNotification):
                with self._lock:
                    self.samples.append((now, item.gyroscopeX, item.gyroscopeY, item.gyroscopeZ,
                                         item.yaw, item.pitch, item.roll))

    def take(self):
        """Return and clear everything collected so far."""
        with self._lock:
            samples, self.samples = self.samples, []
        return samples


def countdown(message, seconds):
    print(f"\n{message}")
    for s in range(3, 0, -1):
        print(f"  starting in {s}...")
        time.sleep(1)
    print(f"  GO ({seconds} s)")


def report_rate(samples, label):
    if len(samples) < 2:
        print(f"  {label}: only {len(samples)} samples, can't compute a rate")
        return
    gaps_ms = [(b[0] - a[0]) * 1000 for a, b in zip(samples, samples[1:])]
    duration = samples[-1][0] - samples[0][0]
    print(f"  {label}: {len(samples)} samples, {len(samples) / duration:.1f} Hz, "
          f"gap median {statistics.median(gaps_ms):.1f} ms, worst {max(gaps_ms):.1f} ms")


def report_gyro(samples, label):
    if not samples:
        return
    for axis, idx in (("gx", 1), ("gy", 2), ("gz", 3)):
        values = [abs(s[idx]) for s in samples]
        print(f"  {label} {axis}: median |value| {statistics.median(values):.0f}, max {max(values):.0f}")


def time_command(fn, trials):
    times_ms = []
    for _ in range(trials):
        start = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - start) * 1000)
        time.sleep(0.3)
    return times_ms


def main():
    with open(CONFIG_PATH, "rb") as f:
        paddle_cfg = tomllib.load(f)["paddle"]
    card_color = getattr(le, f"LEGO_COLOR_{paddle_cfg['card_color']}")

    log = ImuLog()
    doublemotor = le.DoubleMotor()
    try:
        print(f"Connecting to Double Motor ({paddle_cfg['card_color']} card, serial {paddle_cfg['card_serial']})...")
        doublemotor.connect(card_color=card_color, card_serial=paddle_cfg["card_serial"],
                            device_notification_delay=paddle_cfg["notification_delay_ms"])
        if not doublemotor.connected:
            print("Error connecting to Double Motor.")
            return
        print("Connected.")
        doublemotor.set_notification_callback(log.callback)

        countdown("Step 1: hold the motor completely still on the table.", STILL_SECONDS)
        log.take()
        time.sleep(STILL_SECONDS)
        still = log.take()

        countdown("Step 2: shake / swing the motor as fast as you can.", SHAKE_SECONDS)
        log.take()
        time.sleep(SHAKE_SECONDS)
        shake = log.take()

        print("\nStep 3: timing beeps and lights (listen for 20 beeps)...")
        beep_ms = time_command(lambda: doublemotor.beep(frequency=880), BEEP_TRIALS)
        light_ms = time_command(lambda: doublemotor.light_color(le.LEGO_COLOR_GREEN), BEEP_TRIALS)
        beep_nowait_ms = time_command(lambda: doublemotor.beep(frequency=660, blocking=False), BEEP_TRIALS)
        # Let the last fire-and-forget beeps finish; disconnecting while they're
        # still pending prints a harmless "Device disconnected unexpectedly".
        time.sleep(1.5)

        print("\n===== RESULTS =====")
        print(f"Requested update interval: {paddle_cfg['notification_delay_ms']} ms")
        report_rate(still, "IMU while still")
        report_rate(shake, "IMU while shaking")
        report_gyro(still, "noise ")
        report_gyro(shake, "shake ")
        print(f"  beep (wait for reply):  median {statistics.median(beep_ms):.0f} ms, worst {max(beep_ms):.0f} ms")
        print(f"  light (wait for reply): median {statistics.median(light_ms):.0f} ms, worst {max(light_ms):.0f} ms")
        print(f"  beep (fire and forget): median {statistics.median(beep_nowait_ms):.1f} ms to send")
        if shake:
            s = shake[len(shake) // 2]
            print(f"  sample yaw/pitch/roll: {s[4]} / {s[5]} / {s[6]}")
    finally:
        doublemotor.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()

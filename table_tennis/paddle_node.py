"""
Paddle node: the game's connection to the LEGO Double Motor paddle.

Reads the motor's IMU, detects swings, and publishes both to the local MQTT
broker. Beeps and flashes the light the moment a swing is detected (no round
trip through the game). Also listens for game events to flash result colors.

Run from the repo root:  .venv/bin/python table_tennis/paddle_node.py
Press Ctrl+C to quit.
"""

import json
import queue
import sys
import threading
import time

import legoeducation as le

from common import topics
from common.config import TABLE_TENNIS_DIR, load_config

sys.path.insert(0, str(TABLE_TENNIS_DIR.parent))
from mqttlib import MQTTClient  # noqa: E402

IDLE_LIGHT = le.LEGO_COLOR_AZURE
SWING_LIGHT = le.LEGO_COLOR_WHITE
EVENT_LIGHTS = {
    "hit": le.LEGO_COLOR_GREEN,
    "miss": le.LEGO_COLOR_RED,
    "point_won": le.LEGO_COLOR_GREEN,
    "point_lost": le.LEGO_COLOR_RED,
    "match_won": le.LEGO_COLOR_YELLOW,
}
FLASH_SECONDS = 0.25


class SwingDetector:
    """Finds swings in the gyro stream.

    A swing starts when the spin rate on the swing axis goes above
    `start_threshold`. It is reported as soon as the rate has clearly passed
    its peak (dropped below `peak_drop` x peak), so the game hears about it
    right after the moment of "impact" instead of after the follow-through.
    The swing then ends when the rate drops below `end_threshold`, followed by
    a cooldown so the follow-through can't count as a second swing.

    A weaker swing the other way right after one is the paddle being brought
    back, and is ignored. A much STRONGER one may be the real stroke after a wind-up
    (pulling the paddle back first): it is reported with "replaces": True so
    the game can swap it in for the wind-up.

    Only the swing axis is used: wiggles have as much total motion as soft
    swings, but spread over the other axes.
    """

    def __init__(self, cfg):
        self.axis_attr = f"gyroscope{cfg['swing_axis'].upper()}"
        self.start_threshold = cfg["start_threshold"]
        self.end_threshold = cfg["end_threshold"]
        self.min_duration = cfg["min_duration_ms"] / 1000
        self.peak_drop = cfg["peak_drop"]
        self.cooldown = cfg["cooldown_ms"] / 1000
        self.recovery = cfg["recovery_ms"] / 1000
        self.windup_ratio = cfg["windup_ratio"]
        self.last_side = 0
        self.last_peak = 0.0
        self.last_report = -1.0
        self.in_swing = False
        self.reported = False
        self.swing_start = 0.0
        self.cooldown_until = 0.0
        self.peak = None  # dict describing the strongest sample so far

    def update(self, t, imu):
        """Feed one IMU sample. Returns a swing dict once per swing, else None."""
        axis_rate = getattr(imu, self.axis_attr)
        rate = abs(axis_rate)

        if not self.in_swing:
            if rate > self.start_threshold and t >= self.cooldown_until:
                self.in_swing = True
                self.reported = False
                self.swing_start = t
                self.peak = None
            else:
                return None

        if self.peak is None or rate > self.peak["peak_rate"]:
            self.peak = {
                "t_peak": t,
                "peak_rate": rate,
                "side": 1 if axis_rate > 0 else -1,  # spin direction: forehand vs backhand
                "gyro": [imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ],
                "yaw": imu.yaw, "pitch": imu.pitch, "roll": imu.roll,
            }

        if rate < self.end_threshold:
            self.in_swing = False
            self.cooldown_until = t + self.cooldown
            return None

        past_peak = rate < self.peak["peak_rate"] * self.peak_drop
        if past_peak and not self.reported and t - self.swing_start >= self.min_duration:
            self.reported = True
            replaces = self.peak["side"] == -self.last_side and t - self.last_report < self.recovery
            if replaces and self.peak["peak_rate"] < self.last_peak * self.windup_ratio:
                return None                    # bringing the paddle back: not a swing
            self.last_side, self.last_peak, self.last_report = self.peak["side"], self.peak["peak_rate"], t
            return {**self.peak, "duration_ms": round((t - self.swing_start) * 1000), "replaces": replaces}
        return None


class PaddleNode:
    def __init__(self, cfg):
        self.cfg = cfg
        self.swing_cfg = cfg["swing"]
        self.detector = SwingDetector(self.swing_cfg)
        self.motor = le.DoubleMotor()
        # If this program dies, the broker tells everyone the paddle is gone.
        self.mqtt = MQTTClient(broker=cfg["mqtt"]["local_broker"], client_id="paddle_node",
                               will=(topics.PADDLE_STATUS, json.dumps({"connected": False}), 1, True))
        self.swing_count = 0
        self.commands = queue.Queue()
        threading.Thread(target=self.feedback_worker, daemon=True).start()

    # --- feedback on the paddle -------------------------------------------
    # Motor commands can't be sent from inside the notification callback (the
    # library raises), so callbacks only queue requests and a worker thread
    # sends them.

    def flash(self, color):
        """Show `color` briefly, then go back to the idle color."""
        self.commands.put(("light", color))
        self.commands.put(("idle_after", time.monotonic() + FLASH_SECONDS))

    def beep_for(self, peak_rate):
        """Higher pitch for harder swings."""
        s = self.swing_cfg
        strength = min(peak_rate / s["hard_swing_rate"], 1.0)
        freq = s["beep_min_hz"] + strength * (s["beep_max_hz"] - s["beep_min_hz"])
        self.commands.put(("beep", int(freq)))

    def feedback_worker(self):
        idle_at = None  # when to switch the light back to the idle color
        while True:
            timeout = None if idle_at is None else max(idle_at - time.monotonic(), 0)
            try:
                kind, value = self.commands.get(timeout=timeout)
            except queue.Empty:
                kind, value = "idle", None
            if not self.motor.connected:
                idle_at = None
                continue
            try:
                if kind == "beep":
                    self.motor.beep(frequency=value, blocking=False)
                elif kind == "light":
                    self.motor.light_color(value, blocking=False)
                elif kind == "idle_after":
                    idle_at = value
                elif kind == "idle":
                    self.motor.light_color(IDLE_LIGHT, blocking=False)
                    idle_at = None
            except (ConnectionError, RuntimeError) as e:
                print(f"Paddle feedback failed: {e}")

    # --- callbacks ----------------------------------------------------------

    def on_notification(self, data):
        """Runs on the Bluetooth thread for every update from the motor."""
        t = time.time()
        for item in le.device_notification_parser(data):
            if not isinstance(item, le.ImuDeviceNotification):
                continue
            self.mqtt.publish(topics.PADDLE_IMU, json.dumps({
                "t": t, "yaw": item.yaw, "pitch": item.pitch, "roll": item.roll,
                "gx": item.gyroscopeX, "gy": item.gyroscopeY, "gz": item.gyroscopeZ,
            }))
            swing = self.detector.update(t, item)
            if swing:
                self.on_swing(swing)

    def on_swing(self, swing):
        self.swing_count += 1
        if not swing["replaces"]:             # the wind-up already beeped
            self.beep_for(swing["peak_rate"])
            self.flash(SWING_LIGHT)
        self.mqtt.publish(topics.PADDLE_SWING, json.dumps(swing), qos=1)
        print(f"SWING #{self.swing_count}: strength {swing['peak_rate']:.0f}, side {swing['side']:+d}, "
              f"{swing['duration_ms']} ms, roll {swing['roll']} pitch {swing['pitch']}"
              + ("  (replaces the wind-up)" if swing["replaces"] else ""))

    def on_game_event(self, topic, payload):
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            return
        color = EVENT_LIGHTS.get(event.get("type"))
        if color is not None:
            self.flash(color)

    # --- connection ---------------------------------------------------------

    def publish_status(self, connected):
        self.mqtt.publish(topics.PADDLE_STATUS, json.dumps({"connected": connected}), qos=1, retain=True)

    def connect_motor(self):
        p = self.cfg["paddle"]
        print(f"Connecting to Double Motor ({p['card_color']} card, serial {p['card_serial']})...")
        self.motor.connect(card_color=getattr(le, f"LEGO_COLOR_{p['card_color']}"),
                           card_serial=p["card_serial"],
                           device_notification_delay=p["notification_delay_ms"])
        if not self.motor.connected:
            return False
        self.motor.set_notification_callback(self.on_notification)
        self.motor.light_color(IDLE_LIGHT, blocking=False)
        self.publish_status(True)
        print("Connected. Swing away! (Ctrl+C to quit)")
        return True

    def run(self):
        self.mqtt.connect(keepalive=3)
        self.publish_status(False)
        self.mqtt.subscribe(topics.GAME_EVENT, self.on_game_event)
        try:
            while True:
                if not self.motor.connected:
                    if self.connect_motor():
                        continue
                    print("Couldn't connect. Is the motor on? Retrying in 2 s...")
                    time.sleep(2)
                    continue
                time.sleep(0.5)
                if not self.motor.connected:
                    print("Paddle disconnected! Trying to reconnect...")
                    self.publish_status(False)
        except KeyboardInterrupt:
            pass
        finally:
            self.publish_status(False)
            if self.motor.connected:
                self.motor.disconnect()
            self.mqtt.disconnect()
            print("Disconnected.")


def main():
    PaddleNode(load_config()).run()


if __name__ == "__main__":
    main()

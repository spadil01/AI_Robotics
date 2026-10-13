"""
Connects the game to paddle_node and vision_node through the local MQTT broker.

Incoming swings, serve tosses and paddle/camera status are put on the game's
inbox queue (MQTT callbacks run on paho's thread, so they must never touch
the scene). Game events (hit, point won/lost, match won) are published back
so the paddle can flash its light.

If the broker isn't running, the game still works with the keyboard.
"""

import json
import math
import sys
import time

from common import topics
from common.config import TABLE_TENNIS_DIR

sys.path.insert(0, str(TABLE_TENNIS_DIR.parent))
from mqttlib import MQTTClient  # noqa: E402


def swing_strength(peak_rate, swing_cfg):
    """Gyro peak -> 0.2 (gentlest detectable swing) .. 1.0 (full power).
    The curve (power 0.7) lifts medium swings so the difference is easy to feel."""
    low, high = swing_cfg["start_threshold"], swing_cfg["hard_swing_rate"]
    k = min(max((peak_rate - low) / (high - low), 0.0), 1.0)
    return 0.2 + 0.8 * k ** 0.7


def paddle_face(swing, handed, swing_cfg):
    """Paddle tilt at the swing -> face -1 (tilted down) .. 0 (straight) .. +1 (tilted up).

    The motor's roll/pitch (tenths of a degree) give how much its z axis points
    up. Forehands and backhands hit with opposite faces, so the swing direction
    decides which face is hitting (measured with tools/tilt_test.py).
    """
    up = math.cos(math.radians(swing["roll"] / 10)) * math.cos(math.radians(swing["pitch"] / 10))
    tilt = math.degrees(math.asin(max(-1.0, min(1.0, -swing["side"] * handed * up))))
    dead, full = swing_cfg["tilt_deadzone_deg"], swing_cfg["tilt_full_deg"]
    amount = min(max((abs(tilt) - dead) / (full - dead), 0.0), 1.0)
    return math.copysign(amount, tilt)


class SensorLink:
    def __init__(self, cfg, inbox):
        self.cfg = cfg
        self.inbox = inbox
        self.client = MQTTClient(broker=cfg["mqtt"]["local_broker"], client_id="game")
        self.connected = False
        # Continuous streams: only the newest value matters, so they're kept here
        # (a single assignment is safe across threads) instead of queued.
        self.preview_jpeg = None     # bytes of the latest camera picture
        self.pose = None             # latest pose message (dict), or None
        self.pose_at = 0.0           # when it arrived (time.time())
        self.tag = None              # latest tag message (dict)
        self.shirt = None            # latest shirt color message (dict)
        self.config_state = {}       # every setting sent so far (all re-sent each time)
        self.shirt_at = 0.0

    def start(self):
        try:
            self.client.connect(timeout=2)
        except OSError as e:
            print(f"No MQTT broker ({e}). Paddle disabled, keyboard only.")
            return False
        self.client.subscribe(topics.PADDLE_SWING, self._on_swing, qos=1)
        self.client.subscribe(topics.PADDLE_STATUS, self._on_status, qos=1)
        self.client.subscribe(topics.VISION_TOSS, self._on_toss, qos=1)
        self.client.subscribe(topics.VISION_POSE, self._on_pose)
        self.client.subscribe(topics.VISION_TAG, self._on_tag)
        self.client.subscribe(topics.VISION_PREVIEW, self._on_preview, raw=True)
        self.client.subscribe(topics.VISION_SHIRT, self._on_shirt)
        self.connected = True
        return True

    def _on_swing(self, topic, payload):
        try:
            data = json.loads(payload)
            data["strength"] = swing_strength(data["peak_rate"], self.cfg["swing"])
            self.inbox.put(("swing", data))
        except (json.JSONDecodeError, KeyError):
            pass

    def _on_status(self, topic, payload):
        try:
            self.inbox.put(("paddle_status", bool(json.loads(payload).get("connected"))))
        except json.JSONDecodeError:
            pass

    def _on_toss(self, topic, payload):
        self.inbox.put(("toss", None))

    def _on_pose(self, topic, payload):
        try:
            self.pose = json.loads(payload)
            self.pose_at = time.time()
        except json.JSONDecodeError:
            pass

    def _on_tag(self, topic, payload):
        try:
            self.tag = json.loads(payload)
        except json.JSONDecodeError:
            pass

    def _on_shirt(self, topic, payload):
        try:
            self.shirt = json.loads(payload)
            self.shirt_at = time.time()
        except json.JSONDecodeError:
            pass

    def _on_preview(self, topic, payload):
        self.preview_jpeg = payload

    def body_seen(self, max_age=1.0):
        pose = self.pose
        return bool(pose and pose.get("visible") and time.time() - self.pose_at < max_age)

    def send_config(self, **settings):
        """Tell the other programs about settings chosen in the game (kept by the broker).
        All settings go out together, so the kept message never loses an older one."""
        self.config_state.update(settings)
        if self.connected:
            self.client.publish(topics.GAME_CONFIG, json.dumps(self.config_state), qos=1, retain=True)

    def send_event(self, kind, strength=0.0):
        if self.connected:
            self.client.publish(topics.GAME_EVENT, json.dumps({"type": kind, "strength": strength}), qos=1)

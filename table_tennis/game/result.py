"""
Publishes the end-of-match score to the class broker.

The message is the player's total returns (hits) in the match, as
"Hits: <float>" (e.g. "Hits: 23.0"), on ME193/Rogers/<player_name>. It runs on a background
thread and retries, so a slow or missing internet connection never freezes
the game. Every score is also appended to a local log as a backup; if sending
fails, resend it later with tools/send_score.py.
"""

import threading
import time

import paho.mqtt.client as mqtt

from common.config import TABLE_TENNIS_DIR

LOG_PATH = TABLE_TENNIS_DIR / "results.log"
ATTEMPTS = 5
STEP_TIMEOUT = 5      # seconds to wait for the broker to accept us, then again for the message


def send(broker, topic, payload):
    """One try at sending. Returns True only once the broker confirms it got the message."""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    connected = threading.Event()
    client.on_connect = lambda *args: connected.set()
    try:
        client.connect(broker, 1883, keepalive=10)
        client.loop_start()
        if not connected.wait(STEP_TIMEOUT):
            return False
        info = client.publish(topic, payload, qos=1, retain=False)
        info.wait_for_publish(timeout=STEP_TIMEOUT)
        return info.is_published()
    except (OSError, RuntimeError, ValueError):
        return False
    finally:
        client.loop_stop()
        client.disconnect()


def log(line):
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def publish_returns(result_cfg, returns, on_done=None):
    """Start publishing in the background. `on_done(ok, message)` is called with
    progress updates (ok=None) and once more when finished (ok=True/False)."""
    topic = result_cfg["topic_prefix"] + result_cfg["player_name"]
    payload = f"Hits: {float(returns)}"
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log(f"{stamp}  {topic}  {payload}  (sending)")

    def worker():
        if result_cfg["player_name"] == "CHANGE_ME":
            done(False, "Score not sent: set player_name in config.toml")
            return
        for attempt in range(1, ATTEMPTS + 1):
            if attempt > 1:
                report(None, f"Score server not answering, trying again ({attempt} of {ATTEMPTS})...")
            if send(result_cfg["broker"], topic, payload):
                done(True, f"Score sent: {payload} -> {topic}")
                return
            time.sleep(2)
        done(False, "Score server is down. Score saved - resend later with\ntools/send_score.py")

    def report(ok, message):
        if on_done:
            on_done(ok, message)

    def done(ok, message):
        log(f"{stamp}  {topic}  {payload}  {'SENT' if ok else 'NOT SENT'}")
        print(message)
        report(ok, message)

    threading.Thread(target=worker, daemon=True).start()

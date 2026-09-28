# Manual, throwaway diagnostic: confirm live_plot's `gate` actually gets
# unblocked by a real "start" MQTT message. Not part of the test suite
# (deliberately not named test_*.py, so pytest won't try to collect/run it
# -- this opens a real plot window and a real network connection).
#
# Run from anywhere: python audio/manual_gate_check.py
# Then, from a second terminal at the repo root:
#   python publish.py "<START_TOPIC from config.py>" "start"

import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mqttlib import MQTTClient  # noqa: E402

import live_plot  # noqa: E402
from config import START_TOPIC  # noqa: E402

started = threading.Event()


def on_start(_topic, payload):
    if payload.strip().lower() == "start":
        started.set()


with MQTTClient() as client:
    client.subscribe(START_TOPIC, on_start)
    print(f"Waiting for 'start' on {START_TOPIC} -- plot should show WAITING FOR START")
    live_plot.main(gate=started)  # no motor

"""Laptop half of the sumo robot: the LEGO parts' messenger.

Talks to the LEGO Double Motor, Color Sensor and Controller over Bluetooth
and to the UNO Q over MQTT. It does no deciding itself:

  Color Sensor + Controller -> "sumo/lego"  -> UNO Q (behavior tree)
  UNO Q -> "sumo/drive" {left, right, ramp} -> Double Motor (+ Single Motor ramp with --ramp)

One reflex lives here instead of on the UNO Q: the instant the light sensor
sees blue, the laptop brakes, without waiting the ~40 ms for the UNO Q to
answer. The UNO Q still decides how to escape; its escape command takes over
as soon as it arrives. If the UNO Q goes quiet, the motors stop.

    python sumo_control/sumo_control.py           # run the robot
    python sumo_control/sumo_control.py --watch   # just print sensor + controller readings
    python sumo_control/sumo_control.py --ramp    # run the robot WITH the lifting ramp
    python sumo_control/sumo_control.py --ramp-test   # raise the ramp, wait 2 s, lower it (tuning)

With --ramp, start with the ramp DOWN: that position is remembered as "down".

Ctrl+C to stop.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time

import legoeducation as le

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mqttlib import MQTTClient  # noqa: E402

import lose_song  # noqa: E402
from config import (  # noqa: E402
    BLE_SCAN_TIMEOUT_S, BROKER, CARD_COLOR, CARD_SERIAL, DRIVE_STALE_S, DRIVE_TOPIC,
    EDGE_COLORS, EDGE_HUE, EDGE_MIN_SAT, REFLEX_MAX_S,
    INVERT_LEFT, INVERT_RIGHT, LEGO_TOPIC, PART_STALE_S, PORT, RAMP_SPEED, RAMP_UP_DEG, SEND_HZ, SENSOR_NOTIFY_MS, STATUS_TOPIC,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def broker_running():
    try:
        socket.create_connection((BROKER, PORT), timeout=0.5).close()
        return True
    except OSError:
        return False


def start_broker():
    """Start mosquitto (the MQTT post office) unless it's already running."""
    if broker_running():
        return None
    exe = shutil.which("mosquitto") or "/opt/homebrew/sbin/mosquitto"
    proc = subprocess.Popen([exe, "-c", os.path.join(HERE, "mosquitto.conf")],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(20):
        if broker_running():
            print("Started the MQTT broker (mosquitto).")
            return proc
        time.sleep(0.1)
    raise RuntimeError("Couldn't start mosquitto -- is it installed? (brew install mosquitto)")


def find_first(device, timeout_s):
    """Scan until a part with our Connection Card shows up, or give up."""
    deadline = time.monotonic() + timeout_s
    while True:
        found = device.search(timeout=0, card_color=CARD_COLOR, card_serial=CARD_SERIAL)
        if found:
            return found[0]
        if time.monotonic() >= deadline:
            return None


def connect(device, name):
    print(f"Looking for the {name} ...")
    found = find_first(device, BLE_SCAN_TIMEOUT_S)
    if not found:
        raise RuntimeError(f"No {name} found -- is it on, nearby, and paired to the "
                           f"card in config.py?")
    device.connect(device=found, device_notification_delay=SENSOR_NOTIFY_MS)
    if not device.connected:
        raise RuntimeError(f"Couldn't connect to the {name}.")
    print(f"  {name} connected.")
    return device


def number(v):
    """LEGO readings start as NaN until the first update arrives."""
    return None if v is None or v != v else int(v)


last_update = {}  # device -> when it last sent a reading
wake = threading.Event()  # new LEGO reading or wheel command -> act now, don't wait


def track_updates(device):
    """Remember when `device` last sent a reading (so a frozen part is noticed)
    and wake the main loop so the reading goes out immediately."""
    def on_reading(data):
        last_update[device] = time.monotonic()
        wake.set()
    device.set_notification_callback(on_reading)
    return device


def fresh(device):
    return time.monotonic() - last_update.get(device, -1e9) < PART_STALE_S


def read_lego(sensor, controller):
    """A quiet light sensor sends color None (the UNO Q stops); a quiet
    controller counts as centered sticks."""
    sensor_ok, controller_ok = fresh(sensor), fresh(controller)
    return {
        "color": number(sensor.sensor.color) if sensor_ok else None,
        "reflection": number(sensor.sensor.reflection) if sensor_ok else None,
        "hue": number(sensor.sensor.hue) if sensor_ok else None,
        "sat": number(sensor.sensor.saturation) if sensor_ok else None,
        "left": (number(controller.sensor.leftPercent) or 0) if controller_ok else 0,
        "right": (number(controller.sensor.rightPercent) or 0) if controller_ok else 0,
    }


def on_edge(reading):
    """Same blue-tape rule as the UNO Q's behavior_tree.on_edge()."""
    if reading["color"] in EDGE_COLORS:
        return True
    hue, sat = reading["hue"], reading["sat"]
    return hue is not None and sat is not None and \
        EDGE_HUE[0] <= hue <= EDGE_HUE[1] and sat >= EDGE_MIN_SAT


def wheels_to_apply(drive, reflex_active):
    """While the blue-tape reflex is on, brake instead of following the UNO Q's
    older 'keep driving' commands. Its escape command (or a stop) gets through."""
    if reflex_active and not drive["behavior"].startswith("escape"):
        return 0, 0
    return drive["left"], drive["right"]


def connect_ramp():
    """Connect the ramp's Single Motor and remember where it is now as 'down'."""
    ramp = connect(le.SingleMotor(), "Single Motor (ramp)")
    ramp.motor_reset_relative_position(position=0)
    # Actively hold the target position instead of coasting, so the ramp gets
    # all the way back down (and stays there while driving).
    ramp.motor_set_end_state(le.MOTOR_END_STATE_HOLD)
    return ramp


def set_ramp(ramp, up, blocking=False):
    ramp.motor_run_to_relative_position(RAMP_UP_DEG if up else 0, speed=RAMP_SPEED, blocking=blocking)


def main():
    watch = "--watch" in sys.argv
    use_ramp = "--ramp" in sys.argv
    if "--ramp-test" in sys.argv:
        ramp = connect_ramp()
        try:
            print(f"Ramp up ({RAMP_UP_DEG} degrees) ...")
            set_ramp(ramp, True, blocking=True)
            time.sleep(2)
            print(f"  now at {ramp.motor.position} degrees (target {RAMP_UP_DEG})")
            print("Ramp down ...")
            set_ramp(ramp, False, blocking=True)
            time.sleep(1)
            print(f"  now at {ramp.motor.position} degrees (target 0)")
        finally:
            ramp.disconnect()
        return
    broker = start_broker()
    devices = []
    try:
        sensor = connect(track_updates(le.ColorSensor()), "Color Sensor")
        devices.append(sensor)
        controller = connect(track_updates(le.Controller()), "Controller")
        devices.append(controller)

        if watch:
            print("Watching readings (Ctrl+C to stop). Blue = color 3.")
            while True:
                print(read_lego(sensor, controller))
                time.sleep(0.2)

        motor = connect(le.DoubleMotor(), "Double Motor")
        devices.append(motor)
        ramp = None
        if use_ramp:
            ramp = connect_ramp()
            devices.append(ramp)
        else:
            print("(No ramp -- add --ramp to use the lifting ramp.)")

        drive = {"left": 0, "right": 0, "behavior": "", "ramp": "down", "time": -1e9}

        def on_drive(topic, payload):
            try:
                cmd = json.loads(payload)
            except ValueError:
                return
            drive.update(left=int(cmd.get("left", 0)), right=int(cmd.get("right", 0)),
                         behavior=str(cmd.get("behavior", "")), ramp=cmd.get("ramp", "down"),
                         time=time.monotonic())
            wake.set()

        def on_status(topic, payload):
            print(f"UNO Q: {payload}")
            if payload.startswith("lost"):
                print("Out of the ring -- playing the losing song. "
                      "Do the wake-up move to go again.")
                lose_song.play()

        with MQTTClient(broker=BROKER, port=PORT, client_id="laptop-sumo-lego") as client:
            client.subscribe(DRIVE_TOPIC, on_drive)
            client.subscribe(STATUS_TOPIC, on_status)
            client.no_delay()
            print("Running. Waiting for the UNO Q ... (Ctrl+C to stop)")
            applied = None
            ramp_applied = False  # starts down
            sent, sent_time = None, 0.0
            reflex_until = -1e9
            while True:
                # Sleep until something happens (or 1/SEND_HZ at most), then act.
                wake.wait(1 / SEND_HZ)
                wake.clear()
                reading = read_lego(sensor, controller)
                now = time.monotonic()
                if reading != sent or now - sent_time >= 1 / SEND_HZ:
                    client.publish(LEGO_TOPIC, json.dumps(reading))
                    sent, sent_time = reading, now

                if on_edge(reading):
                    reflex_until = now + REFLEX_MAX_S
                if drive["behavior"].startswith("escape"):
                    reflex_until = -1e9  # the UNO Q has taken over
                left, right = wheels_to_apply(drive, now < reflex_until)
                ramp_up = drive["ramp"] == "up"
                if time.monotonic() - drive["time"] > DRIVE_STALE_S:
                    left = right = 0  # UNO Q silent -> stop
                    ramp_up = False
                if ramp and ramp_up != ramp_applied:
                    set_ramp(ramp, ramp_up)
                    ramp_applied = ramp_up
                if INVERT_LEFT:
                    left = -left
                if INVERT_RIGHT:
                    right = -right
                if (left, right) != applied:
                    if left or right:
                        motor.movement_move_tank(speed_left=left, speed_right=right, blocking=False)
                    else:
                        motor.movement_stop(blocking=False)
                    applied = (left, right)
    except KeyboardInterrupt:
        pass
    finally:
        for d in reversed(devices):
            if d.connected:
                if isinstance(d, le.DoubleMotor):
                    d.movement_stop()
                elif isinstance(d, le.SingleMotor):
                    print("Lowering the ramp ...")
                    set_ramp(d, False, blocking=True)  # so next start's "down" is right
                print("Disconnecting (may take a few seconds) ...")
                d.disconnect()
        if broker:
            broker.terminate()


if __name__ == "__main__":
    main()

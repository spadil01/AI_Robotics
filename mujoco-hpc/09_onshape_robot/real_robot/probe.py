"""Measure the real LEGO crawler, so the simulation can be made to match it.

Run it on your LAPTOP (not the cluster), from the AI_Robotics folder, with the robot switched on:
    .venv/bin/python mujoco-hpc/09_onshape_robot/real_robot/probe.py

It talks you through 5 short tests (press Enter between them) and saves everything it measured to
probe_log.txt next to this script. Nothing is changed on the robot.
"""
import math
import statistics
import sys
import time
from pathlib import Path

import legoeducation as le

CARD_COLOR, CARD_SERIAL = le.LEGO_COLOR_AZURE, "1130"   # the robot's Connection Card
NOTIFY_MS = 15                                          # ask for sensor updates every 15 ms (the fastest allowed)
LOG = Path(__file__).with_name("probe_log.txt")

log_file = LOG.open("w")


def say(text=""):
    print(text)
    log_file.write(text + "\n")
    log_file.flush()


def wait_for_enter(instructions):
    say(f"\n>>> {instructions}")
    input("    Press Enter when ready... ")


# Count how often sensor updates arrive.
arrivals = []


def on_notification(data):
    arrivals.append(time.perf_counter())


def record(seconds):
    """Read the sensors as fast as they arrive for `seconds`. Returns a list of snapshots."""
    snaps, last = [], None
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        if arrivals and arrivals[-1] != last:
            last = arrivals[-1]
            m, imu = robot.motor, robot.imu_device
            snaps.append(dict(
                t=time.perf_counter(),
                pos=[m[0].position, m[1].position], abs=[m[0].absolutePosition, m[1].absolutePosition],
                speed=[m[0].speed, m[1].speed], power=[m[0].power, m[1].power],
                yaw=imu.yaw, pitch=imu.pitch, roll=imu.roll,
                acc=[imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ],
                gyro=[imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ],
            ))
        time.sleep(0.001)
    return snaps


def mean(values):
    return statistics.fmean(values) if values else float("nan")


# ---- connect -----------------------------------------------------------------------------------------
say(f"Connecting to the Double Motor (card AZURE {CARD_SERIAL})... turn it on if it is off.")
robot = le.DoubleMotor()
robot.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL, device_notification_delay=NOTIFY_MS)
if not robot.connected:
    say("Could not connect. Is the robot on and close by? Is Bluetooth on?")
    sys.exit(1)
robot.set_notification_callback(on_notification)
say("Connected.")

try:
    # ---- 1. sitting still: update rate, and what the motion sensor reads when still -------------------
    wait_for_enter("TEST 1: put the robot flat on the table, the way it crawls, and do not touch it (3 s).")
    s = record(3.0)
    gaps = [b - a for a, b in zip(arrivals[-len(s):], arrivals[-len(s) + 1:])] if len(s) > 1 else []
    say(f"updates: {len(s)} in 3 s = {len(s) / 3:.0f} per second; gap between updates "
        f"avg {1000 * mean(gaps):.0f} ms, worst {1000 * max(gaps, default=float('nan')):.0f} ms")
    say(f"still: yaw {mean([x['yaw'] for x in s]):.1f}  pitch {mean([x['pitch'] for x in s]):.1f}  "
        f"roll {mean([x['roll'] for x in s]):.1f}")
    for k in range(3):
        say(f"still: accelerometer[{'xyz'[k]}] {mean([x['acc'][k] for x in s]):8.1f}   "
            f"gyroscope[{'xyz'[k]}] {mean([x['gyro'][k] for x in s]):8.1f}")

    # ---- 2. arm angle reference: where "0" is in the simulation ----------------------------------------
    wait_for_enter("TEST 2: turn BOTH arms by hand so they point straight FORWARD, flat (parallel to the table),\n"
                   "    exactly like the CAD picture. Then let go.")
    s = record(1.0)
    for k, name in enumerate(("left", "right")):
        say(f"arm reference: {name} arm absolute angle {mean([x['abs'][k] for x in s]):.1f}  "
            f"(position {mean([x['pos'][k] for x in s]):.1f})")

    # ---- 3. arms in the air: top speed and which way 'clockwise' turns -------------------------------
    wait_for_enter("TEST 3: hold the robot UP IN THE AIR so the arms spin freely (about 10 s).")
    for k, name in enumerate(("left", "right")):
        for pct in (50, 100):
            robot.motor_run(direction=le.MOTOR_MOVE_DIRECTION_CLOCKWISE, motor=k, speed=pct)
            time.sleep(0.5)                       # let it get up to speed
            s = record(1.5)
            robot.motor_stop(motor=k)
            dpos = s[-1]["pos"][k] - s[0]["pos"][k]
            dt = s[-1]["t"] - s[0]["t"]
            say(f"air: {name} arm clockwise {pct:3d}% -> position changes {dpos / dt:+7.1f} per second "
                f"(reported speed {mean([x['speed'][k] for x in s]):+.1f}, power {mean([x['power'][k] for x in s]):+.1f})")
            time.sleep(0.3)

    # ---- 4. turning by hand: units of the turning-speed sensor ----------------------------------------
    wait_for_enter("TEST 4: put the robot flat on the table. After Enter, slowly turn it by hand a QUARTER TURN\n"
                   "    to the LEFT (counter-clockwise seen from above), over about 2 seconds, then let go.")
    s = record(3.5)
    for k in range(3):
        area = sum(0.5 * (a["gyro"][k] + b["gyro"][k]) * (b["t"] - a["t"]) for a, b in zip(s, s[1:]))
        say(f"turn: gyroscope[{'xyz'[k]}] added up over time = {area:+10.1f}")
    say(f"turn: yaw went from {s[0]['yaw']:.1f} to {s[-1]['yaw']:.1f} (change {s[-1]['yaw'] - s[0]['yaw']:+.1f})")

    # ---- 5. on the ground: which way is 'forward' ----------------------------------------------------
    wait_for_enter("TEST 5: put the robot on the floor facing FORWARD with some room in front. Both arms will spin\n"
                   "    at half speed for 3 s using LEGO's own 'move forward'.")
    y0 = robot.imu_device.yaw
    robot.movement_move(direction=le.MOVEMENT_DIRECTION_FORWARD, speed=50, blocking=False)
    s = record(3.0)
    robot.movement_stop()
    for k, name in enumerate(("left", "right")):
        dpos = s[-1]["pos"][k] - s[0]["pos"][k]
        say(f"ground: {name} arm position changed {dpos:+.0f} in 3 s")
    say(f"ground: yaw changed {robot.imu_device.yaw - y0:+.1f}")
    answer = input("    Which way did it go? Type f (forward), b (backward), s (spun / stayed): ").strip().lower()
    say(f"ground: you saw '{answer}'")
finally:
    try:
        robot.motor_stop(motor=le.MOTOR_BOTH)
    finally:
        robot.disconnect()
        say(f"\nDone. Results saved in {LOG}")
        log_file.close()

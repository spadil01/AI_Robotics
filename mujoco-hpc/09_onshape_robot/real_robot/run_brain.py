"""Drive the REAL LEGO crawler with the brain trained in the simulation (crawl_brain.onnx).

Put the robot on the floor facing the way you want it to go ("forward" = the way it faces when this starts), switch
it on, then from the AI_Robotics folder run:
    .venv/bin/python mujoco-hpc/09_onshape_robot/real_robot/run_brain.py            # crawls for 10 seconds
    .venv/bin/python mujoco-hpc/09_onshape_robot/real_robot/run_brain.py 20         # crawls for 20 seconds
Press Ctrl-C at any time to stop the motors. They also stop by themselves if the robot stays tipped over.

25 times a second (as in the simulation) it: reads the robot's sensors, turns them into the same numbers the brain
saw in the simulation, asks the brain for two arm speeds, and sends them to the motors. Everything is also written
to run_log.csv next to this script.

The links between the real robot and the simulation were measured with probe.py:
  - arm angles: 0 = arm pointing straight forward, flat (like the CAD); the motor's absolute angle reads ~0 there
  - "forward" crawling: the left motor turns its position DOWN, the right motor turns it UP; in the simulation both
    arm angles go down. So the right arm's angle, speed and command change sign between the two.
  - motor speed reading: about 11.8 degrees per second per unit (88 at 100% = 1030 degrees per second)
  - motion sensor: yaw, pitch and roll in tenths of a degree, gyroscope in degrees per second, turning left = +
    (same as the sim)
"""
import csv
import json
import math
import sys
import time
from pathlib import Path

import legoeducation as le
import numpy as np
import onnxruntime as ort

CARD_COLOR, CARD_SERIAL = le.LEGO_COLOR_AZURE, "1130"
RUN_S = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
SWAP_ARMS = False   # if the robot steers the WRONG way (turns more and more instead of straightening), set True
SPEED_UNIT_DEG_S = 1030 / 88.3     # the motor's speed reading -> degrees per second (probe.py, arms in the air)
TIPPED_DEG = 60                    # tilted more than this (forwards/backwards or sideways)...
TIPPED_FOR_S = 0.3                 # ...for this long = tipped over: stop. (Crawling jolts it; a quick bump is fine.)

HERE = Path(__file__).resolve().parent
INFO = json.loads((HERE / "crawl_brain.json").read_text())
BRAIN = ort.InferenceSession(str(HERE / "crawl_brain.onnx"))
INPUT_NAME = BRAIN.get_inputs()[0].name
DECISION_S = INFO["decision_period_s"]         # how often the brain decided in the simulation (0.04 s)
FULL_SPEED = INFO["max_arm_speed_rad_s"]       # rad/s of the real motor at 100%

# The real robot's motors, in the simulation's arm order. sign = +1 if the real motor's angle grows the same way as
# the simulation's arm angle, -1 if the opposite way (see the notes at the top).
REAL = {"left_wheel": (le.MOTOR_LEFT, +1), "right_wheel": (le.MOTOR_RIGHT, -1)}
if SWAP_ARMS:
  REAL = {"left_wheel": (le.MOTOR_RIGHT, -1), "right_wheel": (le.MOTOR_LEFT, +1)}
ARMS = [REAL[name] for name in INFO["arm_order"]]   # [(motor, sign), (motor, sign)] in the brain's order
assert [i["name"] for i in INFO["inputs"]] == ["heading", "yaw_rate", "arm_phase", "arm_vel", "last_action"], \
  "crawl_brain.json lists different inputs than this script builds: export and run_brain.py must match"


def read_sensors():
  """The robot's latest readings, in simulation units and the simulation's arm order."""
  imu = robot.imu_device
  angles = [sign * math.radians(robot.motor[motor].absolutePosition) for motor, sign in ARMS]
  speeds = [sign * math.radians(robot.motor[motor].speed * SPEED_UNIT_DEG_S) for motor, sign in ARMS]
  heading = math.radians(imu.yaw / 10.0)       # yaw comes in tenths of a degree
  yaw_rate = math.radians(imu.gyroscopeZ)      # degrees per second
  tilt = max(abs(imu.pitch), abs(imu.roll)) / 10.0   # degrees
  return angles, speeds, heading, yaw_rate, tilt


def arm_speeds(action):
  """The brain's two outputs (-1..1) -> arm speeds in the simulation (rad/s): -1 = stopped, +1 = top forward speed."""
  return INFO["action_offset"] + INFO["action_scale"] * np.asarray(action)


def send(action):
  """Send the brain's two arm speeds to the real motors as percentages, both in one Bluetooth message."""
  robot.begin_batch()
  for (motor, sign), speed in zip(ARMS, arm_speeds(action)):
    percent = int(round(sign * 100 * speed / FULL_SPEED))   # + = clockwise (the motor's position goes up)
    direction = le.MOTOR_MOVE_DIRECTION_CLOCKWISE if percent >= 0 else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE
    robot.motor_run(direction=direction, motor=motor, speed=abs(percent))
  robot.end_batch(blocking=False)


print(f"Connecting to the Double Motor (card AZURE {CARD_SERIAL})...")
robot = le.DoubleMotor()
robot.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL, device_notification_delay=15)
if not robot.connected:
  sys.exit("Could not connect. Is the robot on and close by? Is Bluetooth on?")

log_path = HERE / "run_log.csv"
with log_path.open("w", newline="") as f:
  log = csv.writer(f)
  log.writerow(["t_s", "heading_deg", "yaw_rate_deg_s", "tilt_deg", "arm1_angle_deg", "arm2_angle_deg",
                "arm1_speed_rad_s", "arm2_speed_rad_s", "arm1_cmd", "arm2_cmd"])
  try:
    time.sleep(0.2)                                   # let the first readings arrive
    start_heading = read_sensors()[2]
    action = np.full(2, -1.0)                         # start with both arms stopped
    print(f"Crawling for {RUN_S:.0f} s (Ctrl-C stops it).")
    print("  time   heading   turning     arm speed asked -> measured (rad/s), in the brain's arm order")
    start = time.perf_counter()
    next_decision, next_print = start, 0.0
    tilted_since = None
    while (t := time.perf_counter() - start) < RUN_S:
      angles, speeds, heading, yaw_rate, tilt = read_sensors()
      if tilt > TIPPED_DEG:
        tilted_since = t if tilted_since is None else tilted_since
        if t - tilted_since > TIPPED_FOR_S:
          print(f"  {t:4.1f}s  the robot tipped over (tilted {tilt:.0f} degrees): stopping.")
          break
      else:
        tilted_since = None
      h = math.remainder(heading - start_heading, 2 * math.pi)   # 0 = the way it faced at the start

      # The same numbers, in the same order, as the brain saw in the simulation (crawl_task.py, actor_terms).
      obs = np.array(
        [math.sin(h), math.cos(h), yaw_rate]
        + [math.sin(a) for a in angles] + [math.cos(a) for a in angles]
        + speeds + list(action),
        dtype=np.float32,
      )
      action = np.clip(BRAIN.run(None, {INPUT_NAME: obs[None]})[0][0], -1.0, 1.0)
      send(action)

      log.writerow([f"{t:.3f}", f"{math.degrees(h):.1f}", f"{math.degrees(yaw_rate):.1f}", f"{tilt:.0f}",
                    *[f"{math.degrees(a) % 360:.0f}" for a in angles], *[f"{v:.2f}" for v in speeds],
                    *[f"{a:.2f}" for a in action]])
      if t >= next_print:
        next_print += 1.0
        arms = "   ".join(f"{a:+6.1f} -> {v:+6.1f}" for a, v in zip(arm_speeds(action), speeds))
        print(f"  {t:4.1f}s  {math.degrees(h):+6.1f} deg  {math.degrees(yaw_rate):+6.0f} deg/s   {arms}")

      # wait for the next decision time (25 times a second, like the simulation)
      next_decision += DECISION_S
      time.sleep(max(0.0, next_decision - time.perf_counter()))
  except KeyboardInterrupt:
    print("Stopped by you.")
  finally:
    try:
      robot.motor_stop(motor=le.MOTOR_BOTH)
    finally:
      robot.disconnect()
      print(f"Motors stopped. Log saved in {log_path}")

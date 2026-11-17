"""Crawl the LEGO robot (a motor box with two spinning arms) in a straight line, and film it.

No brain yet: both arms just spin at the same steady speed. This checks that the robot CAN crawl before we train it.
It runs once per floor (tile, table, carpet) and makes one video with the three side by side.

Run it from the mujoco-hpc folder:
    python3 runIt.py --gpu --with 09_onshape_robot/onshape 09_onshape_robot/crawl.py

Everything you may want to change is in the first block of constants.
"""
import argparse
import math
import os

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering; must be set before importing mujoco

import imageio
import mujoco
import numpy as np

# Optional: crawl.py [model.xml] [--kd 0.1] [--limit 0.2]   (to try other motor strengths without editing this file)
_parser = argparse.ArgumentParser()
_parser.add_argument("model", nargs="?", default="onshape/scene.xml")
_parser.add_argument("--kd", type=float, default=0.2)
_parser.add_argument("--limit", type=float, default=0.3)
ARGS = _parser.parse_args()
MODEL = ARGS.model
LEFT, RIGHT = "left_wheel", "right_wheel"   # the arm joint names in robot.xml
OUT = "crawl.mp4"

# ---- the motors (guesses for a LEGO motor; change them if you measure the real one) -------------------
REAL_TOP_SPEED = math.radians(1030)   # rad/s at 100%: measured on the real robot, arms spinning in the air (probe.py)
ARM_SPEED = 0.5 * REAL_TOP_SPEED      # rad/s the arms are told to spin: 50%, like probe.py's floor test
SPIN = -1                       # +1 or -1: which way the arms spin. -1 = hands come down in front and pull back
                                # (a swimmer's stroke); +1 swings them up and over and the robot flips.
MOTOR_KD = ARGS.kd                              # how hard a motor pushes toward its speed (N*m per rad/s it is off)
MOTOR_TORQUE_LIMIT = ARGS.limit                   # N*m: the most a motor can push
# On the real floor at 50% the arms kept about 93% of their speed (probe.py). kd 0.2 + limit 0.3 gives 94% in the sim.

# ---- the floors -----------------------------------------------------------------------------------------
# How grippy each floor is for plastic (friction coefficient: 0 = ice, 1 = rubber-like grip).
# Carpet is also soft and bumpy in real life; here it is just "grippier".
SURFACES = {"tile": 0.4, "table": 0.5, "carpet": 0.9}
ROBOT_FRICTION = 0.1   # kept low so the floor's value is the one used (MuJoCo uses the grippier of the two)

ROBOT_WEIGHT_G = 102   # the real robot on a scale. The CAD says about 147 g, so every part is made lighter to match.

SETTLE_S = 1.0    # seconds to drop onto the floor and settle, arms still
CRAWL_S = 10.0    # seconds of crawling
FORWARD_LOCAL = np.array([0.0, -1.0, 0.0])   # the robot's "forward" direction, in the motor box's own frame
# ---------------------------------------------------------------------------------------------------------


def set_weight(spec, kg):
    """Scale every part's mass (and how hard it is to spin, its inertia) by the same amount, so the robot weighs kg."""
    scale = kg / sum(b.mass for b in spec.bodies)
    for b in spec.bodies:
        b.mass *= scale
        b.inertia = b.inertia * scale
        b.fullinertia = b.fullinertia * scale


def build_model():
    spec = mujoco.MjSpec.from_file(MODEL)
    for actuator in list(spec.actuators):   # the export made position motors with no strength limit; replace them
        spec.delete(actuator)
    set_weight(spec, ROBOT_WEIGHT_G / 1000)

    chassis = spec.worldbody.first_body()
    # The arms' axle runs along the box's x. Turn the robot a quarter turn so it crawls along the world's +x.
    chassis.quat = [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]
    if not any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in spec.joints):
        chassis.add_freejoint()             # let the robot move around (the export pins it in place)

    # Plain motors: we decide the push (torque) ourselves each step, like a real speed-controlled motor.
    for name in (LEFT, RIGHT):
        spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                          ctrlrange=[-MOTOR_TORQUE_LIMIT, MOTOR_TORQUE_LIMIT])

    # A white line on the floor along the path, with a tick every 10 cm, to see drift and distance.
    line = dict(type=mujoco.mjtGeom.mjGEOM_BOX, contype=0, conaffinity=0, rgba=[1, 1, 1, 1])
    spec.worldbody.add_geom(size=[0.6, 0.0015, 0.0005], pos=[0.5, 0, 0.0005], **line)
    for i in range(12):
        spec.worldbody.add_geom(size=[0.0015, 0.012, 0.0005], pos=[i * 0.1, 0, 0.0005], **line)

    model = spec.compile()
    return model, model.body(chassis.name).id


def crawl(model, chassis, surface, friction):
    """Crawl on one floor. Returns the video frames and a summary line."""
    floor = model.geom("floor").id
    robot_geoms = model.geom_bodyid != 0
    model.geom_friction[robot_geoms, 0] = ROBOT_FRICTION
    model.geom_friction[floor, 0] = friction
    data = mujoco.MjData(model)

    # Put the robot just above the floor: find its lowest point and lift it from there.
    mujoco.mj_forward(model, data)
    lowest = (data.geom_xpos[robot_geoms, 2] - model.geom_rbound[robot_geoms]).min()
    data.qpos[2] += 0.005 - lowest

    arms = [model.actuator(LEFT).id, model.actuator(RIGHT).id]
    arm_dofs = [model.jnt_dofadr[model.actuator_trnid[a, 0]] for a in arms]

    def motors(speed):
        """Each motor pushes toward `speed`, never harder than its limit."""
        for a, dof in zip(arms, arm_dofs):
            data.ctrl[a] = np.clip(MOTOR_KD * (speed - data.qvel[dof]), -MOTOR_TORQUE_LIMIT, MOTOR_TORQUE_LIMIT)

    def where():
        """Centre of the robot (x, y) and its heading in degrees (0 = straight down +x)."""
        x, y = data.subtree_com[chassis][:2]
        f = data.xmat[chassis].reshape(3, 3) @ FORWARD_LOCAL
        return x, y, math.degrees(math.atan2(f[1], f[0]))

    renderer = mujoco.Renderer(model, 360, 480)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING     # follow the robot
    camera.trackbodyid = chassis
    camera.distance, camera.azimuth, camera.elevation = 0.35, 120, -20

    FPS = 30
    frames = []
    start = None
    arm_speeds = []
    print(f"\n== {surface} (grip {friction}) ==")
    while data.time < SETTLE_S + CRAWL_S:
        while data.time < len(frames) / FPS:
            motors(0.0 if data.time < SETTLE_S else SPIN * ARM_SPEED)
            mujoco.mj_step(model, data)
            if data.time >= SETTLE_S + 1.0:
                arm_speeds.append(abs(data.qvel[arm_dofs]).mean())
        if start is None and data.time >= SETTLE_S:
            start = where()
        if start is not None and len(frames) % FPS == 0:
            x, y, heading = where()
            upright = data.xmat[chassis].reshape(3, 3)[2, 2] > 0.5
            print(f"t={data.time - SETTLE_S:4.1f}s  forward {100 * (x - start[0]):+6.1f} cm   "
                  f"sideways {100 * (y - start[1]):+5.1f} cm   heading {heading - start[2]:+6.1f} deg"
                  + ("" if upright else "   TIPPED OVER"))
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render())
    renderer.close()

    x, y, heading = where()
    fwd, side, turn = 100 * (x - start[0]), 100 * (y - start[1]), heading - start[2]
    summary = (f"{surface:7s} forward {fwd:+6.1f} cm  ({fwd / CRAWL_S:+.1f} cm/s)   "
               f"sideways {side:+5.1f} cm   heading {turn:+6.1f} deg   arms kept {100 * np.mean(arm_speeds) / ARM_SPEED:.0f}% of their speed")
    return frames, summary


model, chassis = build_model()
print("joints:   ", [model.joint(i).name for i in range(model.njnt) if model.joint(i).name])
print("actuators:", [model.actuator(i).name for i in range(model.nu)])
print(f"robot weight in the sim: {1000 * model.body_subtreemass[chassis]:.0f} g;  motors: kd {MOTOR_KD}, limit {MOTOR_TORQUE_LIMIT} N*m")

videos, summaries = [], []
for surface, friction in SURFACES.items():
    frames, summary = crawl(model, chassis, surface, friction)
    videos.append(frames)
    summaries.append(summary)

print("\n== summary (after", CRAWL_S, "s of crawling) ==")
print("\n".join(summaries))

# One video: the floors side by side, left to right in the order of SURFACES.
imageio.mimwrite(OUT, [np.hstack(f) for f in zip(*videos)], fps=30)
print(f"wrote {OUT}: floors left to right = {', '.join(SURFACES)}")

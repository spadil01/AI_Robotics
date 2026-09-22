"""Tracks an AprilTag using a webcam and drives a LEGO Education Double Motor
to keep the tag centered horizontally in frame.

This merges two rig-specific scripts:

- "moving tag": a fixed external webcam watches a tag mounted on the robot
  and drives the robot to re-center that (moving) tag.
- "stationary tag": a phone camera (e.g. via DroidCam) is mounted on the
  robot and moves with it; the tag is fixed in the room, and the robot
  drives itself so the stationary tag stays centered in its own view.

Both rigs assume the robot sits side-on to the tag, with its two
wheels/tracks oriented so that driving both at the same speed and direction
translates it left/right (rather than turning it) - that's what re-centers
the tag horizontally. A PD controller turns the tag's horizontal pixel error
(distance from the frame's center column) into a single translation speed
applied equally to both wheels, clamped to a configurable min/max range.

Rig-specific bits (camera index, which physical direction a positive pixel
error should drive the robot, and the PD/dead-zone/timeout tuning that was
found to work for each rig) are resolved at startup either from command-line
flags or, for whatever wasn't passed on the command line, from a short
interactive wizard - including a brief calibration movement to test/confirm
the wheel direction sign, since getting that backed rather than forwards
just makes the robot drive itself further off-center instead of correcting.

This first version assumes both wheels spin at the same actual speed for the
same commanded speed. It does not yet compensate for the motor speed
mismatch mentioned separately - that's a planned follow-up (using the
Double Motor's built-in gyroscope to trim for heading drift).
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import legoeducation as le

# ---- Configuration shared by both rigs ----
CARD_COLOR = le.LEGO_COLOR_MAGENTA
CARD_SERIAL = "0998"

# AprilTag family/dictionary to detect, and (optionally) a specific tag ID to
# lock onto. Set --tag-id to track whichever tag of this family is seen
# first each frame (the default).
TAG_FAMILY = cv2.aruco.DICT_APRILTAG_36h11

# Hard limits on the commanded speed (-100..100 units). The PD output is
# clamped to this range, so it doubles as the "cruise" speed: once
# kp * error alone would exceed MAX_SPEED, the output saturates there
# instead of climbing further, so raising PD_KP is what makes the robot
# reach full speed at a smaller pixel error instead of only at the extreme
# edge of the frame.
MAX_SPEED = 100.0

# Minimum commanded speed to actually send whenever the robot should be
# moving at all. Below this, the motor's own friction stalls it before it
# can move, so tapering the command all the way to 0 as the tag approaches
# center just makes the robot stall/kick/stall instead of gliding to a
# stop. Tune this to the lowest speed that reliably moves your robot in
# place - too low won't help, too high causes overshoot/jitter right at the
# dead zone edge. Set to 0 to disable this floor entirely.
MIN_EFFECTIVE_SPEED = 0.0

PD_OUTPUT_MAX = MAX_SPEED  # clamp on final commanded speed (-100..100 range)

# Speed/duration of the brief scripted movement used to calibrate
# DIRECTION_SIGN interactively (see run_direction_calibration below).
CALIBRATION_TEST_SPEED = 40.0
CALIBRATION_TEST_DURATION_S = 0.6

# ---- Per-rig defaults ----
# These are the values each original rig-specific script had tuned by hand.
# They're only used to pre-fill the interactive wizard/CLI defaults - any of
# them can still be overridden individually via command-line flags.
MODE_PROFILES = {
    "moving": {
        "camera_index": 0,
        "direction_sign": -1,
        "pd_kp": 0.12,
        "pd_kd": 0.03,
        "dead_zone_px": 10,
        "lost_timeout_s": 0.8,
        "window_title": "AprilTag Tracking (Moving Tag)",
    },
    "stationary": {
        "camera_index": 1,
        "direction_sign": 1,
        "pd_kp": 0.12,
        "pd_kd": 0.03,
        "dead_zone_px": 10,
        "lost_timeout_s": 0.8,
        "window_title": "AprilTag Tracking (Stationary Tag / Phone Camera)",
    },
}


@dataclass
class TrackerConfig:
    camera_index: int
    direction_sign: Optional[int]
    tag_id: Optional[int]
    pd_kp: float
    pd_kd: float
    dead_zone_px: int
    lost_timeout_s: float
    window_title: str


class PDController:
    """A basic PD controller (no integral term) with output clamping.

    Neither rig this script supports ever used a nonzero integral gain, so
    it's dropped entirely here rather than kept around unused.
    """

    def __init__(self, kp, kd, output_limit):
        self.kp = kp
        self.kd = kd
        self.output_limit = output_limit
        self.prev_error = None
        self.prev_time = None

    def reset(self):
        """Clear accumulated state. Call this when the tag is lost so a
        stale derivative doesn't cause a jolt when it reappears."""
        self.prev_error = None
        self.prev_time = None

    def update(self, error, now):
        if self.prev_time is None:
            dt = 0.0
        else:
            dt = now - self.prev_time

        if dt > 0 and self.prev_error is not None:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0.0

        output = self.kp * error + self.kd * derivative
        output = max(-self.output_limit, min(self.output_limit, output))

        self.prev_error = error
        self.prev_time = now
        return output


def make_detector():
    """Build an ArucoDetector tuned to be more permissive than the defaults,
    since the default thresholds are tuned for classic ArUco boards and can
    miss printed AprilTags under normal webcam lighting/size conditions."""
    dictionary = cv2.aruco.getPredefinedDictionary(TAG_FAMILY)
    params = cv2.aruco.DetectorParameters()

    # Try more adaptive-threshold window sizes (default step is coarse),
    # which helps catch tags under uneven lighting.
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 43
    params.adaptiveThreshWinSizeStep = 4

    # Accept smaller candidate quads (default 0.03 requires the tag's
    # perimeter to be at least 3% of the image's perimeter).
    params.minMarkerPerimeterRate = 0.01

    # Refine corners for a more reliable perimeter/bit sampling, at a small
    # performance cost.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_tag(detector, gray_frame, tag_id):
    """Detect AprilTags in a grayscale frame and return the (corners, id) of
    the tag to track, or (None, None) if no matching tag is found.

    corners is a (4, 2) array of the tag's four corner points, in pixels,
    in the order OpenCV's ArUco detector reports them (starting top-left,
    clockwise).
    """
    corners_list, ids, _ = detector.detectMarkers(gray_frame)
    if ids is None or len(ids) == 0:
        return None, None

    if tag_id is not None:
        for tag_corners, detected_id in zip(corners_list, ids.flatten()):
            if detected_id == tag_id:
                return tag_corners.reshape(4, 2), detected_id
        return None, None

    # No specific ID requested: track whichever tag was detected first.
    return corners_list[0].reshape(4, 2), ids.flatten()[0]


def draw_overlay(frame, center_px, tag_corners, tag_center_px, speed, tag_lost):
    h, w = frame.shape[:2]
    cx, _ = center_px

    # Vertical line marking the frame's horizontal center - the column the
    # tag should sit on once centered.
    cv2.line(frame, (cx, 0), (cx, h), (255, 255, 255), 1)

    if tag_corners is not None:
        # Red box outlining the tag's four corners.
        pts = tag_corners.astype(int).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        # Red dot at the tag's center.
        cv2.circle(frame, tag_center_px, 6, (0, 0, 255), -1)

        # Label showing the tag center's pixel coordinates, offset a bit so
        # it doesn't sit directly on top of the dot.
        cv2.putText(
            frame,
            f"({tag_center_px[0]}, {tag_center_px[1]})",
            (tag_center_px[0] + 10, tag_center_px[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    if tag_lost:
        cv2.putText(
            frame,
            "STOPPED (tag not detected)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
    else:
        cv2.putText(
            frame,
            f"Speed: {speed:.0f}%",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )


def prompt_mode():
    while True:
        choice = input(
            "Which rig is this?\n"
            "  [1] Webcam watching a tag mounted on the robot (moving tag)\n"
            "  [2] Camera mounted on the robot, tag fixed in the room (stationary tag)\n"
            "Choice [1/2]: "
        ).strip()
        if choice == "1":
            return "moving"
        if choice == "2":
            return "stationary"
        print("Please enter 1 or 2.")


def prompt_int(prompt_text, default):
    raw = input(f"{prompt_text} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"Not a number, using default ({default}).")
        return default


def prompt_yes_no(prompt_text):
    while True:
        raw = input(f"{prompt_text} [y/n]: ").strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Please answer y or n.")


def build_config(args):
    mode = args.mode
    needs_profile_defaults = any(
        value is None
        for value in (args.camera_index, args.kp, args.kd, args.dead_zone_px, args.lost_timeout_s)
    )
    if mode is None and needs_profile_defaults:
        mode = prompt_mode()
    profile = MODE_PROFILES[mode] if mode else {}

    camera_index = args.camera_index
    if camera_index is None:
        camera_index = prompt_int("Camera index to use", profile.get("camera_index", 0))

    return TrackerConfig(
        camera_index=camera_index,
        direction_sign=args.direction,
        tag_id=args.tag_id,
        pd_kp=args.kp if args.kp is not None else profile.get("pd_kp", 0.15),
        pd_kd=args.kd if args.kd is not None else profile.get("pd_kd", 0.03),
        dead_zone_px=(
            args.dead_zone_px if args.dead_zone_px is not None else profile.get("dead_zone_px", 10)
        ),
        lost_timeout_s=(
            args.lost_timeout_s if args.lost_timeout_s is not None else profile.get("lost_timeout_s", 0.6)
        ),
        window_title=profile.get("window_title", "AprilTag Tracking"),
    )


def run_direction_calibration(doublemotor):
    """Interactively determine DIRECTION_SIGN by briefly driving the robot
    and asking the user which way it actually moved.

    A positive pixel error means the tracked tag is to the right of the
    frame's center column. This runs the same commanded-speed movement the
    main loop would send for that case (assuming direction_sign=1) and asks
    whether the robot moved the way that would bring a tag on its right back
    toward center - if not, the sign needs to be flipped.
    """
    print(
        "\nCalibrating wheel direction: the robot will move briefly, as if "
        "it needed to correct for a tag detected to its RIGHT.\n"
        "Watch which way it moves."
    )
    input("Press Enter to run the test movement...")

    doublemotor.movement_move_tank(
        speed_left=int(CALIBRATION_TEST_SPEED),
        speed_right=int(CALIBRATION_TEST_SPEED),
        blocking=False,
    )
    time.sleep(CALIBRATION_TEST_DURATION_S)
    doublemotor.movement_stop()

    correct = prompt_yes_no(
        "Did the robot move the way that would bring a tag on its RIGHT back toward center?"
    )
    direction_sign = 1 if correct else -1
    print(f"Using direction_sign = {direction_sign}\n")
    return direction_sign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_PROFILES),
        default=None,
        help="Rig preset to use for defaults not otherwise passed on the command line. "
        "Prompted for interactively if omitted and needed.",
    )
    parser.add_argument("--camera-index", type=int, default=None, help="OpenCV camera index to open.")
    parser.add_argument(
        "--direction",
        type=int,
        choices=(-1, 1),
        default=None,
        help="Wheel direction sign. Skips the interactive calibration test if passed.",
    )
    parser.add_argument(
        "--tag-id", type=int, default=None, help="Specific AprilTag ID to track (default: first seen)."
    )
    parser.add_argument("--kp", type=float, default=None, help="PD proportional gain.")
    parser.add_argument("--kd", type=float, default=None, help="PD derivative gain.")
    parser.add_argument("--dead-zone-px", type=int, default=None, help="Dead-zone radius, in pixels.")
    parser.add_argument(
        "--lost-timeout-s",
        type=float,
        default=None,
        help="Seconds to keep coasting after the tag disappears before stopping.",
    )
    args = parser.parse_args()

    config = build_config(args)

    doublemotor = le.DoubleMotor()
    cap = None

    pd = PDController(config.pd_kp, config.pd_kd, PD_OUTPUT_MAX)
    detector = make_detector()

    try:
        print("Connecting to Double Motor...")
        doublemotor.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)
        if not doublemotor.connected:
            print("Error connecting to Double Motor.")
            return

        if config.direction_sign is None:
            config.direction_sign = run_direction_calibration(doublemotor)

        cap = cv2.VideoCapture(config.camera_index)
        if not cap.isOpened():
            print(f"Error: Could not access camera index {config.camera_index}")
            return

        # Timestamp of the last frame in which the tag was seen. None means
        # the tag has never been seen since the tag was last lost.
        last_seen_time = None
        current_speed = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error obtaining frame")
                break

            # Note: deliberately not mirroring the frame (unlike the
            # hand-tracking script). Mirroring flips the image's chirality,
            # not just its rotation, so AprilTag decoding - which only
            # tries the four 90-degree rotations against the dictionary -
            # would find the tag's border but never decode its bit pattern.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            h, w = frame.shape[:2]
            center_px = (w // 2, h // 2)

            tag_corners, _ = detect_tag(detector, gray, config.tag_id)

            now = time.time()
            tag_center_px = None

            if tag_corners is not None:
                last_seen_time = now
                tag_center_px = tuple(np.mean(tag_corners, axis=0).astype(int))

                error = tag_center_px[0] - center_px[0]
                if abs(error) < config.dead_zone_px:
                    current_speed = 0.0
                    pd.reset()
                else:
                    pd_output = pd.update(error, now)
                    # The motor can't actually turn at very low commanded
                    # speeds (it needs enough duty cycle to overcome
                    # friction/stiction). Without this floor, the PD output
                    # tapers smoothly through that dead range as the tag
                    # nears the center, so the robot stalls, then jerks
                    # forward again once noise/derivative pushes the output
                    # back above the real moving threshold - repeatedly.
                    # Clamping the magnitude up to MIN_EFFECTIVE_SPEED skips
                    # straight over the range the motor can't act on anyway.
                    magnitude = max(MIN_EFFECTIVE_SPEED, abs(pd_output))
                    current_speed = config.direction_sign * math.copysign(magnitude, pd_output)
                tag_lost = False
            else:
                tag_lost = last_seen_time is None or (now - last_seen_time) > config.lost_timeout_s
                if tag_lost:
                    current_speed = 0.0
                    pd.reset()
                # else: tag briefly lost, keep coasting at current_speed
                # until lost_timeout_s elapses.

            # Same speed and direction on both wheels: this is a pure
            # translation (no turning), matching the robot's side-on
            # mounting relative to the tag.
            doublemotor.movement_move_tank(
                speed_left=int(current_speed),
                speed_right=int(current_speed),
                blocking=False,
            )

            draw_overlay(frame, center_px, tag_corners, tag_center_px, current_speed, tag_lost)
            cv2.imshow(config.window_title, frame)

            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

    finally:
        print("Cleaning up...")
        if doublemotor.connected:
            try:
                doublemotor.movement_stop()
            except Exception as exc:
                print(f"Error stopping motor: {exc}")
            doublemotor.disconnect()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

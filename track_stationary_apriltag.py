"""Tracks a stationary AprilTag using a phone camera (streamed to this
computer via DroidCam) mounted on a LEGO Education Double Motor, and drives
the Double Motor to keep the tag centered horizontally in frame.

This is the inverse setup of track_moving_apriltag.py: there, a fixed
webcam watched a tag mounted on the robot and drove the robot to re-center
that (moving) tag. Here, the tag is fixed somewhere in the room and the
camera itself - a phone mounted on the robot - moves with the robot, which
drives itself so the stationary tag stays centered in its own view.

The robot is assumed to sit side-on to the tag, with its two wheels/tracks
oriented so that driving both at the same speed and direction translates it
left/right (rather than turning it), which is what re-centers the tag
horizontally in the phone's view. A PID controller turns the tag's
horizontal pixel error (distance from the frame's center column) into a
single translation speed applied equally to both wheels, clamped to a
configurable min/max speed range.

This first version assumes both wheels spin at the same actual speed for
the same commanded speed. It does not yet compensate for the motor speed
mismatch mentioned separately - that's a planned follow-up (using the
Double Motor's built-in gyroscope to trim for heading drift).
"""

import math
import time

import cv2
import numpy as np

import legoeducation as le

# ---- Configuration ----
# Index of the DroidCam virtual camera as seen by OpenCV, not the Mac's
# built-in webcam. This varies by machine/setup, so confirm it rather than
# assuming - e.g. run:
#   for i in range(6): print(i, cv2.VideoCapture(i).isOpened())
# and check the preview window shows the phone's feed, not the webcam's.
CAMERA_INDEX = 1

CARD_COLOR = le.LEGO_COLOR_MAGENTA
CARD_SERIAL = "0998"

# AprilTag family/dictionary to detect, and (optionally) a specific tag ID to
# lock onto. Set TAG_ID to None to track whichever tag of this family is seen
# first each frame.
TAG_FAMILY = cv2.aruco.DICT_APRILTAG_36h11
TAG_ID = None

# Flips which physical direction (left/right wheel command sign) corresponds
# to a positive pixel error (tag right of center). The sign that was correct
# for track_moving_apriltag.py (external camera watching a tag on the robot)
# is not guaranteed to still be correct now that the camera itself is
# mounted on the robot and moves with it - re-verify by testing.
DIRECTION_SIGN = 1

# ---- Speed profile (global) ----
# Hard limits on the commanded speed (-100..100 units). The PID output below
# is clamped to this range, so it doubles as the "cruise" speed: once
# kp * error alone would exceed MAX_SPEED, the output saturates there
# instead of climbing further, so raising PID_KP is what makes the robot
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

# Pixel radius around the frame's center column where the robot holds still,
# so small detection jitter doesn't cause constant tiny movements.
DEAD_ZONE_PX = 9

# ---- PID controller (global) ----
# Tuned against horizontal pixel error (tag center x - frame center x),
# active across the whole frame outside DEAD_ZONE_PX. Raise PID_KP so the
# output saturates at MAX_SPEED at a smaller error (i.e. the robot reaches
# full speed without needing to be near the frame's edge) instead of only
# ramping down smoothly close to the center; PID_KD damps overshoot as the
# error approaches 0, and PID_KI corrects any steady-state error the P term
# alone leaves behind. Starting from the same values as
# track_moving_apriltag.py - DroidCam's WiFi link can introduce occasional
# dropped/late frames beyond what a wired webcam has, so PID_KD in
# particular may need to come down a bit if that causes jitter; tune live.
PID_KP = 0.12
PID_KI = 0.0
PID_KD = 0.03
PID_INTEGRAL_LIMIT = 500.0  # anti-windup clamp on the accumulated integral term
PID_OUTPUT_MAX = MAX_SPEED  # clamp on final commanded speed (-100..100 range)

# How long to keep holding the last command after the tag disappears before
# forcing a stop, in seconds. A bit more forgiving than a wired webcam setup
# would need, since a WiFi camera feed can briefly stall/drop frames.
LOST_TAG_TIMEOUT_S = 0.8


class PIDController:
    """A basic PID controller with anti-windup and output clamping."""

    def __init__(self, kp, ki, kd, integral_limit, output_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.integral = 0.0
        self.prev_error = None
        self.prev_time = None

    def reset(self):
        """Clear accumulated state. Call this when the tag is lost so a
        stale integral/derivative doesn't cause a jolt when it reappears."""
        self.integral = 0.0
        self.prev_error = None
        self.prev_time = None

    def update(self, error, now):
        if self.prev_time is None:
            dt = 0.0
        else:
            dt = now - self.prev_time

        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))

        if dt > 0 and self.prev_error is not None:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0.0

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
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


def detect_tag(detector, gray_frame):
    """Detect AprilTags in a grayscale frame and return the (corners, id) of
    the tag to track, or (None, None) if no matching tag is found.

    corners is a (4, 2) array of the tag's four corner points, in pixels,
    in the order OpenCV's ArUco detector reports them (starting top-left,
    clockwise).
    """
    corners_list, ids, _ = detector.detectMarkers(gray_frame)
    if ids is None or len(ids) == 0:
        return None, None

    if TAG_ID is not None:
        for tag_corners, tag_id in zip(corners_list, ids.flatten()):
            if tag_id == TAG_ID:
                return tag_corners.reshape(4, 2), tag_id
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


def main():
    doublemotor = le.DoubleMotor()
    cap = None

    pid = PIDController(PID_KP, PID_KI, PID_KD, PID_INTEGRAL_LIMIT, PID_OUTPUT_MAX)

    detector = make_detector()

    try:
        print("Connecting to Double Motor...")
        doublemotor.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)
        if not doublemotor.connected:
            print("Error connecting to Double Motor.")
            return

        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print("Error: Could not access the phone camera (DroidCam). "
                  "Check CAMERA_INDEX and that DroidCam is connected.")
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

            tag_corners, _ = detect_tag(detector, gray)

            now = time.time()
            tag_center_px = None

            if tag_corners is not None:
                last_seen_time = now
                tag_center_px = tuple(np.mean(tag_corners, axis=0).astype(int))

                error = tag_center_px[0] - center_px[0]
                if abs(error) < DEAD_ZONE_PX:
                    current_speed = 0.0
                    pid.reset()
                else:
                    pid_output = pid.update(error, now)
                    # The motor can't actually turn at very low commanded
                    # speeds (it needs enough duty cycle to overcome
                    # friction/stiction). Without this floor, the PID output
                    # tapers smoothly through that dead range as the tag
                    # nears the center, so the robot stalls, then jerks
                    # forward again once noise/derivative pushes the output
                    # back above the real moving threshold - repeatedly.
                    # Clamping the magnitude up to MIN_EFFECTIVE_SPEED skips
                    # straight over the range the motor can't act on anyway.
                    magnitude = max(MIN_EFFECTIVE_SPEED, abs(pid_output))
                    current_speed = DIRECTION_SIGN * math.copysign(magnitude, pid_output)
                tag_lost = False
            else:
                tag_lost = last_seen_time is None or (now - last_seen_time) > LOST_TAG_TIMEOUT_S
                if tag_lost:
                    current_speed = 0.0
                    pid.reset()
                # else: tag briefly lost, keep coasting at current_speed
                # until LOST_TAG_TIMEOUT_S elapses.

            # Same speed and direction on both wheels: this is a pure
            # translation (no turning), matching the robot's side-on
            # mounting relative to the tag.
            doublemotor.movement_move_tank(
                speed_left=int(current_speed),
                speed_right=int(current_speed),
                blocking=False,
            )

            draw_overlay(frame, center_px, tag_corners, tag_center_px, current_speed, tag_lost)
            cv2.imshow("Stationary AprilTag Tracking (Phone Camera)", frame)

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

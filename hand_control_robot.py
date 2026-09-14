"""Controls a LEGO Education Double Motor and Single Motor with your hands via webcam
using mediapipe and opencv libraries.

Uses MediaPipe to track hands in the webcam feed and treats a single
detected hand as a virtual joystick: its position relative to the center of
the frame sets forward/backward and turning speed for the Double Motor. Showing 
no hands stops the Double Motor. Separately, showing two hands triggers a 
second, independent Single Motor to spin for a fixed duration (and, since the 
joystick needs exactly one hand, also stops the Double Motor for as long as 
the Single Motor is running).
"""

import math
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import legoeducation as le

# ---- Configuration ----
MODEL_PATH = "hand_landmarker.task"
CAMERA_INDEX = 0

CARD_COLOR = le.LEGO_COLOR_PURPLE
CARD_SERIAL = "6235"

# Connection Card for the separate Single Motor, actuated by showing two
# hands. Update these to match the Connection Card actually attached to it.
SINGLE_MOTOR_CARD_COLOR = le.LEGO_COLOR_PURPLE
SINGLE_MOTOR_CARD_SERIAL = "6235"

SINGLE_MOTOR_SPEED = 30        # speed (%) the Single Motor spins at when actuated
SINGLE_MOTOR_RUN_TIME_S = 2.0   # how long the Single Motor spins for, in seconds
SINGLE_MOTOR_DIRECTION = le.MOTOR_MOVE_DIRECTION_CLOCKWISE  # spin direction

# Joystick zone sizes, as a fraction of the frame's shorter dimension.
DEAD_ZONE_FRAC = 0.07   # radius around the frame center that maps to 0 speed
MAX_RANGE_FRAC = 0.45    # radius from the center that maps to full (100%) speed
SMOOTHING = 0.5         # exponential smoothing factor: 0 = no smoothing, 1 = instant

AXIS_SNAP_DEG = 30      # buffer (in degrees) around straight up/down/left/right
                        # that snaps to a pure forward/backward/turn command,
                        # so a hand that's meant to be held straight up but is
                        # slightly off-axis doesn't also turn the robot
MAX_TURN_SPEED = 30     # caps the turn contribution to each wheel's speed, since
                        # an unclamped +/-100 turn spins the robot very fast
MAX_THROTTLE_SPEED = 100 # caps the forward/backward speed, independent of turning

# Wrist + four knuckle (MCP) landmarks, averaged to approximate the palm center.
PALM_LANDMARK_INDICES = [0, 5, 9, 13, 17]

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode


def palm_center(hand_landmarks):
    """Average the wrist + knuckle landmarks into one (x, y) point.

    MediaPipe reports 21 landmarks per hand, each normalized to [0, 1] across
    the frame width/height. Averaging several points near the base of the
    hand is steadier frame-to-frame than tracking a single landmark (e.g.
    the wrist alone), which can wobble as fingers move.
    """
    xs = [hand_landmarks[i].x for i in PALM_LANDMARK_INDICES]
    ys = [hand_landmarks[i].y for i in PALM_LANDMARK_INDICES]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def joystick_to_speeds(dx, dy, dead_zone_px, max_range_px):
    """Convert a hand's pixel offset from the frame center into tank-drive speeds for
    Double Motor.

    dx > 0 means the hand is right of center (turn right), dy > 0 means the
    hand is above center (drive forward).
    """
    # Straight-line distance from the center. Inside the dead zone, both
    # motors stay at 0 so resting your hand near the middle doesn't cause
    # unwanted drift.
    r = math.hypot(dx, dy)
    if r < dead_zone_px:
        return 0.0, 0.0

    # How far past the dead zone we are, as a 0..1 fraction of the way to
    # max_range_px (where speed reaches 100%). Clamped so anything beyond
    # max_range_px still maps to exactly 100%, not more.
    scale = (r - dead_zone_px) / (max_range_px - dead_zone_px)
    scale = max(0.0, min(1.0, scale))

    # (ux, uy) is the direction from the center to the hand, as a unit
    # vector. Splitting it out lets us scale x/y independently below.
    ux, uy = dx / r, dy / r
    turn = ux * scale * 100
    throttle = uy * scale * 100

    # Snap to a pure axis when the hand is within AXIS_SNAP_DEG of straight
    # up/down/left/right. A hand aimed at "straight forward" is rarely
    # pixel-perfect, so without this buffer, small off-axis wobble bleeds
    # into the other component and makes the robot turn while driving
    # straight (or drift while turning in place).
    angle = math.degrees(math.atan2(ux, uy))  # 0=up, 90=right, -90=left, +-180=down
    if abs(angle) <= AXIS_SNAP_DEG or abs(angle) >= 180 - AXIS_SNAP_DEG:
        turn = 0.0
    elif abs(abs(angle) - 90) <= AXIS_SNAP_DEG:
        throttle = 0.0

    # Cap how much turning can contribute to each wheel, since an
    # uncapped +/-100 turn spins the robot in place very fast.
    turn = max(-MAX_TURN_SPEED, min(MAX_TURN_SPEED, turn))
    # Cap forward/backward speed too, independent of the turn cap above.
    throttle = max(-MAX_THROTTLE_SPEED, min(MAX_THROTTLE_SPEED, throttle))

    # Standard differential-drive mixing: driving straight uses throttle
    # only (turn cancels out), turning in place uses turn only (throttle is
    # 0), and anything in between blends the two.
    left = max(-100.0, min(100.0, throttle + turn))
    right = max(-100.0, min(100.0, throttle - turn))
    return left, right


def draw_overlay(
    frame,
    center_px,
    dead_zone_px,
    max_range_px,
    joystick_point_px,
    left_speed,
    right_speed,
    no_hands_detected,
    single_motor_running,
):
    cx, cy = center_px
    # Green ring = max-range radius (distance at which speed reaches 100%).
    cv2.circle(frame, (cx, cy), int(max_range_px), (0, 255, 0), 1)
    # Yellow ring = dead zone (hand inside this radius produces 0 speed).
    cv2.circle(frame, (cx, cy), int(dead_zone_px), (0, 255, 255), 2)
    # Crosshair marks the exact joystick origin (the frame's center point).
    cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)

    if joystick_point_px is not None:
        # Line + dot show where the tracked hand is relative to the origin.
        cv2.line(frame, (cx, cy), joystick_point_px, (0, 200, 0), 2)
        cv2.circle(frame, joystick_point_px, 8, (0, 0, 255), -1)

    if no_hands_detected:
        cv2.putText(
            frame,
            "STOPPED (no hands detected)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
    else:
        cv2.putText(
            frame,
            f"L: {left_speed:.0f}%  R: {right_speed:.0f}%",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    if single_motor_running:
        cv2.putText(
            frame,
            "Single Motor: ON (2 hands)",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )


def main():
    doublemotor = le.DoubleMotor()
    singlemotor = le.SingleMotor()
    cap = None
    landmarker = None

    # Everything after this point runs inside try/finally so that no matter
    # how the loop below ends (user quits, an exception, Ctrl+C), the
    # `finally` block always stops the motor and releases the camera. This
    # keeps the robot from being left running if the script crashes.
    try:
        print("Connecting to Double Motor...")
        doublemotor.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)
        if not doublemotor.connected:
            print("Error connecting to Double Motor.")
            return

        print("Connecting to Single Motor...")
        singlemotor.connect(card_color=SINGLE_MOTOR_CARD_COLOR, card_serial=SINGLE_MOTOR_CARD_SERIAL)
        if not singlemotor.connected:
            print("Error connecting to Single Motor.")
            return

        options = HandLandmarkerOptions(
            # Force CPU inference: MediaPipe's default GPU (Metal) delegate
            # crashes on some Macs with "Service is unavailable" during
            # TensorsToDetectionsCalculator::Open().
            base_options=BaseOptions(model_asset_path=MODEL_PATH, delegate=BaseOptions.Delegate.CPU),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
        )
        landmarker = HandLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print("Error: Could not access the webcam")
            return

        start_time = time.time()
        # Current motor speeds, filtered over time to smooth out jitter.
        # These persist across loop iterations (unlike target_left/right
        # below, which are recomputed fresh each frame).
        smoothed_left = 0.0
        smoothed_right = 0.0
        # Timestamp (time.time()) until which the Single Motor is still
        # spinning from a past two-hands trigger. 0 means it's idle. While
        # the current run is still active, seeing two hands again is
        # ignored so holding them up doesn't keep restarting the timer.
        single_motor_active_until = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error obtaining frame")
                break

            # Mirror the frame so it feels natural to the user.
            frame = cv2.flip(frame, 1)

            # MediaPipe expects RGB input, but OpenCV captures frames as BGR.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # VIDEO mode requires a monotonically increasing timestamp per
            # frame; using elapsed time since start works well for this.
            timestamp_ms = int((time.time() - start_time) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            h, w = frame.shape[:2]
            # The joystick's origin is the middle of the video frame.
            center_px = (w // 2, h // 2)
            # Zone radii are computed per-frame (not once) in case the
            # camera resolution or window size ever changes.
            dead_zone_px = DEAD_ZONE_FRAC * min(w, h)
            max_range_px = MAX_RANGE_FRAC * min(w, h)

            num_hands = len(result.hand_landmarks) if result.hand_landmarks else 0
            joystick_point_px = None
            no_hands_detected = num_hands == 0

            # Two-hands check happens before the joystick logic below so
            # that triggering (or still running from a previous trigger)
            # can override the Double Motor and force it to stay stopped
            # for the whole duration of the Single Motor's spin, instead of
            # the two motors running at the same time.
            now = time.time()
            single_motor_running = now < single_motor_active_until
            if not single_motor_running and num_hands == 2:
                singlemotor.motor_run_for_time(
                    int(SINGLE_MOTOR_RUN_TIME_S * 1000),
                    direction=SINGLE_MOTOR_DIRECTION,
                    speed=SINGLE_MOTOR_SPEED,
                    blocking=False,
                )
                single_motor_active_until = now + SINGLE_MOTOR_RUN_TIME_S
                single_motor_running = True

            if single_motor_running:
                # Single Motor has exclusive control right now: force the
                # Double Motor to a full stop, ignoring whatever the
                # joystick hand is doing.
                smoothed_left = 0.0
                smoothed_right = 0.0
            elif num_hands == 1:
                # Exactly one hand: use it as the joystick. Convert its
                # normalized (0..1) palm position to pixel coordinates, then
                # to an offset from the frame center, and turn that into
                # target motor speeds.
                px, py = palm_center(result.hand_landmarks[0])
                joystick_point_px = (int(px * w), int(py * h))
                dx = joystick_point_px[0] - center_px[0]
                # Flip the y-axis: image coordinates increase downward, but
                # we want "hand above center" to mean positive/forward.
                dy = center_px[1] - joystick_point_px[1]
                target_left, target_right = joystick_to_speeds(dx, dy, dead_zone_px, max_range_px)
                # Exponential smoothing: nudge the current speed a fraction
                # of the way toward the new target each frame, rather than
                # jumping straight to it. Reduces jitter from noisy landmark
                # detection.
                smoothed_left += (target_left - smoothed_left) * SMOOTHING
                smoothed_right += (target_right - smoothed_right) * SMOOTHING
            else:
                # No hands (the stop gesture), or two hands where the
                # joystick isn't well-defined: snap to a full stop
                # immediately instead of smoothing down.
                smoothed_left = 0.0
                smoothed_right = 0.0

            # Sent every frame since this is a continuous real-time control
            # loop; blocking=False so the call returns immediately instead
            # of waiting for the motor to finish, which would stall the
            # video feed.
            doublemotor.movement_move_tank(
                speed_left=int(smoothed_left),
                speed_right=int(smoothed_right),
                blocking=False,
            )

            draw_overlay(
                frame,
                center_px,
                dead_zone_px,
                max_range_px,
                joystick_point_px,
                smoothed_left,
                smoothed_right,
                no_hands_detected,
                single_motor_running,
            )
            cv2.imshow("Hand-Controlled Lego Motor", frame)

            # Esc (27) or 'q' exits the loop and falls through to cleanup.
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

    finally:
        # Runs on a clean exit, an exception, or Ctrl+C alike, so the motor
        # never keeps spinning after the script stops.
        print("Cleaning up...")
        if doublemotor.connected:
            try:
                doublemotor.movement_stop()
            except Exception as exc:
                print(f"Error stopping motor: {exc}")
            doublemotor.disconnect()
        if singlemotor.connected:
            try:
                singlemotor.motor_stop()
            except Exception as exc:
                print(f"Error stopping single motor: {exc}")
            singlemotor.disconnect()
        if cap is not None:
            cap.release()
        if landmarker is not None:
            landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

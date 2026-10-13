"""
Vision node: the game's "eyes".

Reads the webcam, finds the paddle's AprilTag and the player's body
(MediaPipe pose), and publishes both to the local MQTT broker. A preview
window shows what it sees, with fps and tag size so camera placement and
tag size can be checked.

All x coordinates are mirrored (x -> 1 - x) before publishing, so moving
the paddle to your right moves things to the right on screen, like a mirror.

It also finds the player's shirt color: the chest area (just below the
shoulders) is split into a few color groups with k-means clustering, and the
biggest group is the shirt. The game paints the player's character with it.

It also publishes a small JPEG of the (mirrored) camera view with the
skeleton drawn on it, which the game shows on its calibration and menu
screens. With --headless there's no preview window of its own, so the game
is the only window (run_all.py starts it this way).

Run from the repo root:  .venv/bin/python table_tennis/vision_node.py [--headless]
Press q in the preview window (or Ctrl+C) to quit.
"""

import argparse
import json
import sys
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from common import topics
from common.config import TABLE_TENNIS_DIR, load_config

sys.path.insert(0, str(TABLE_TENNIS_DIR.parent))
from mqttlib import MQTTClient  # noqa: E402

TAG_FAMILY = cv2.aruco.DICT_APRILTAG_36h11

# MediaPipe pose landmark indices we care about (person's own left/right).
LANDMARKS = {
    "nose": 0,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16,
    "l_hip": 23, "r_hip": 24,
}
SKELETON = [("l_shoulder", "r_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
            ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"), ("l_shoulder", "l_hip"),
            ("r_shoulder", "r_hip"), ("l_hip", "r_hip")]


def make_tag_detector():
    """Same permissive settings as apriltag_centering/, which work for printed tags."""
    dictionary = cv2.aruco.getPredefinedDictionary(TAG_FAMILY)
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 43
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.01
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, params)


def make_pose_landmarker(model_path):
    options = vision.PoseLandmarkerOptions(
        # CPU delegate: MediaPipe's GPU (Metal) delegate crashes on some Macs.
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path),
                                           delegate=mp_python.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def find_tag(detector, gray, tag_ids):
    """Return (id, corners as a (4, 2) array) of the first paddle tag seen, or
    (None, None), plus the IDs of any other tags of this family in view (shown
    in the preview so a tag with an unexpected ID is easy to spot)."""
    corners_list, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None, None, []
    found_id, found_corners, others = None, None, []
    for corners, tag_id in zip(corners_list, ids.flatten()):
        if found_id is None and int(tag_id) in tag_ids:
            found_id, found_corners = int(tag_id), corners.reshape(4, 2)
        else:
            others.append(int(tag_id))
    return found_id, found_corners, others


class TossDetector:
    """Serve toss = raising the free hand (the one without the paddle) above its
    shoulder for a few frames in a row. The hand has to come back down below
    the shoulder before another toss can count, so holding it up only tosses once.

    MediaPipe labels landmarks by the person's own left/right, so this works the
    same whether or not the preview is mirrored.
    """

    def __init__(self, cfg, handedness):
        self.handedness = handedness
        side = "l" if handedness == "right" else "r"
        self.wrist, self.shoulder = f"{side}_wrist", f"{side}_shoulder"
        self.margin = cfg["toss_margin"]
        self.frames_needed = cfg["toss_frames"]
        self.cooldown = cfg["toss_cooldown_ms"] / 1000
        self.min_visibility = cfg["min_visibility"]
        self.count = 0
        self.armed = True
        self.cooldown_until = 0.0
        self.raised = False

    def update(self, t, points):
        """Feed one frame's landmarks. Returns True on the frame a toss happens."""
        wrist, shoulder = points[self.wrist], points[self.shoulder]
        if min(wrist.visibility, shoulder.visibility) < self.min_visibility:
            self.count, self.raised = 0, False
            return False
        self.raised = wrist.y < shoulder.y - self.margin    # image y grows downward
        if not self.raised:
            self.count = 0
            if wrist.y > shoulder.y:
                self.armed = True
            return False
        self.count += 1
        if self.armed and self.count >= self.frames_needed and t >= self.cooldown_until:
            self.armed = False
            self.cooldown_until = t + self.cooldown
            return True
        return False


def shirt_color(frame, points, cfg):
    """Main color of the chest area, as ((r, g, b) 0..1, share of pixels, box), or None.

    The box starts a little below the shoulders and is narrower than them, so
    it's mostly shirt. k-means splits its pixels into a few color groups (in Lab
    color space, where distances match how different colors look to people);
    the biggest group is the shirt, even if an arm or the paddle is in the way.
    """
    ls, rs = points["l_shoulder"], points["r_shoulder"]
    if min(ls.visibility, rs.visibility) < cfg["min_visibility"]:
        return None
    h, w = frame.shape[:2]
    x0, x1 = sorted((ls.x * w, rs.x * w))
    span = x1 - x0
    if span < 40:
        return None                                   # too far away / side-on to measure
    top = max(ls.y, rs.y) * h + 0.15 * span
    box = (int(max(x0 + 0.2 * span, 0)), int(max(top, 0)), int(min(x1 - 0.2 * span, w)), int(min(top + 0.6 * span, h)))
    patch = frame[box[1]:box[3], box[0]:box[2]]
    if patch.shape[0] < 10 or patch.shape[1] < 10:
        return None
    step = max(1, int((patch.shape[0] * patch.shape[1] / 3000) ** 0.5))     # about 3000 pixels is plenty
    lab = cv2.cvtColor(np.ascontiguousarray(patch[::step, ::step]), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(lab.astype(np.float32), cfg["shirt_clusters"], None, criteria, 3,
                                    cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten())
    biggest = int(counts.argmax())
    b, g, r = cv2.cvtColor(centers[biggest].reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_LAB2BGR)[0, 0]
    return (float(r) / 255, float(g) / 255, float(b) / 255), float(counts[biggest] / counts.sum()), box


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="no preview window (the game shows the view)")
    args = parser.parse_args()

    cfg = load_config()
    vcfg = cfg["vision"]
    tag_ids = set(vcfg["tag_ids"])

    cap = cv2.VideoCapture(vcfg["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, vcfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, vcfg["height"])
    if not cap.isOpened():
        print(f"Could not open camera {vcfg['camera_index']}.")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera running at {width}x{height}")
    print("ready")

    detector = make_tag_detector()
    landmarker = make_pose_landmarker(TABLE_TENNIS_DIR / vcfg["pose_model"])
    mqtt = MQTTClient(broker=cfg["mqtt"]["local_broker"], client_id="vision_node")
    mqtt.connect()

    # The game publishes the handedness found during calibration; the toss
    # detector watches the other hand. (Read on the main loop, set by MQTT's thread.)
    settings = {"handedness": cfg["game"]["handedness"]}

    def on_game_config(topic, payload):
        try:
            settings.update(json.loads(payload))
        except json.JSONDecodeError:
            pass

    mqtt.subscribe(topics.GAME_CONFIG, on_game_config, qos=1)
    preview_every = 1 / vcfg["preview_fps"]
    last_preview = 0.0
    last_shirt = 0.0
    shirt_box = None              # (box, BGR color) drawn on the preview

    toss = TossDetector(vcfg, settings["handedness"])
    toss_flash_until = 0.0
    frame_times = deque(maxlen=30)
    tag_seen = deque(maxlen=60)  # last ~2 s of frames: was the tag found?
    start = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            t = time.time()  # capture time, shared clock with every other process
            if not ok:
                print("Camera frame failed.")
                break
            frame_times.append(time.monotonic())

            # --- AprilTag ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tag_id, corners, other_ids = find_tag(detector, gray, tag_ids)
            tag_seen.append(tag_id is not None)
            tag_msg = {"t": t, "visible": tag_id is not None}
            tag_px = 0.0
            if tag_id is not None:
                cx, cy = corners.mean(axis=0)
                tag_px = float(cv2.norm(corners[0] - corners[1]))
                tag_msg.update(id=tag_id, cx=float(1 - cx / width), cy=float(cy / height), size_px=tag_px)
                cv2.polylines(frame, [corners.astype(int)], True, (0, 200, 255), 3)
            mqtt.publish(topics.VISION_TAG, json.dumps(tag_msg))

            if toss.handedness != settings["handedness"]:
                toss = TossDetector(vcfg, settings["handedness"])
                print(f"Now watching the {'left' if toss.wrist[0] == 'l' else 'right'} hand for tosses")

            # --- Pose ---
            rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(rgb, int((time.monotonic() - start) * 1000))
            pose_msg = {"t": t, "visible": bool(result.pose_landmarks)}
            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                points = {name: lms[i] for name, i in LANDMARKS.items()}
                pose_msg["landmarks"] = {name: [round(1 - p.x, 4), round(p.y, 4), round(p.visibility, 2)]
                                         for name, p in points.items()}
                if toss.update(t, points):
                    mqtt.publish(topics.VISION_TOSS, json.dumps({"t": t, "hand": toss.wrist[0]}), qos=1)
                    toss_flash_until = t + 0.6
                    print("TOSS!")
                if t - last_shirt >= vcfg["shirt_every_s"]:
                    last_shirt = t
                    found = shirt_color(frame, points, vcfg)     # measured before anything is drawn on the frame
                    shirt_box = None
                    if found:
                        color, share, box = found
                        mqtt.publish(topics.VISION_SHIRT, json.dumps({"t": t, "rgb": color, "share": share}))
                        shirt_box = (box, tuple(int(c * 255) for c in reversed(color)))
                if shirt_box and settings.get("show_shirt_box", True):     # hidden during matches
                    (bx0, by0, bx1, by1), bgr = shirt_box
                    cv2.rectangle(frame, (bx0, by0), (bx1, by1), bgr, -1)                 # filled with the found color
                    cv2.rectangle(frame, (bx0, by0), (bx1, by1), (255, 255, 255), 3)
                for a, b in SKELETON:
                    pa, pb = points[a], points[b]
                    cv2.line(frame, (int(pa.x * width), int(pa.y * height)),
                             (int(pb.x * width), int(pb.y * height)), (0, 255, 0), 3)
            mqtt.publish(topics.VISION_POSE, json.dumps(pose_msg))

            # --- Preview (mirrored, text drawn after flipping so it reads normally) ---
            view = cv2.flip(frame, 1)
            # Send the game a small copy before any debug text is drawn on it.
            if t - last_preview >= preview_every:
                last_preview = t
                small = cv2.resize(view, (vcfg["preview_width"], vcfg["preview_width"] * height // width))
                ok, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    mqtt.publish(topics.VISION_PREVIEW, jpeg.tobytes())
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0
            lines = [
                f"fps: {fps:.0f}",
                f"tag: {'id ' + str(tag_id) if tag_id is not None else 'not found'}"
                + (f"  size {tag_px:.0f} px" if tag_id is not None else ""),
                f"tag found in {100 * sum(tag_seen) / len(tag_seen):.0f}% of recent frames",
                f"body: {'found' if result.pose_landmarks else 'not found'}",
                f"free hand: {'UP' if toss.raised else 'down'}",
            ]
            if other_ids:
                lines.append(f"other tag IDs seen (not in config tag_ids): {other_ids}")
            for i, text in enumerate(lines):
                cv2.putText(view, text, (20, 40 + 35 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            if args.headless:
                continue
            if t < toss_flash_until:
                cv2.putText(view, "TOSS!", (width // 2 - 120, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 3.0,
                            (0, 220, 255), 6)
            cv2.imshow("vision_node (q to quit)", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        landmarker.close()
        mqtt.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

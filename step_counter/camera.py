"""Films your feet with the laptop's camera while the step counter records a trial.

    python step_counter/camera.py [--camera 0] [--board 10.247.137.83]

Runs on the laptop. Point the camera at the floor where you walk (laptop on
the floor, or a USB webcam on a long cable). It:
  1. Shows the camera in a window so you can aim it. Press q in the window to quit.
  2. Listens to the board (same WebSocket as the page) for two things:
     - every sensor reading's board time, to work out how the board's clock
       lines up with the laptop's (the fastest-arriving reading tells us);
     - when a trial starts and stops, to know when to film.
  3. Saves each trial as data/video/<trial name>.mp4, plus <trial name>.csv
     listing every frame's time on the BOARD's clock, so frames line up with
     the sensor rows. Cancelled or too-short trials are deleted.

--camera picks which camera: 0 is usually the built-in one, 1 a USB webcam
(or an iPhone, if Continuity Camera is on). macOS asks for camera permission
the first time.
"""
import argparse
import collections
import json
import queue
import threading
import time
from pathlib import Path

import cv2
from websockets.sync.client import connect

VIDEO_DIR = Path(__file__).resolve().parent / "data" / "video"
CLOCK_WINDOW_S = 10   # line up the clocks using the last 10 s of readings
KEEP_BEFORE_S = 1.5   # also save the frames from just before the trial started
WIDTH, HEIGHT = 640, 480

lock = threading.Lock()
clock = collections.deque()   # (laptop time, laptop time - board time) per reading
events = queue.Queue()        # ("start", file) / ("end", saved) from the board
board_connected = False


def board_offset():
    """Laptop time minus board time, in seconds (None until readings arrive).
    Each reading reaches us a little late; the smallest gap is the least late,
    so it's the best guess at the true difference."""
    with lock:
        return min(o for _, o in clock) if clock else None


# ---------------- board (its own thread) ----------------

def talk_to_board(host):
    global board_connected
    while True:
        try:
            with connect(f"ws://{host}:8765", max_size=None, open_timeout=5) as ws:
                print(f"board: connected to {host}")
                board_connected = True
                filming = False
                for text in ws:
                    msg = json.loads(text)
                    kind = msg.get("type")
                    if kind == "imu":
                        now = time.time()
                        with lock:
                            clock.append((now, now - msg["t"] / 1000))
                            while clock[0][0] < now - CLOCK_WINDOW_S:
                                clock.popleft()
                    elif kind == "recording":
                        if msg["phase"] == "recording" and not filming:
                            filming = True
                            events.put(("start", msg["file"]))
                        elif msg["phase"] == "idle" and filming:
                            filming = False
                            events.put(("end", bool(msg.get("saved"))))
        except Exception as e:
            board_connected = False
            events.put(("end", False))  # a trial cut off mid-way isn't usable
            print(f"board: {e} -- retrying in 2 s")
            time.sleep(2)


# ---------------- one trial's video ----------------

class Trial:
    def __init__(self, file, fps):
        stem = Path(file).stem
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        self.video_path = VIDEO_DIR / f"{stem}.mp4"
        self.csv_path = VIDEO_DIR / f"{stem}.csv"
        self.video = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT))
        self.csv = open(self.csv_path, "w")
        self.csv.write("frame,board_ms,laptop_time\n")
        self.frames = 0
        print(f"filming {self.video_path.name}")

    def add(self, t, frame):
        offset = board_offset()
        self.video.write(frame)
        board_ms = "" if offset is None else round((t - offset) * 1000)
        self.csv.write(f"{self.frames},{board_ms},{t:.3f}\n")
        self.frames += 1

    def close(self, keep):
        self.video.release()
        self.csv.close()
        if keep:
            print(f"kept {self.frames} frames in {self.video_path.name}")
        else:
            self.video_path.unlink(missing_ok=True)
            self.csv_path.unlink(missing_ok=True)
            print("trial cancelled or too short: video deleted")


# ---------------- camera + preview (main thread: macOS wants it here) ----------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="which camera: 0 = built-in, 1 = USB webcam")
    ap.add_argument("--board", default="10.247.137.83", help="the UNO Q's address")
    args = ap.parse_args()

    cam = cv2.VideoCapture(args.camera)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cam.isOpened():
        raise SystemExit(f"Couldn't open camera {args.camera}. Try --camera 1, and check macOS "
                         "Settings -> Privacy & Security -> Camera allows your terminal / VS Code.")
    fps = cam.get(cv2.CAP_PROP_FPS) or 30
    threading.Thread(target=talk_to_board, args=(args.board,), daemon=True).start()

    recent = collections.deque()  # (time, frame) from the last KEEP_BEFORE_S
    trial = None
    count, rate, rate_time = 0, 0, time.time()
    while True:
        ok, frame = cam.read()
        t = time.time()
        if not ok:
            print("camera: no picture -- is another app using it?")
            time.sleep(0.5)
            continue
        if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

        while not events.empty():
            kind, value = events.get()
            if kind == "start" and not trial:
                trial = Trial(value, fps)
                for old_t, old_frame in recent:  # the moments just before Go
                    trial.add(old_t, old_frame)
            elif kind == "end" and trial:
                trial.close(keep=value)
                trial = None

        recent.append((t, frame))
        while recent[0][0] < t - KEEP_BEFORE_S:
            recent.popleft()
        if trial:
            trial.add(t, frame)

        count += 1
        if t - rate_time >= 1:
            rate, count, rate_time = count, 0, t
        lines = [f"{rate} pictures/s",
                 "board: " + ("connected" if board_connected else "NOT connected"),
                 "clocks: " + ("waiting for readings" if board_offset() is None else "lined up")]
        if trial:
            lines.append(f"REC  {trial.frames} frames")
        shown = frame.copy()
        for i, text in enumerate(lines):
            color = (0, 0, 255) if text.startswith("REC") or "NOT" in text else (255, 255, 255)
            cv2.putText(shown, text, (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(shown, text, (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        cv2.imshow("step counter camera (q = quit)", shown)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if trial:
        trial.close(keep=False)
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

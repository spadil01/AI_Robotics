"""Collect labeled hand pictures for the finger-counting safe.

Show your hand to the camera and press a key:
    0-5   record a burst of pictures of that many fingers
    n     record a burst of "no hand" pictures (no hand in view, random stuff)
    space stop a burst early
    u     undo (delete) the last burst
    q     quit

During a burst, slowly move your hand around: closer, farther, tilted,
left, right. Variety is what teaches the network.

Pictures are saved to data/<person>/<label>/ so later we can quiz the
network on people it never trained on.

Run:  python collect_data.py --person sergio [--camera 1]
"""

import argparse
import time
from pathlib import Path

import cv2

LABELS = ["0", "1", "2", "3", "4", "5", "none"]
BURST_SIZE = 40          # pictures per key press
BURST_GAP_S = 0.12       # time between pictures in a burst (~8 per second)
SAVE_WIDTH = 256         # the whole camera picture is shrunk to 256 wide, keeping its shape
DATA_DIR = Path(__file__).parent / "data"


def count_pictures(folder):
    return {lab: len(list((folder / lab).glob("*.jpg"))) for lab in LABELS}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--person", required=True, help="whose hand this is, e.g. sergio")
    ap.add_argument("--camera", type=int, default=0, help="which camera: try 0, 1, 2...")
    args = ap.parse_args()

    person_dir = DATA_DIR / args.person.strip().lower()
    for lab in LABELS:
        (person_dir / lab).mkdir(parents=True, exist_ok=True)
    counts = count_pictures(person_dir)

    cam = cv2.VideoCapture(args.camera)
    if not cam.isOpened():
        raise SystemExit(f"Couldn't open camera {args.camera}. Try --camera 1, and check macOS "
                         "Settings -> Privacy & Security -> Camera allows your terminal / VS Code.")

    burst_label = None    # label being recorded right now (None = not recording)
    burst_files = []      # files saved in the current/last burst (for undo)
    next_shot = 0.0
    message = "Press 0-5 or n to record"

    while True:
        ok, frame = cam.read()
        if not ok:
            print("camera: no picture -- is another app using it?")
            time.sleep(0.5)
            continue
        frame = cv2.flip(frame, 1)  # mirror, so moving right looks like moving right

        # --- take a picture if a burst is running ---
        if burst_label is not None and time.time() >= next_shot:
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (SAVE_WIDTH, round(SAVE_WIDTH * h / w)), interpolation=cv2.INTER_AREA)
            path = person_dir / burst_label / f"{time.time():.3f}.jpg"
            cv2.imwrite(str(path), small, [cv2.IMWRITE_JPEG_QUALITY, 95])
            burst_files.append(path)
            counts[burst_label] += 1
            next_shot = time.time() + BURST_GAP_S
            if len(burst_files) >= BURST_SIZE:
                message = f"Saved {len(burst_files)} pictures of '{burst_label}'"
                burst_label = None

        # --- draw the screen (on a copy, so the drawings never get saved) ---
        view = frame.copy()
        recording = burst_label is not None
        color = (0, 0, 255) if recording else (0, 200, 0)  # red while recording, green otherwise
        cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1), color, 6)  # border
        if recording:
            message = f"RECORDING '{burst_label}'  {len(burst_files)}/{BURST_SIZE}  - move your hand around!"
        cv2.putText(view, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        tally = "  ".join(f"{lab}:{counts[lab]}" for lab in LABELS)
        cv2.putText(view, f"{args.person}  {tally}", (10, view.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("collect data (q = quit)", view)

        # --- keys ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif not recording and (ord("0") <= key <= ord("5") or key == ord("n")):
            burst_label = "none" if key == ord("n") else chr(key)
            burst_files = []
            next_shot = time.time() + 0.5  # half a second to get ready
        elif key == ord(" ") and recording:
            message = f"Stopped: saved {len(burst_files)} pictures of '{burst_label}'"
            burst_label = None
        elif key == ord("u") and not recording and burst_files:
            lab = burst_files[0].parent.name
            for p in burst_files:
                p.unlink(missing_ok=True)
            counts[lab] -= len(burst_files)
            message = f"Undid {len(burst_files)} pictures of '{lab}'"
            burst_files = []

    cam.release()
    cv2.destroyAllWindows()
    print(f"{args.person}: " + ", ".join(f"{lab}={counts[lab]}" for lab in LABELS))


if __name__ == "__main__":
    main()

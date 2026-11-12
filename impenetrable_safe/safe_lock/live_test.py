"""Watch the finger-counting network guess live on the camera.

Big number at the top = the network's guess. The bars on the left show how
sure it is about each answer (all bars add up to 100%).

Run:  python live_test.py [--camera 1]      q = quit
"""

import argparse
import time
from pathlib import Path

import cv2
import torch

from finger_net import LABELS, load, prepare, to_tensor

MODEL_PATH = Path(__file__).parent / "model.pt"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="which camera: try 0, 1, 2...")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = load(MODEL_PATH, device)

    cam = cv2.VideoCapture(args.camera)
    if not cam.isOpened():
        raise SystemExit(f"Couldn't open camera {args.camera}. Try --camera 1.")

    fps, last = 0.0, time.time()
    while True:
        ok, frame = cam.read()
        if not ok:
            print("camera: no picture -- is another app using it?")
            time.sleep(0.5)
            continue
        frame = cv2.flip(frame, 1)  # mirror, same as collect_data.py

        # Ask the network: one picture in, 7 "sureness" numbers out.
        with torch.no_grad():
            scores = net(to_tensor([prepare(frame)]).to(device))
            sure = torch.softmax(scores, 1)[0].cpu().tolist()
        best = max(range(len(LABELS)), key=lambda i: sure[i])

        # --- draw ---
        view = frame
        scale = view.shape[0] / 720  # keep text a sensible size on any camera
        answer = "no hand" if LABELS[best] == "none" else LABELS[best]
        color = (0, 200, 0) if sure[best] > 0.8 else (0, 200, 255)  # green = sure, orange = unsure
        cv2.putText(view, f"{answer}  ({100 * sure[best]:.0f}%)", (int(20 * scale), int(80 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5 * scale, color, int(6 * scale))
        for i, lab in enumerate(LABELS):
            y = int((140 + i * 40) * scale)
            cv2.rectangle(view, (int(90 * scale), y - int(25 * scale)),
                          (int((90 + 300 * sure[i]) * scale), y), color if i == best else (200, 200, 200), -1)
            cv2.putText(view, lab, (int(20 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale,
                        (255, 255, 255), int(2 * scale))

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now
        cv2.putText(view, f"{fps:.0f} pictures/sec", (int(20 * scale), view.shape[0] - int(20 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (255, 255, 255), int(2 * scale))
        cv2.imshow("live test (q = quit)", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""The impenetrable safe: camera + neural network + game rules, all on this laptop.

The UNO Q only obeys: we tell it which lights to turn on and where to move the
servo, and it tells us when the button is pressed.

How to play:
  1. Press the button. The yellow + green LEDs do a light show (about 4 seconds): get ready.
  2. Show the 1st digit with your fingers and hold it still until the bar fills (2 seconds).
     Right -> its yellow LED turns on. Wrong -> everything resets.
  3. Take your hand away, then show the next digit. Same for the 3rd.
  4. All 3 right -> green LED + inside lights on, and the safe opens.
  5. Press the button again to close and lock it.

Keys in the window:  space = pretend button press,  q = quit

Run:  python safe_game.py [--camera 1] [--no-board]
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from board_link import BOARD_IP, Board
from finger_net import LABELS, load, prepare, to_tensor
from safe_rules import CODE, GREEN, INSIDE, READY_S, STEADY_S, SURE, YELLOW, Safe
from vault_scene import VaultScene


MODEL_PATH = Path(__file__).parent / "model.pt"

# Screen colors (OpenCV uses Blue, Green, Red order)
BG = (30, 30, 30)
PANEL = (50, 50, 50)
WHITE = (255, 255, 255)
GREY = (130, 130, 130)
YELLOW_C = (0, 215, 255)
GREEN_C = (80, 220, 80)
RED_C = (70, 70, 255)
TONES = {"info": YELLOW_C, "good": GREEN_C, "bad": RED_C}


# ---------------- the screen ----------------
#
#   +-------------------------+----------+
#   |                         |  camera  |
#   |       VAULT SCENE       +----------+
#   |                         | sees: 3  |
#   |                         | message  |
#   +-------------------------+ bars     |
#   | loading bar      LEDs   |          |
#   +-------------------------+----------+

W, H = 1280, 720
VAULT_X, VAULT_Y, VAULT_W, VAULT_H = 20, 20, 880, 560
SIDE_X = VAULT_X + VAULT_W + 20       # right-hand column
SIDE_W = W - SIDE_X - 20
CAM_H = SIDE_W * 9 // 16


def text(img, msg, x, y, size, color, thick=2, center_w=None):
    if center_w:  # center the text inside a box this wide starting at x
        tw = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, size, thick)[0][0]
        x += (center_w - tw) // 2
    cv2.putText(img, msg, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)


def wrap(msg, size, width, thick=2):
    """Split a message into lines that fit inside width pixels."""
    lines, line = [], ""
    for word in msg.split():
        test = f"{line} {word}".strip()
        if cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, size, thick)[0][0] > width and line:
            lines.append(line)
            line = word
        else:
            line = test
    return lines + [line]


def draw(frame, safe, chances, steady_s, vault):
    screen = np.full((H, W, 3), BG, np.uint8)
    best = int(np.argmax(chances))
    label, sure = LABELS[best], float(chances[best])

    # --- the vault scene (big, top left) ---
    screen[VAULT_Y:VAULT_Y + VAULT_H, VAULT_X:VAULT_X + VAULT_W] = vault.render(safe, time.time(), READY_S)
    cv2.rectangle(screen, (VAULT_X - 1, VAULT_Y - 1), (VAULT_X + VAULT_W, VAULT_Y + VAULT_H), (60, 110, 140), 2)

    # --- loading bar (under the vault) ---
    bx0, by0, bx1, by1 = VAULT_X, VAULT_Y + VAULT_H + 25, VAULT_X + 560, VAULT_Y + VAULT_H + 85
    cv2.rectangle(screen, (bx0, by0), (bx1, by1), PANEL, -1)
    if safe.holding() and label != "none" and sure >= SURE:
        frac = min(safe.fresh(steady_s) / STEADY_S, 1.0)
        cv2.rectangle(screen, (bx0, by0), (bx0 + int((bx1 - bx0) * frac), by1), GREEN_C, -1)
        text(screen, f"Recognizing {label}...  {100 * frac:.0f}%", bx0 + 20, by0 + 42, 1.1, WHITE, 2)
    elif safe.holding():
        text(screen, "Hold up a number", bx0 + 20, by0 + 42, 1.1, GREY, 2)
    cv2.rectangle(screen, (bx0, by0), (bx1, by1), WHITE, 2)

    # --- pretend LEDs (match the real ones), next to the loading bar ---
    leds = [(YELLOW[0], YELLOW_C, "1"), (YELLOW[1], YELLOW_C, "2"), (YELLOW[2], YELLOW_C, "3"),
            (GREEN, GREEN_C, "OK"), (INSIDE, WHITE, "in")]
    for i, (pin, color, name) in enumerate(leds):
        x, y = bx1 + 50 + i * 62, by0 + 22
        on = safe.lit.get(pin, False)
        cv2.circle(screen, (x, y), 18, color if on else PANEL, -1, cv2.LINE_AA)
        cv2.circle(screen, (x, y), 18, WHITE, 2, cv2.LINE_AA)
        text(screen, name, x - 30, y + 42, 0.55, WHITE, 1, center_w=60)

    # --- right column: camera, what the network sees, message, bars ---
    cv2.rectangle(screen, (SIDE_X, VAULT_Y), (SIDE_X + SIDE_W, H - 20), PANEL, -1)
    screen[VAULT_Y:VAULT_Y + CAM_H, SIDE_X:SIDE_X + SIDE_W] = cv2.resize(frame, (SIDE_W, CAM_H))

    top = VAULT_Y + CAM_H
    text(screen, "NETWORK SEES", SIDE_X, top + 30, 0.65, GREY, 2, center_w=SIDE_W)
    sure_color = GREEN_C if sure >= SURE else YELLOW_C
    if label == "none":
        text(screen, "no hand", SIDE_X, top + 95, 1.6, sure_color, 3, center_w=SIDE_W)
    else:
        text(screen, label, SIDE_X, top + 112, 3.2, sure_color, 7, center_w=SIDE_W)
    text(screen, f"{100 * sure:.0f}% sure", SIDE_X, top + 145, 0.75, WHITE, 2, center_w=SIDE_W)

    mx0, my0, mx1, my1 = SIDE_X + 12, top + 160, SIDE_X + SIDE_W - 12, top + 265
    tone = TONES[safe.tone]
    cv2.rectangle(screen, (mx0, my0), (mx1, my1), tone, 3)
    lines = wrap(safe.message, 0.75, mx1 - mx0 - 24)
    ty = my0 + (my1 - my0 - 32 * len(lines)) // 2 + 25
    for i, line in enumerate(lines):
        text(screen, line, mx0, ty + i * 32, 0.75, tone, 2, center_w=mx1 - mx0)

    gx, gy, bar_w = SIDE_X + 75, top + 280, SIDE_W - 140
    for i, lab in enumerate(LABELS):
        y = gy + i * 28
        text(screen, lab, SIDE_X + 15, y + 20, 0.6, WHITE, 2)
        cv2.rectangle(screen, (gx, y + 4), (gx + bar_w, y + 22), BG, -1)
        cv2.rectangle(screen, (gx, y + 4), (gx + int(bar_w * chances[i]), y + 22),
                      sure_color if i == best else GREY, -1)
        text(screen, f"{100 * chances[i]:.0f}%", gx + bar_w + 8, y + 20, 0.5, WHITE, 1)

    return screen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="which camera: try 0, 1, 2...")
    ap.add_argument("--board", default=BOARD_IP, help="the UNO Q's WiFi address")
    ap.add_argument("--no-board", action="store_true", help="play on screen only (space = button)")
    args = ap.parse_args()

    assert len(CODE) == 3 and all(c in "012345" for c in CODE), "CODE must be 3 digits, each 0-5"

    board = None
    if not args.no_board:
        try:
            board = Board(args.board)
            print(f"Connected to the safe ({board.how})")
        except OSError as e:
            raise SystemExit(f"Couldn't reach the safe ({e}). Is the impenetrable-safe app running "
                             "on the UNO Q? Or use --no-board to play on screen only.")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = load(MODEL_PATH, device)
    cam = cv2.VideoCapture(args.camera)
    if not cam.isOpened():
        raise SystemExit(f"Couldn't open camera {args.camera}. Try --camera 1.")

    safe = Safe(board)
    vault = VaultScene(VAULT_W, VAULT_H)
    seeing, seeing_since = None, time.time()   # what the network steadily sees, and since when

    while True:
        ok, frame = cam.read()
        if not ok:
            print("camera: no picture -- is another app using it?")
            time.sleep(0.5)
            continue
        frame = cv2.flip(frame, 1)  # mirror, same as collect_data.py

        with torch.no_grad():
            chances = torch.softmax(net(to_tensor([prepare(frame)]).to(device)), 1)[0].cpu().numpy()
        best = int(chances.argmax())

        # How long has the network been sure it sees this same thing?
        now_seeing = LABELS[best] if chances[best] >= SURE else None
        if now_seeing != seeing:
            seeing, seeing_since = now_seeing, time.time()
        steady_s = time.time() - seeing_since

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") or (board and board.button_pressed()):
            safe.button()
        safe.update(seeing, steady_s)

        cv2.imshow("impenetrable safe (space = button, q = quit)", draw(frame, safe, chances, steady_s, vault))

    safe.lights()   # leave the lights off; the lock stays where it is
    cam.release()
    cv2.destroyAllWindows()
    if board:
        board.close()


if __name__ == "__main__":
    main()

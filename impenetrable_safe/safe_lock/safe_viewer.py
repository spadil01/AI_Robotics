"""Show the safe's screen while the UNO Q makes all the decisions.

The board (impenetrable-safe-onboard app) reads the camera, runs the neural
network and the game rules, and sends us the picture plus what it decided.
This script only draws: the same screen as safe_game.py (vault, loading bar,
LEDs, what the network sees, messages, bars).

Keys in the window:  space = pretend button press (sent to the board),  q = quit

Run:  python safe_viewer.py [--board 10.247.137.83]
"""

import argparse
import base64
import json
import threading
import time

import cv2
import numpy as np

from board_link import BOARD_IP, connect
from safe_game import H, VAULT_H, VAULT_W, draw
from vault_scene import VaultScene

VIEW_PORT = 7200


class RemoteSafe:
    """Looks like safe_rules.Safe to the drawing code, but its values come from the board."""

    def __init__(self):
        self.state, self.prev, self.message, self.tone = "locked", None, "Waiting for the board...", "info"
        self.since = time.time()
        self.entered, self.entered_at, self.need_clear, self.lit = [], [], False, {}

    def take(self, msg, now):
        # The board sends times as "seconds ago"; turn them back into our own clock.
        since = now - msg["age"]
        if msg["state"] != self.state or msg["message"] != self.message or abs(since - self.since) > 0.3:
            self.since = since          # only move it on real changes, so animations don't wobble
        self.state, self.prev = msg["state"], msg["prev"]
        self.message, self.tone = msg["message"], msg["tone"]
        if len(msg["entered_ages"]) != len(self.entered_at):
            self.entered_at = [now - a for a in msg["entered_ages"]]
        self.entered, self.need_clear = msg["entered"], msg["need_clear"]
        self.lit = {int(pin): on for pin, on in msg["lit"].items()}

    def fresh(self, steady_s):
        return min(steady_s, time.time() - self.since)

    def holding(self):
        return self.state == "digit" and not self.need_clear


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", default=BOARD_IP, help="the UNO Q's WiFi address")
    args = ap.parse_args()

    try:
        sock, how = connect(args.board, VIEW_PORT)
    except OSError as e:
        raise SystemExit(f"Couldn't reach the board ({e}). Is the impenetrable-safe-onboard app running "
                         "(python push_to_board.py)? Is the laptop on the same WiFi as the board?")
    print(f"Connected to the board ({how})")
    sock.settimeout(None)

    latest = {"msg": None, "when": 0.0}
    lock = threading.Lock()

    def listen():
        for line in sock.makefile():
            msg = json.loads(line)
            with lock:
                latest["msg"], latest["when"] = msg, time.time()
        print("the board closed the connection")

    threading.Thread(target=listen, daemon=True).start()

    safe = RemoteSafe()
    vault = VaultScene(VAULT_W, VAULT_H)
    frame = np.zeros((216, 384, 3), np.uint8)
    chances, steady, got = np.zeros(7), 0.0, 0.0
    last_seen = None

    while True:
        with lock:
            msg, when = latest["msg"], latest["when"]
        if msg is not None and msg is not last_seen:
            last_seen = msg
            safe.take(msg, when)
            frame = cv2.imdecode(np.frombuffer(base64.b64decode(msg["jpg"]), np.uint8), cv2.IMREAD_COLOR)
            chances, steady, got = np.array(msg["chances"]), msg["steady"], when

        # The board only updates ~10 times a second; keep the loading bar moving in between.
        steady_now = steady + (time.time() - got) if got else 0.0
        screen = draw(frame, safe, chances, steady_now, vault)
        if msg is None or time.time() - when > 2:
            cv2.putText(screen, "no news from the board...", (30, H - 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (70, 70, 255), 2)
        cv2.imshow("impenetrable safe -- board decides (space = button, q = quit)", screen)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            sock.sendall(b"button\n")

    sock.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

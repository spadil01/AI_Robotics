"""Move the safe's servo from the laptop keyboard (to find the open/locked angles).

The impenetrable-safe app must be running on the UNO Q.

Keys (no Enter needed):
    d / a   +5 / -5 degrees
    D / A   +20 / -20 degrees
    q       quit

The servo doesn't move until your first key press. On that press it jumps to
--start (+/- the step), so set --start to where the servo is right now:
0 if it's at its smallest angle, 180 if at its biggest.

Run:  python servo_keys.py          (locked = 180, open = about 115)
"""

import argparse
import sys
import termios
import tty

from board_link import BOARD_IP, Board


def read_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=180, help="where the servo is now (180 = locked)")
    ap.add_argument("--board", default=BOARD_IP, help="the UNO Q's WiFi address")
    args = ap.parse_args()

    board = Board(args.board)
    print(f"Connected to the safe ({board.how}). Servo assumed at {args.start} degrees.")
    print("d/a = +5/-5   D/A = +20/-20   q = quit")

    angle = args.start
    steps = {"d": 5, "a": -5, "D": 20, "A": -20}
    while True:
        key = read_key()
        if key in ("q", "\x03"):  # q or Ctrl-C
            break
        if key not in steps:
            continue
        angle = max(0, min(180, angle + steps[key]))
        board.servo(angle)
        print(f"servo at {angle} degrees")
    board.close()
    print(f"Last angle: {angle}. Write down the locked and open angles!")


if __name__ == "__main__":
    main()

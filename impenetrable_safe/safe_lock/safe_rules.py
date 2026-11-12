"""The safe's game rules, shared by both ways of running it:

  - safe_game.py (laptop decides) uses them on the laptop
  - the impenetrable-safe-onboard app (UNO Q decides) uses a copy on the board
    (push_to_board.py copies this file there)

No camera or drawing code here, just the rules. "board" can be anything with
light(pin, on) and servo(angle) methods.
"""

import time

CODE = "512"           # <-- the secret code: 3 digits, each 0-5
LOCKED_ANGLE = 180     # servo angle that locks the door
OPEN_ANGLE = 115       # servo angle that opens it

FILL_S = 0.3           # light show: each LED joins this long after the last one
BLINK_S = 2.0          # light show: then all blink for this long
HOLD_S = 1.0           # light show: then all stay on for this long
READY_S = 4 * FILL_S + BLINK_S + HOLD_S   # the whole "get ready" show
STEADY_S = 2.0         # hold a digit this long for it to count
CLEAR_S = 0.3          # show "no hand" this long between digits
SURE = 0.8             # the network must be at least this sure (80%)

# Pins on the safe
YELLOW = [8, 9, 10]   # one per digit: 1st, 2nd, 3rd
GREEN = 6
INSIDE = 2


class Safe:
    """All the game rules. Keeps track of what the safe is doing and tells the board."""

    def __init__(self, board):
        self.board = board
        self.lit = {}                  # pin -> on/off, so we can draw pretend LEDs too
        self.need_clear = False
        self.entered, self.entered_at = [], []   # correct digits so far, and when each was accepted
        self.state = None
        self.lights()
        if board:
            board.servo(LOCKED_ANGLE)
        self.go("locked", "Press the button to start", "info")

    # --- talking to the hardware ---
    def light(self, pin, on):
        if self.lit.get(pin) != on:
            self.lit[pin] = on
            if self.board:
                self.board.light(pin, on)

    def lights(self, yellows=(False, False, False), green=False, inside=False):
        for pin, on in zip(YELLOW, yellows):
            self.light(pin, on)
        self.light(GREEN, green)
        self.light(INSIDE, inside)

    def go(self, state, message, tone):
        """Switch to a new state. tone = message color: info (yellow), good (green), bad (red)."""
        self.prev = self.state       # the animations use this (e.g. door closing after open)
        self.state, self.message, self.tone, self.since = state, message, tone, time.time()
        print(f"[{state}] {message}")

    # --- what happens when the button is pressed ---
    def button(self):
        if self.state == "locked":
            self.go("ready", "Get ready...", "info")
        elif self.state == "open":
            if self.board:
                self.board.servo(LOCKED_ANGLE)
            self.lights()
            self.go("locked", "Locked. Press the button to start", "info")
        else:  # pressed in the middle of a try: give up and start over
            self.lights()
            self.go("locked", "Cancelled. Press the button to start", "info")

    def light_show(self):
        """'Get ready' animation: yellow 1, 2, 3 and green light up one by one,
        then they all blink, then they all stay on."""
        t = time.time() - self.since
        order = YELLOW + [GREEN]
        fill_end = len(order) * FILL_S
        if t < fill_end:                       # 1. one by one
            how_many = int(t / FILL_S) + 1
            for i, pin in enumerate(order):
                self.light(pin, i < how_many)
        elif t < fill_end + BLINK_S:           # 2. all blink (4 times a second)
            on = int((t - fill_end) / 0.125) % 2 == 1
            for pin in order:
                self.light(pin, on)
        else:                                  # 3. all stay on
            for pin in order:
                self.light(pin, True)

    # --- what happens every camera picture ---
    def update(self, digit, steady_s):
        """digit = what the network sees and is sure about ("0"-"5", "none", or None
        if it isn't sure); steady_s = how long it has been seeing that same thing."""
        if self.state == "ready":
            if time.time() - self.since < READY_S:
                self.light_show()
            else:
                self.lights()
                self.entered, self.entered_at = [], []
                self.need_clear = False
                self.go("digit", "Show a number and hold it still", "info")

        elif self.state == "digit":
            steady_s = self.fresh(steady_s)
            if self.need_clear:
                if digit == "none" and steady_s >= CLEAR_S:
                    self.need_clear = False
                    self.go("digit", "Show the next number and hold it still", "info")
            elif digit not in (None, "none") and steady_s >= STEADY_S:
                spot = len(self.entered)
                if digit == CODE[spot]:
                    self.entered.append(digit)
                    self.entered_at.append(time.time())
                    self.light(YELLOW[spot], True)
                    if len(self.entered) == len(CODE):
                        self.light(GREEN, True)
                        self.light(INSIDE, True)
                        if self.board:
                            self.board.servo(OPEN_ANGLE)
                        self.go("open", "OPEN! Press the button to close it", "good")
                    else:
                        self.need_clear = True
                        self.go("digit", "Correct! Take your hand out of view", "good")
                else:
                    self.lights()
                    self.go("locked", f"Wrong ({digit})! Press the button to start over", "bad")

    def fresh(self, steady_s):
        """Only count holding time since the game last asked for something."""
        return min(steady_s, time.time() - self.since)

    def holding(self):
        """True while the game is waiting for someone to hold up a digit."""
        return self.state == "digit" and not self.need_clear

"""
Calibration screen shown when the game starts.

Steps (each can be skipped with ENTER, e.g. when testing without hardware):
1. body     - raise both hands: proves the camera sees your whole upper body,
              including a raised hand (needed for the serve toss)
2. hand     - hold the paddle up: the wrist nearest the AprilTag is your paddle
              hand (or press L / R)
3. confirm  - raise your free hand: confirms the choice and practices the toss
4. swing    - swing the paddle once: proves the paddle works

The live camera picture is shown on the left so players can see where to stand.
"""

import math
import time

from direct.gui.DirectGui import DirectFrame
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode

STEPS = [
    ("body", "Stand where you'll play",
     "Stand about 1 to 1.5 m from the screen\nand raise BOTH hands above your head"),
    ("hand", "Find your paddle hand",
     "Hold the paddle up next to your face,\nwith the AprilTag facing the camera\n\n(or press L / R on the keyboard)"),
    ("confirm", "Practice the serve toss",
     "Raise your FREE hand (the one without\nthe paddle) above your shoulder.\nThat's how you toss the ball to serve!"),
    ("swing", "Test the paddle", "Swing the paddle once"),
]
BODY_HOLD = 1.0          # seconds both hands must stay up
HAND_VOTES = 15          # frames of "tag is next to this wrist" needed
HAND_AGREEMENT = 0.8     # ...and this share of them must agree
DONE_PAUSE = 1.5         # "All set!" shown this long
TEXT_LEFT = 0.22         # left edge of the instructions (the camera picture is to their left)

GREEN = (0.45, 1, 0.5, 1)
WHITE = (1, 1, 1, 1)
GRAY = (0.65, 0.65, 0.7, 1)
YELLOW = (1, 0.9, 0.35, 1)


def visible(lm, min_vis):
    return lm is not None and lm[2] >= min_vis


class Calibration:
    def __init__(self, game):
        self.game = game
        self.min_vis = game.cfg["vision"]["min_visibility"]
        self.step = 0
        self.done_at = None
        self.body_since = None
        self.votes = {"left": 0, "right": 0}
        self.last_tag_t = None

        a2d = game.base.aspect2d
        # Text starts right of the camera picture and wraps before the screen's right edge,
        # leaving room for a narrower window (e.g. full screen on a Mac laptop).
        room = min(game.base.getAspectRatio(), 1.5) - TEXT_LEFT - 0.08
        self.overlay = DirectFrame(parent=a2d, frameColor=(0.05, 0.08, 0.15, 0.82), frameSize=(-3, 3, -1, 1))
        self.title = OnscreenText(text="Let's get you set up!", pos=(0, 0.82), scale=0.1, fg=WHITE,
                                  parent=self.overlay, mayChange=True)
        self.checklist = [
            OnscreenText(text="", pos=(TEXT_LEFT, 0.5 - 0.11 * i), scale=0.06, align=TextNode.ALeft,
                         wordwrap=room / 0.06, parent=self.overlay, mayChange=True)
            for i in range(len(STEPS))
        ]
        self.instruction = OnscreenText(text="", pos=(TEXT_LEFT, -0.08), scale=0.065, fg=YELLOW,
                                        align=TextNode.ALeft, wordwrap=room / 0.065, parent=self.overlay,
                                        mayChange=True)
        self.status = OnscreenText(text="", pos=(TEXT_LEFT, -0.5), scale=0.055, fg=WHITE, align=TextNode.ALeft,
                                   wordwrap=room / 0.055, parent=self.overlay, mayChange=True)
        OnscreenText(text="ENTER = skip this step      L / R = choose paddle hand", pos=(0, -0.88), scale=0.045,
                     fg=GRAY, parent=self.overlay)
        game.camera_view.place(-0.72, 0.05, 0.8)
        self.refresh()

    # --- events routed from the game ------------------------------------------

    def on_swing(self):
        if self.current == "swing":
            self.next()

    def on_toss(self):
        if self.current == "confirm":
            self.next()

    def on_key(self, key):
        if self.done_at is not None:
            return
        if key == "enter":
            self.next()
        elif key in ("l", "r") and self.current in ("body", "hand"):
            self.game.set_handedness("left" if key == "l" else "right")
            self.step = 2
            self.refresh()

    # --- flow -------------------------------------------------------------------

    @property
    def current(self):
        return STEPS[self.step][0] if self.step < len(STEPS) else "done"

    def next(self):
        self.game.sounds.play("select", volume=0.6)
        self.step += 1
        self.body_since = None
        if self.step >= len(STEPS):
            self.done_at = time.time()
        self.refresh()

    def refresh(self):
        for i, (text_np, (_, name, _)) in enumerate(zip(self.checklist, STEPS)):
            if i < self.step:
                text_np.setText(f"OK   {name}")
                text_np.setFg(GREEN)
            elif i == self.step:
                text_np.setText(f">    {name}")
                text_np.setFg(WHITE)
            else:
                text_np.setText(f"      {name}")
                text_np.setFg(GRAY)
        if self.current == "done":
            hand = self.game.cfg["game"]["handedness"]
            self.instruction.setText(f"All set! ({hand}-handed)\nStarting the game...")
            self.instruction.setFg(GREEN)
        else:
            self.instruction.setText(STEPS[self.step][2])

    def update(self):
        """Called every frame. Returns True when calibration is finished."""
        sensors = self.game.sensors
        self.game.camera_view.update(sensors.preview_jpeg)
        if self.done_at is not None:
            if time.time() - self.done_at >= DONE_PAUSE:
                self.overlay.destroy()
                self.game.camera_view.hide()
                return True
            return False

        status = ""
        if not sensors.connected:
            status = "Sensors aren't running. Press ENTER to skip."
        elif sensors.preview_jpeg is None:
            status = ("Camera is starting up...\n(the first start can take a few minutes\n"
                      "while your Mac security-checks it)")
        elif not sensors.body_seen():
            status = "The camera can't see you yet."
        elif self.current == "body":
            status = self.check_body()
        elif self.current == "hand":
            status = self.check_hand()
        elif self.current == "swing" and not self.game.paddle_connected:
            status = "Paddle not connected. Press SPACE or ENTER to skip."
        self.status.setText(status)
        frame_color = GREEN if sensors.body_seen() else GRAY if sensors.preview_jpeg is None else (1, 0.4, 0.4, 1)
        self.game.camera_view.place(-0.72, 0.05, 0.8, border_color=frame_color)
        return False

    def check_body(self):
        lm = self.game.sensors.pose.get("landmarks", {})
        needed = ("l_wrist", "r_wrist", "l_shoulder", "r_shoulder")
        if not all(visible(lm.get(n), self.min_vis) for n in needed):
            self.body_since = None
            return "Step back until your head, shoulders and hands all fit in the picture."
        hands_up = lm["l_wrist"][1] < lm["l_shoulder"][1] and lm["r_wrist"][1] < lm["r_shoulder"][1]
        if not hands_up:
            self.body_since = None
            return "I can see you! Now raise both hands."
        self.body_since = self.body_since or time.time()
        if time.time() - self.body_since >= BODY_HOLD:
            self.next()
        return "Hold it..."

    def check_hand(self):
        tag = self.game.sensors.tag
        lm = self.game.sensors.pose.get("landmarks", {})
        if not tag or not tag.get("visible"):
            return "I can't see the AprilTag. Turn it toward the camera."
        if tag["t"] == self.last_tag_t:
            return ""
        self.last_tag_t = tag["t"]
        distances = {}
        for side, name in (("left", "l_wrist"), ("right", "r_wrist")):
            if visible(lm.get(name), self.min_vis):
                # x is stretched to screen proportions so distances are fair in both directions
                distances[side] = math.hypot((lm[name][0] - tag["cx"]) * 16 / 9, lm[name][1] - tag["cy"])
        if not distances:
            return "I can't see your hands."
        self.votes[min(distances, key=distances.get)] += 1
        total = sum(self.votes.values())
        leader = max(self.votes, key=self.votes.get)
        if total >= HAND_VOTES:
            if self.votes[leader] / total >= HAND_AGREEMENT:
                self.game.set_handedness(leader)
                self.next()
                return ""
            self.votes = {"left": 0, "right": 0}     # undecided: start counting again
        return f"Looks like your {leader} hand... hold still"

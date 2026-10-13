"""
Menus controlled by pointing the paddle (its AprilTag) at the screen, like
the Wii Remote pointer. Hover over a button for a moment to select it; the
mouse works too (hover or click), as a backup.

- Pointer: tag position from the camera -> smoothed screen cursor
- Button:  a box with a label and a fill bar that shows the hover progress
- MainMenu:  difficulty, match length, start, practice, controls, recalibrate
- ControlsMenu: how to play, with a Back button
- MatchOverMenu: a single "Continue" button under the final score
- PracticeMenu: a small "End practice" button, shown between rallies
"""

import math
import time

from direct.gui.DirectGui import DirectFrame
from direct.gui.OnscreenImage import OnscreenImage
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import PNMImage, TextNode, Texture, TransparencyAttrib

from common.filters import OneEuroFilter

WHITE = (1, 1, 1, 1)
DARK = (0.05, 0.08, 0.15, 0.75)
IDLE = (0.25, 0.3, 0.42, 0.95)
HOVER = (0.35, 0.42, 0.6, 1)
SELECTED = (0.15, 0.45, 0.85, 1)
GO = (0.2, 0.65, 0.3, 1)
GO_HOVER = (0.28, 0.78, 0.38, 1)
PRACTICE = (0.5, 0.3, 0.75, 1)
PRACTICE_HOVER = (0.6, 0.4, 0.85, 1)
YELLOW = (1, 0.9, 0.35, 1)
CONTROLS = [
    ("Serve", "raise your free hand (or press T)"),
    ("Swing", "swing the paddle (or Z = soft, SPACE = normal, X = hard)"),
    ("High / low", "tilt the paddle face up or down (or hold UP / DOWN)"),
    ("Menus", "point the paddle's AprilTag at a button and hold still"),
    ("Practice", "press M or choose End practice before your serve"),
    ("Keys", "C = calibrate    TAB = timing info    ESC = quit"),
]
TAG_TIMEOUT = 0.4        # s without the tag before the mouse may take over
MOUSE_TIMEOUT = 3.0      # s since the mouse last moved before it stops driving the pointer


def round_cursor_texture(size=64):
    """An orange dot with a white ring, transparent outside (drawn pixel by pixel)."""
    img = PNMImage(size, size, 4)
    c = (size - 1) / 2
    for x in range(size):
        for y in range(size):
            r = ((x - c) ** 2 + (y - c) ** 2) ** 0.5 / c       # 0 center .. 1 edge
            alpha = min(max((1.0 - r) * c, 0.0), 1.0)          # soft 1-pixel edge
            color = (1, 0.55, 0.1) if r < 0.68 else (1, 1, 1)
            img.setXelA(x, y, *color, alpha)
    tex = Texture("cursor")
    tex.load(img)
    return tex


class Pointer:
    """Screen cursor driven by the paddle's AprilTag (or the mouse as a backup)."""

    def __init__(self, game):
        self.game = game
        pcfg = game.cfg["pointer"]
        self.x_range, self.y_range = pcfg["x_range"], pcfg["y_range"]
        self.fx = OneEuroFilter(pcfg["min_cutoff"], pcfg["beta"])
        self.fy = OneEuroFilter(pcfg["min_cutoff"], pcfg["beta"])
        self.last_tag_t = None
        self.tag_seen_at = 0.0
        self.mouse_pos = None
        self.mouse_moved_at = 0.0
        self.glide = pcfg["glide_s"]
        self.target = (0.0, 0.0)           # where the tag/mouse says to go (starts mid-screen)
        self.pos = (0.0, 0.0)              # where the cursor is drawn: always on screen
        self.last_update = time.time()
        a2d = game.base.aspect2d
        self.cursor = OnscreenImage(image=round_cursor_texture(), parent=a2d, scale=0.036)
        self.cursor.setTransparency(TransparencyAttrib.MAlpha)
        self.cursor.setBin("gui-popup", 20)
        self.cursor.hide()

    def _from_tag(self, now):
        tag = self.game.sensors.tag
        if not tag or not tag.get("visible") or tag["t"] == self.last_tag_t:
            return None
        self.last_tag_t = tag["t"]
        self.tag_seen_at = now
        aspect = self.game.base.getAspectRatio()
        (x0, x1), (y0, y1) = self.x_range, self.y_range
        u = min(max((tag["cx"] - x0) / (x1 - x0), 0.0), 1.0)
        v = min(max((tag["cy"] - y0) / (y1 - y0), 0.0), 1.0)
        return (-aspect + 2 * aspect * u, 1 - 2 * v)

    def _from_mouse(self, now):
        watcher = self.game.base.mouseWatcherNode
        if watcher is None or not watcher.hasMouse():
            return None
        m = watcher.getMouse()
        pos = (m.getX() * self.game.base.getAspectRatio(), m.getY())
        if self.mouse_pos is None or abs(pos[0] - self.mouse_pos[0]) + abs(pos[1] - self.mouse_pos[1]) > 0.003:
            self.mouse_moved_at = now
        self.mouse_pos = pos
        return pos if now - self.mouse_moved_at < MOUSE_TIMEOUT else None

    def update(self):
        """Returns the cursor position (aspect2d units), or None while nothing is
        pointing (so buttons don't select themselves). The cursor itself never
        disappears: if the tag is lost it waits where it was."""
        now = time.time()
        dt, self.last_update = min(now - self.last_update, 0.1), now
        tag_pos = self._from_tag(now)
        mouse_pos = self._from_mouse(now)
        tracking = tag_pos or now - self.tag_seen_at < TAG_TIMEOUT
        if tag_pos:
            self.target = (self.fx(tag_pos[0], now), self.fy(tag_pos[1], now))
        elif mouse_pos and not tracking:
            tracking = True
            self.target = mouse_pos
            self.fx.reset()
            self.fy.reset()
        # Glide toward the target every frame, so a tag that jumps between
        # camera frames (fast moves) still makes the cursor slide smoothly.
        k = 1 - math.exp(-dt / self.glide)
        self.pos = tuple(p + (t - p) * k for p, t in zip(self.pos, self.target))
        self.cursor.setPos(self.pos[0], 0, self.pos[1])
        self.cursor.show()
        return self.pos if tracking else None

    def destroy(self):
        self.cursor.destroy()


class Button:
    def __init__(self, parent, label, pos, size, on_click, go=False, colors=None, sublabel=None):
        """go: a green action button. colors: (normal, hovered) for another action color.
        sublabel: a small second line under the label."""
        w, h = size
        self.w, self.h = w, h
        self.x, self.y = pos
        self.on_click = on_click
        self.colors = (GO, GO_HOVER) if go else colors
        self.selected = False
        self.frame = DirectFrame(parent=parent, pos=(self.x, 0, self.y), frameSize=(-w / 2, w / 2, -h / 2, h / 2),
                                 frameColor=self.colors[0] if self.colors else IDLE)
        if sublabel:
            OnscreenText(text=label, parent=self.frame, pos=(0, h * 0.02), scale=h * 0.4, fg=WHITE)
            OnscreenText(text=sublabel, parent=self.frame, pos=(0, -h * 0.32), scale=h * 0.2, fg=(1, 1, 1, 0.8))
        else:
            OnscreenText(text=label, parent=self.frame, pos=(0, -h * 0.13), scale=h * 0.42, fg=WHITE)
        self.bar = DirectFrame(parent=self.frame, pos=(-w / 2, 0, -h / 2), frameSize=(0, w, 0, 0.022),
                               frameColor=(1, 1, 1, 0.9))
        self.set_progress(0)

    def contains(self, p):
        return p is not None and abs(p[0] - self.x) <= self.w / 2 and abs(p[1] - self.y) <= self.h / 2

    def set_progress(self, k):
        self.bar.setSx(max(k, 0.001))

    def style(self, hovered):
        if self.colors:
            color = self.colors[1] if hovered else self.colors[0]
        else:
            color = SELECTED if self.selected else HOVER if hovered else IDLE
        self.frame["frameColor"] = color


class Menu:
    """A screen of buttons selected by hovering (dwell) or clicking."""

    def __init__(self, game, dim=True):
        self.game = game
        self.dwell = game.cfg["pointer"]["dwell_s"]
        self.root = DirectFrame(parent=game.base.aspect2d, frameColor=DARK if dim else (0, 0, 0, 0),
                                frameSize=(-3, 3, -1, 1))
        self.buttons = []
        self.pointer = Pointer(game)
        self.hovered = None
        self.hover_time = 0.0
        self.fired = False
        game.base.accept("mouse1", self.click)

    def button(self, *args, **kwargs):
        b = Button(self.root, *args, **kwargs)
        self.buttons.append(b)
        return b

    def text(self, text, pos, scale, fg=WHITE):
        return OnscreenText(text=text, pos=pos, scale=scale, fg=fg, parent=self.root, mayChange=True)

    def click(self):
        if self.hovered:
            self.fired = True
            self.game.sounds.play("select", volume=0.6)
            self.hovered.on_click()     # may destroy this menu, so nothing after this

    def update(self, dt):
        p = self.pointer.update()
        hovered = next((b for b in self.buttons if b.contains(p)), None)
        if hovered is not self.hovered:
            self.hovered, self.hover_time, self.fired = hovered, 0.0, False
        if hovered and not self.fired:
            self.hover_time += dt
            if self.hover_time >= self.dwell:
                self.fired = True       # must move off the button before it can fire again
                self.game.sounds.play("select", volume=0.6)
                hovered.on_click()      # may destroy this menu, so stop here
                return
        for b in self.buttons:
            b.style(b is hovered)
            b.set_progress(self.hover_time / self.dwell if b is hovered and not self.fired else 0)

    def destroy(self):
        self.game.base.ignore("mouse1")
        self.pointer.destroy()
        self.root.destroy()


class MainMenu(Menu):
    def __init__(self, game, on_start, on_practice, on_controls):
        super().__init__(game)
        self.on_start = on_start
        settings = game.settings
        self.text("Table Tennis", (0, 0.72), 0.14)

        self.text("Difficulty", (0, 0.42), 0.07)
        self.difficulty = {}
        for i, (key, label) in enumerate((("beginner", "Beginner"), ("intermediate", "Intermediate"), ("pro", "Pro"))):
            self.difficulty[key] = self.button(label, (-0.72 + 0.72 * i, 0.27), (0.64, 0.17),
                                               lambda k=key: self.choose("difficulty", k))

        self.text("Match length", (0, 0.02), 0.07)
        self.points = {}
        for i, n in enumerate((6, 11)):
            self.points[n] = self.button(f"{n} points", (-0.38 + 0.76 * i, -0.13), (0.64, 0.17),
                                         lambda n=n: self.choose("match_points", n))

        self.button("Start match!", (-0.36, -0.45), (0.64, 0.2), self.on_start, go=True)
        self.button("Practice", (0.36, -0.45), (0.64, 0.2), on_practice,
                    colors=(PRACTICE, PRACTICE_HOVER), sublabel="no score, just rally")
        left = -game.base.getAspectRatio() + 0.33
        self.button("Controls", (left, -0.73), (0.5, 0.1), on_controls)
        self.button("Recalibrate", (left, -0.86), (0.5, 0.1), game.recalibrate_from_menu)
        self.text("Point the paddle's AprilTag at a button and hold still to choose (the mouse works too)",
                  (0, -0.7), 0.042, fg=(0.8, 0.8, 0.85, 1))
        self.refresh()

    def choose(self, setting, value):
        self.game.settings[setting] = value
        self.refresh()

    def refresh(self):
        for key, b in self.difficulty.items():
            b.selected = key == self.game.settings["difficulty"]
        for n, b in self.points.items():
            b.selected = n == self.game.settings["match_points"]


class ControlsMenu(Menu):
    """How to play, listed on its own screen."""

    def __init__(self, game, on_back):
        super().__init__(game)
        self.text("Controls", (0, 0.72), 0.12)
        for i, (name, how) in enumerate(CONTROLS):
            y = 0.42 - 0.15 * i
            OnscreenText(text=name, pos=(-0.42, y), scale=0.06, fg=YELLOW, align=TextNode.ARight, parent=self.root)
            OnscreenText(text=how, pos=(-0.34, y), scale=0.06, fg=WHITE, align=TextNode.ALeft, parent=self.root)
        self.button("Back", (0, -0.62), (0.6, 0.17), on_back, go=True)


class MatchOverMenu(Menu):
    """Just a Continue button; the game's own text shows the result above it."""

    def __init__(self, game, on_continue):
        super().__init__(game, dim=False)
        self.button("Continue", (0, -0.5), (0.7, 0.18), on_continue, go=True)


class PracticeMenu(Menu):
    """A small button in the top-left corner to leave practice."""

    def __init__(self, game, on_end):
        super().__init__(game, dim=False)
        self.button("End practice", (-game.base.getAspectRatio() + 0.36, 0.86), (0.6, 0.12), on_end)

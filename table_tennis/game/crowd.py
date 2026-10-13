"""
The audience: bleachers on both sides and behind the far end of the court, filled with
little bean spectators, plus low barrier boards around the play area.

Each spectator supports the player, the opponent, or nobody (many wear their
player's shirt color). After a point the winner's fans jump and cheer, the
loser's fans slump with their hands on their heads, and neutral fans clap.
"""

import math
import random

from panda3d.core import Point3

from .physics import CPU, PLAYER
from .shapes import part

SKINS = [(0.98, 0.84, 0.72, 1), (0.94, 0.76, 0.6, 1), (0.8, 0.6, 0.45, 1), (0.6, 0.42, 0.3, 1), (0.42, 0.28, 0.2, 1)]
HAIRS = [(0.1, 0.08, 0.07, 1), (0.35, 0.22, 0.12, 1), (0.75, 0.55, 0.25, 1), (0.55, 0.15, 0.08, 1), (0.85, 0.85, 0.85, 1)]
SHIRTS = [(0.9, 0.3, 0.3, 1), (0.25, 0.55, 0.9, 1), (0.95, 0.75, 0.2, 1), (0.65, 0.4, 0.85, 1),
          (0.2, 0.7, 0.75, 1), (0.95, 0.95, 0.95, 1), (0.9, 0.45, 0.65, 1)]
EYE = (0.12, 0.1, 0.1, 1)
SEAT = (0.62, 0.65, 0.72, 1)
SEAT_EDGE = (0.5, 0.53, 0.6, 1)
BOARD_BLUE = (0.16, 0.3, 0.62, 1)
BOARD_WHITE = (0.95, 0.95, 0.97, 1)

# Inner edge of the first bleacher row (the stands form a rectangle around this).
INNER_X = 3.3
INNER_NEAR_Y = -4.6
INNER_FAR_Y = 5.4
TIERS = 3
TIER_DEPTH = 0.62
TIER_HEIGHT = 0.34
SEAT_SPACING = 0.5
EMPTY_SEATS = 0.15
# Who people support: player, opponent, or neutral.
SUPPORT = [(PLAYER, 0.45), (CPU, 0.4), (None, 0.15)]
TEAM_SHIRT_CHANCE = 0.6


class Spectator:
    def __init__(self, parent, pos, look_at, team, team_shirt, rng):
        self.team = team
        self.root = parent.attachNewNode("spectator")
        self.root.setPos(pos)
        self.root.lookAt(look_at)
        self.root.setP(0)
        self.root.setScale(0.9)
        self.base_z = pos.z
        self.phase = rng.uniform(0, 2 * math.pi)
        self.speed = rng.uniform(2.5, 4.0)
        self.action = None            # ("cheer" | "sad" | "clap", seconds, delay, duration)
        self.wears_team = team is not None and rng.random() < TEAM_SHIRT_CHANCE
        shirt = team_shirt if self.wears_team else rng.choice(SHIRTS)
        skin = rng.choice(SKINS)

        # Static pieces are merged ("flattened") so the crowd stays cheap to draw.
        self.body = self.root.attachNewNode("body")
        part(self.body, "capsule", shirt, pos=(0, 0, 0.3), scale=(0.32, 0.26, 0.32), cyl=0.85)
        self.body.flattenStrong()
        self.head = self.root.attachNewNode("head")
        self.head.setPos(0, 0, 0.74)
        head_geom = self.head.attachNewNode("head_geom")
        part(head_geom, "sphere", skin, scale=0.27)
        part(head_geom, "sphere", rng.choice(HAIRS), pos=(0, -0.015, 0.05), scale=(0.28, 0.26, 0.2))
        for side in (-1, 1):
            part(head_geom, "sphere", EYE, pos=(0.05 * side, 0.12, 0.0), scale=(0.035, 0.025, 0.05))
        head_geom.flattenStrong()
        self.hands = [(part(self.root, "sphere", skin, scale=0.09), side) for side in (-1, 1)]

    def react(self, kind, strength, rng):
        if kind == "cheer" and rng.random() < 0.1 * (1 - strength):
            return                                    # a few stay seated for small moments
        duration = rng.uniform(0.9, 1.4) * (1 + strength)
        self.action = [kind, 0.0, rng.uniform(0, 0.25), duration]

    def update(self, dt, t):
        z = self.base_z + 0.015 * math.sin(t * self.speed + self.phase)
        head_p = 0.0
        hand_x, hand_y, hand_z = 0.22, 0.08, 0.32      # resting on the lap
        if self.action:
            kind, elapsed, delay, duration = self.action
            self.action[1] += dt
            a = elapsed - delay
            if a > duration:
                self.action = None
            elif a > 0:
                ramp = min(a / 0.15, 1.0) * min((duration - a) / 0.2, 1.0)
                if kind == "cheer":                    # jump with both hands up
                    z += 0.16 * abs(math.sin(a * math.pi * 2.6))
                    hand_x, hand_z = 0.22 + 0.06 * ramp, 0.32 + 0.55 * ramp
                elif kind == "sad":                    # slump, head down, hands on head
                    z -= 0.05 * ramp
                    head_p = -22 * ramp
                    hand_x, hand_y, hand_z = 0.22 - 0.08 * ramp, 0.08, 0.32 + 0.48 * ramp
                else:                                  # clap in front of the chest
                    clap = abs(math.sin(a * math.pi * 3.5))
                    hand_x, hand_y, hand_z = 0.04 + 0.12 * clap, 0.2, 0.46
        self.root.setZ(z)
        self.head.setP(head_p)
        for hand, side in self.hands:
            hand.setPos(hand_x * side, hand_y, hand_z)


class Crowd:
    def __init__(self, render, team_shirts, seed=7):
        """team_shirts: {PLAYER: color, CPU: color} so fans can wear their player's color."""
        self.rng = random.Random(seed)
        self.time = 0.0
        self.spectators = []
        self.node = render.attachNewNode("crowd")
        self.team_shirts = team_shirts

        for tier in range(TIERS):
            d = tier * TIER_DEPTH                     # how far this row is set back
            top = (tier + 1) * TIER_HEIGHT
            x_in, near_in, far_in = INNER_X + d, INNER_NEAR_Y - d, INNER_FAR_Y + d
            x_out, near_out, far_out = x_in + TIER_DEPTH, near_in - TIER_DEPTH, far_in + TIER_DEPTH
            mid = TIER_DEPTH / 2
            # Side benches run from behind the camera to the far stand, so every
            # edge of the visible court is enclosed and the far corners are filled.
            for side in (-1, 1):
                self._bench((side * (x_in + mid), (near_out + far_out) / 2), (TIER_DEPTH, far_out - near_out), top)
            self._bench((0, far_in + mid), (2 * x_in, TIER_DEPTH), top)
            # Seats along the middle of each bench, everyone facing the table.
            for side in (-1, 1):
                self._row([(side * (x_in + mid), y) for y in self._steps(near_out + mid, far_out - mid)], top)
            self._row([(x, far_in + mid) for x in self._steps(-x_in + 0.25, x_in - 0.25)], top)

        for x in (-2.55, 2.55):
            self._board((x, 0.4), (0.06, 8.6))
        for y in (-3.9, 4.7):
            self._board((0, y), (5.16, 0.06))

    @staticmethod
    def _steps(start, end):
        n = int((end - start) / SEAT_SPACING)
        return [start + i * (end - start) / max(n, 1) for i in range(n + 1)]

    def _row(self, spots, top):
        for x, y in spots:
            if self.rng.random() < EMPTY_SEATS:
                continue
            team = self.rng.choices([s for s, _ in SUPPORT], [w for _, w in SUPPORT])[0]
            self.spectators.append(Spectator(self.node, Point3(x, y, top), Point3(0, 0.4, top), team,
                                             self.team_shirts.get(team), self.rng))

    def _bench(self, center, size, top):
        x, y = center
        part(self.node, "box", SEAT, pos=(x, y, top / 2), scale=(size[0], size[1], top))
        part(self.node, "box", SEAT_EDGE, pos=(x, y, top - 0.015), scale=(size[0] + 0.01, size[1] + 0.01, 0.03))

    def _board(self, center, size):
        x, y = center
        part(self.node, "box", BOARD_BLUE, pos=(x, y, 0.22), scale=(size[0], size[1], 0.44))
        part(self.node, "box", BOARD_WHITE, pos=(x, y, 0.46), scale=(size[0] + 0.01, size[1] + 0.01, 0.04))

    def set_team_color(self, team, color):
        """Fans wearing their player's shirt switch to the new color."""
        for s in self.spectators:
            if s.team == team and s.wears_team:
                s.body.setColor(*color, 1)        # priority 1 overrides the color baked in by flattening

    def react(self, winner, strength=0.5):
        """After a point: winner's fans cheer, loser's fans are sad, neutral fans clap.
        strength 0..1 makes reactions longer (1 = end of the match)."""
        for s in self.spectators:
            kind = "clap" if s.team is None else "cheer" if s.team == winner else "sad"
            s.react(kind, strength, self.rng)

    def update(self, dt):
        self.time += dt
        for s in self.spectators:
            s.update(dt, self.time)

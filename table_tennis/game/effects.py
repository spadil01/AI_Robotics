"""
Hit effect that makes good shots feel punchy: a burst of glowing bits where
the paddle meets the ball. Harder hits make more, bigger and faster sparkles.
"""

import math
import random

from panda3d.core import Vec3

from .shapes import part

GRAVITY = 4.0            # m/s^2 pulling sparkles down (gentler than real, so they float a bit)
LIFE = 0.35              # s a sparkle lasts
WHITE_HOT = (1.0, 0.97, 0.8, 1)
GOLD = (1.0, 0.85, 0.3, 1)
SMASH_ORANGE = (1.0, 0.5, 0.15, 1)


class Sparkles:
    def __init__(self, render, rng=None):
        self.node = render.attachNewNode("sparkles")
        self.rng = rng or random.Random()
        self.bits = []          # [node, velocity, age, size]

    def burst(self, pos, power, smash=False):
        """pos: Panda3D point where the ball was hit. power 0..1."""
        count = int(6 + 14 * power) + (10 if smash else 0)
        speed = 0.8 + 1.6 * power + (0.8 if smash else 0)
        colors = (SMASH_ORANGE, GOLD, WHITE_HOT) if smash else (WHITE_HOT, GOLD)
        for _ in range(count):
            # Random direction, biased up and toward the far side (where the ball goes).
            theta = self.rng.uniform(0, 2 * math.pi)
            up = self.rng.uniform(-0.3, 1.0)
            flat = math.sqrt(max(1 - up * up, 0))
            v = Vec3(flat * math.cos(theta), flat * math.sin(theta) + 0.4, up) * speed * self.rng.uniform(0.5, 1.0)
            size = self.rng.uniform(0.02, 0.035) * (0.6 + 0.8 * power) * (1.3 if smash else 1.0)
            bit = part(self.node, "sphere", self.rng.choice(colors), pos=pos, scale=size)
            self.bits.append([bit, v, 0.0, size])

    def update(self, dt):
        alive = []
        for bit in self.bits:
            node, v, age, size = bit
            age += dt
            if age >= LIFE:
                node.removeNode()
                continue
            v.z -= GRAVITY * dt
            v *= math.exp(-3 * dt)                    # air drag
            node.setPos(node.getPos() + v * dt)
            node.setScale(size * (1 - age / LIFE))    # shrink away
            bit[2] = age
            alive.append(bit)
        self.bits = alive

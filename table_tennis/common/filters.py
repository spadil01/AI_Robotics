"""
One Euro filter (Casiez et al., 2012): smooths a noisy signal a lot when it
moves slowly (steady cursor) and only a little when it moves fast (no lag).
"""

import math


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff = min_cutoff   # lower = steadier when still
        self.beta = beta               # higher = less lag when moving fast
        self.d_cutoff = d_cutoff
        self.x = None
        self.dx = 0.0
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x = None

    def __call__(self, x, t):
        if self.x is None or t <= self.t:
            self.x, self.dx, self.t = x, 0.0, t
            return x
        dt = t - self.t
        self.t = t
        dx = (x - self.x) / dt
        self.dx += self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        self.x += self._alpha(cutoff, dt) * (x - self.x)
        return self.x

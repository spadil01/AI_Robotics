"""
Ball physics: gravity, table bounces, the net and the floor.

Uses plain floats (no Ursina) so the ball can be copied and run ahead in time
to predict where it will be, which is how the game knows when a swing should
connect.

Coordinates (meters): x to the player's right, y up, z away from the player.
The table is centered on the origin; the player's half is z < 0.
"""

import math
from dataclasses import dataclass, field

G = 9.81
TABLE_TOP = 0.58             # lower than a real table (0.76) to suit the small Mii-style characters
HALF_WIDTH = 0.7625
HALF_LENGTH = 1.37
NET_HEIGHT = 0.1525
NET_OVERHANG = 0.15          # the net sticks out past the table edges
BALL_RADIUS = 0.02
RESTITUTION = 0.88           # bounce height kept after hitting the table
TABLE_FRICTION = 0.97        # horizontal speed kept after hitting the table
DT = 1 / 240                 # physics step (s), independent of frame rate

# Barrier boards around the court (must match crowd.py). The ball bounces off
# them (and the stands behind them) instead of flying through.
ARENA_HALF_WIDTH = 2.52
ARENA_NEAR = -3.87
ARENA_FAR = 4.67
WALL_BOUNCE = 0.35           # speed kept when bouncing off a barrier
FLOOR_BOUNCE = 0.55          # bounce height kept on the floor
FLOOR_FRICTION = 0.8         # horizontal speed kept per floor bounce

PLAYER, CPU = "player", "cpu"


def side_of(z):
    return PLAYER if z < 0 else CPU


def other(side):
    return CPU if side == PLAYER else PLAYER


@dataclass
class Ball:
    pos: list = field(default_factory=lambda: [0.0, 1.0, 0.0])
    vel: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # Velocity to use after the next table bounce instead of a normal bounce.
    # Serves use it so the second half of the serve lands exactly on target.
    bounce_override: list | None = None
    dead: bool = False           # came to rest on the floor
    on_floor: bool = False       # has touched the floor (it bounces and rolls after that)

    def copy(self):
        return Ball(list(self.pos), list(self.vel),
                    None if self.bounce_override is None else list(self.bounce_override), self.dead, self.on_floor)

    def step(self):
        """Advance one DT. Returns a list of (event, side) tuples, where event
        is "net", "bounce" or "floor"."""
        if self.dead:
            return []
        events = []
        prev_y, prev_z = self.pos[1], self.pos[2]
        self.vel[1] -= G * DT
        for i in range(3):
            self.pos[i] += self.vel[i] * DT
        x, y, z = self.pos

        crossed_net = (prev_z < 0) != (z < 0)
        if (crossed_net and TABLE_TOP - 0.02 < y - BALL_RADIUS < TABLE_TOP + NET_HEIGHT
                and abs(x) < HALF_WIDTH + NET_OVERHANG):
            self.pos[2] = math.copysign(BALL_RADIUS, prev_z)
            self.vel = [self.vel[0] * 0.3, self.vel[1] * 0.3, -self.vel[2] * 0.15]
            events.append(("net", side_of(prev_z)))
        elif (self.vel[1] < 0 and prev_y - BALL_RADIUS >= TABLE_TOP >= y - BALL_RADIUS
              and abs(x) <= HALF_WIDTH and abs(z) <= HALF_LENGTH):
            # Crossed the table surface during this step (works even for fast balls).
            self.pos[1] = TABLE_TOP + BALL_RADIUS
            if self.bounce_override is not None:
                self.vel, self.bounce_override = self.bounce_override, None
            else:
                self.vel = [self.vel[0] * TABLE_FRICTION, -self.vel[1] * RESTITUTION,
                            self.vel[2] * TABLE_FRICTION]
            events.append(("bounce", side_of(z)))

        # Barriers: bounce back off them.
        if abs(self.pos[0]) > ARENA_HALF_WIDTH:
            self.pos[0] = math.copysign(ARENA_HALF_WIDTH, self.pos[0])
            self.vel[0] *= -WALL_BOUNCE
        if not ARENA_NEAR <= self.pos[2] <= ARENA_FAR:
            self.pos[2] = min(max(self.pos[2], ARENA_NEAR), ARENA_FAR)
            self.vel[2] *= -WALL_BOUNCE

        # Floor: report the first touch, then bounce and roll to a stop.
        if self.pos[1] <= BALL_RADIUS:
            self.pos[1] = BALL_RADIUS
            if not self.on_floor:
                self.on_floor = True
                events.append(("floor", side_of(z)))
            if self.vel[1] < 0:
                self.vel = [self.vel[0] * FLOOR_FRICTION, -self.vel[1] * FLOOR_BOUNCE, self.vel[2] * FLOOR_FRICTION]
            if abs(self.vel[1]) < 0.3:
                self.vel[1] = 0.0
                self.vel[0] *= 0.98                  # rolling
                self.vel[2] *= 0.98
                if abs(self.vel[0]) + abs(self.vel[2]) < 0.05:
                    self.dead = True
        return events


def predict_contact(ball, side, plane_z, max_time=4.0):
    """Run a copy of the ball forward to find where `side` should hit it.

    The ball must bounce once on `side`'s half first. Normally contact is where
    it crosses the hitting plane `plane_z` behind the table. If it would bounce
    a second time before getting there (a short ball), contact is at the top
    of its first bounce instead. Returns (seconds_from_now, position) or None
    if the ball never becomes playable (net, out, etc.).
    """
    b = ball.copy()
    bounced = False
    apex = None
    t = 0.0
    while t < max_time and not b.dead:
        rising = b.vel[1] > 0
        events = b.step()
        t += DT
        for event, ev_side in events:
            if event in ("net", "floor"):
                return None
            if event == "bounce" and ev_side == side:
                if bounced:
                    return apex
                bounced = True
        if bounced and rising and b.vel[1] <= 0:
            apex = (t, list(b.pos))
        crossed = b.pos[2] <= plane_z if side == PLAYER else b.pos[2] >= plane_z
        if crossed:
            return (t, list(b.pos)) if bounced else None
    return None


def solve_launch(start, target, flight_time):
    """Velocity that carries the ball from `start` to `target` in `flight_time`
    seconds under gravity (exact, since there's no drag)."""
    T = flight_time
    return [(target[0] - start[0]) / T,
            (target[1] - start[1] + 0.5 * G * T * T) / T,
            (target[2] - start[2]) / T]


def clears_net(start, vel, margin=0.03):
    if vel[2] == 0:
        return True
    t = -start[2] / vel[2]  # when the ball reaches the net plane z = 0
    if t <= 0:
        return True  # moving away from the net
    y = start[1] + vel[1] * t - 0.5 * G * t * t
    return y >= TABLE_TOP + NET_HEIGHT + BALL_RADIUS + margin


def safe_launch(start, target, flight_time):
    """Like solve_launch, but slows the shot down (higher arc) until it clears the net."""
    T = flight_time
    vel = solve_launch(start, target, T)
    while not clears_net(start, vel) and T < 2.5:
        T *= 1.08
        vel = solve_launch(start, target, T)
    return vel

"""
Turns swings into ball velocities.

Shots aim for a landing spot first, then solve for the velocity that reaches
it (see physics.safe_launch). Good swings always land; bad ones miss in ways
that make sense (wide, long, into the net), which is what makes Wii-style
table tennis feel fair.
"""

import random
from dataclasses import dataclass

from .physics import (BALL_RADIUS, HALF_LENGTH, HALF_WIDTH, PLAYER, TABLE_TOP,
                      safe_launch, solve_launch)

LANDING_Y = TABLE_TOP + BALL_RADIUS


@dataclass
class Swing:
    t: float          # when the swing peaked (time.time())
    strength: float   # 0 = gentle, 1 = full power
    face: float       # paddle face: -1 closed (low/net) .. 0 flat .. +1 open (high/long)
    side: int = 1     # +1 / -1 spin direction (forehand / backhand)


def lerp(a, b, t):
    return a + (b - a) * t


def player_shot(contact, swing, timing, handed, cfg):
    """Velocity for the player's return.

    timing: -1 (as early as allowed) .. 0 (perfect) .. +1 (as late as allowed)
    handed: +1 right-handed, -1 left-handed
    """
    # Early swings pull the ball cross-court, late swings push it the other way.
    if abs(timing) > cfg["wide_timing"]:
        target_x = -timing / abs(timing) * (HALF_WIDTH + 0.25) * handed      # out wide
    else:
        target_x = -timing * cfg["aim_width"] * HALF_WIDTH * handed

    flight = lerp(cfg["flight_time_soft"], cfg["flight_time_hard"], swing.strength)
    if contact[1] > TABLE_TOP + cfg["smash_height"] and swing.strength > 0.8:
        flight *= cfg["smash_factor"]       # smash: hard swing at a high ball
    face = swing.face
    if face < -cfg["face_safe"]:
        # Face too closed: driven into the net.
        return solve_launch(contact, [target_x, TABLE_TOP + 0.06, 0.0], flight * 0.5)
    if face > cfg["face_safe"]:
        # Face too open: sails off the end of the table.
        return solve_launch(contact, [target_x, LANDING_Y, HALF_LENGTH + 0.4], flight * 1.1)
    depth = (face + cfg["face_safe"]) / (2 * cfg["face_safe"])       # 0 short .. 1 deep
    target_z = lerp(0.45, HALF_LENGTH - 0.15, depth)
    # Open face = high, slow lob; closed face = low, fast drive.
    return safe_launch(contact, [target_x, LANDING_Y, target_z], flight * (1 + 0.35 * face))


def opponent_shot(contact, diff, incoming_height, rng=random):
    """Velocity for the CPU's return, aimed somewhere on the player's half."""
    target_x = rng.uniform(-1, 1) * diff["target_spread"] * HALF_WIDTH
    target_z = -rng.uniform(*diff["depth"])
    flight = rng.uniform(*diff["return_time"])
    if incoming_height > TABLE_TOP + 0.45 and rng.random() < diff["smash_chance"]:
        flight *= 0.6
    return safe_launch(contact, [target_x, LANDING_Y, target_z], flight)


def serve_shot(contact, server, target_x, strength, cfg):
    """Velocity for a serve, plus the velocity to use after its first bounce.

    A serve bounces once on the server's half, then once on the receiver's.
    The first part is aimed at a spot on the server's half; at that bounce the
    ball is given the exact velocity that reaches the receiver's target.
    """
    direction = 1 if server == PLAYER else -1          # +z when the player serves
    first_bounce = [contact[0] * 0.5, LANDING_Y, -direction * 0.55]
    second_bounce = [target_x, LANDING_Y, direction * lerp(0.7, 1.0, strength)]
    vel = solve_launch(contact, first_bounce, cfg["serve_first_time"])
    after_bounce = safe_launch(first_bounce, second_bounce,
                               lerp(cfg["serve_time_soft"], cfg["serve_time_hard"], strength))
    return vel, after_bounce

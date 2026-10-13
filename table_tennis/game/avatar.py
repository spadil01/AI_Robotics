"""
Mii-style "bean" characters: a rounded body, a floating head, two floating
hands and little feet. With no arms or legs, nothing can bend the wrong way
or poke through the body.

Animation is procedural, every frame:
- the paddle hand follows a smooth curve (back, out to meet the ball, then a
  relaxed finish near the opposite shoulder) that always stays in front of
  the body and above the table
- the wrist turns the paddle along the swing; the face opens for lobs
- hoppy side-steps with squash and stretch, leaning into the movement
- an automatic wind-up as the ball approaches, head follows the ball
- happy jumps and a sad droop after points

With dog=True the same character is a dog (the Pro opponent): fur, a snout,
floppy ears that bounce and droop, and a tail that wags faster when happy.
It wears a sweatband in its team color.

Built facing +y (Panda3D "forward"); the opponent is turned around. All
positions are Panda3D (x right, y forward, z up), relative to the feet.
"""

import math

from panda3d.core import Point3, Vec3

from .shapes import part

SKIN = (0.94, 0.78, 0.65, 1)
EYE = (0.16, 0.12, 0.12, 1)
SHOE = (0.3, 0.3, 0.32, 1)
PADDLE_RED = (0.78, 0.16, 0.16, 1)
PADDLE_BLACK = (0.08, 0.08, 0.09, 1)
HANDLE = (0.67, 0.47, 0.27, 1)
FUR = (0.72, 0.5, 0.3, 1)
LIGHT_FUR = (0.95, 0.86, 0.72, 1)
DARK_FUR = (0.42, 0.27, 0.16, 1)
NOSE = (0.08, 0.06, 0.06, 1)

# Pill-shaped body: BODY_WIDTH wide, from BODY_BOTTOM to BODY_TOP above the floor.
BODY_WIDTH = 0.4
BODY_DEPTH = 0.32
BODY_BOTTOM = 0.14
BODY_TOP = 0.84
BODY_CENTER = (BODY_BOTTOM + BODY_TOP) / 2
BODY_TUBE = (BODY_TOP - BODY_BOTTOM) / BODY_WIDTH - 1     # capsule tube length (in widths)
HEAD_Z = 1.04

# Paddle-hand key positions for a right-hander (x is mirrored for lefties).
# Every point is in front of the body (y > 0.2 when crossing the middle) and
# above table height (0.76) when reaching forward.
READY = Vec3(0.3, 0.24, 0.95)
WINDUP = Vec3(0.46, -0.02, 0.93)
BACKSWING = Vec3(0.5, -0.14, 0.92)
CONTACT = Vec3(0.42, 0.34, 1.0)
FOLLOW = Vec3(-0.04, 0.27, 1.2)
FREE_HAND = Vec3(-0.3, 0.16, 0.86)
SERVE_HOLD = Vec3(0.2, 0.34, 0.95)     # free hand reaches across, in front, to hold the serve
TOSS_LIFT = Vec3(0, 0.03, 0.22)        # how far the free hand pushes up when tossing
TOSS_TIME = 0.35

# Paddle heading (degrees) at those keys: face turned back, square, across.
PADDLE_H = {"ready": -15, "windup": -50, "back": -65, "contact": 0, "follow": 70}

SLIDE_SPEED = 7
SCALE = 0.85             # whole character size (1 = the measurements above)


def clamp01(t):
    return min(max(t, 0.0), 1.0)


def smooth(t):
    t = clamp01(t)
    return t * t * (3 - 2 * t)


def ease_out(t):
    t = clamp01(t)
    return 1 - (1 - t) ** 3


def approach(current, target, rate, dt):
    return current + (target - current) * min(1.0, rate * dt)


def catmull_rom(p0, p1, p2, p3, t):
    t2, t3 = t * t, t * t * t
    return (p1 * 2 + (p2 - p0) * t + (p0 * 2 - p1 * 5 + p2 * 4 - p3) * t2 + (p1 * 3 - p0 - p2 * 3 + p3) * t3) * 0.5


def swing_curve(back, contact, follow, u):
    """Smooth path back -> contact -> follow for u in 0..1 (contact at u = 0.5)."""
    before = back + (back - contact) * 0.5
    after = follow + (follow - contact) * 0.5
    if u < 0.5:
        return catmull_rom(before, back, contact, follow, u * 2)
    return catmull_rom(back, contact, follow, after, u * 2 - 1)


class Avatar:
    def __init__(self, parent, shirt, pants, hair, pos, facing=1, handed=1, dog=False):
        self.handed = handed
        self.dog = dog
        self.facing = facing
        self.root = parent.attachNewNode("avatar")
        self.root.setPos(*pos)
        self.root.setScale(SCALE)
        if facing < 0:
            self.root.setH(180)
        self.target_x = pos[0]
        self.time = 0.0
        self.step_phase = 0.0
        self.velocity = 0.0
        self.lean = 0.0

        self.windup = 0.0
        self.windup_target = 0.0
        self.swing_t = None
        self.swing_strength = 0.6
        self.swing_face = 0.0
        self.swing_start_pos = Vec3(READY)
        self.contact_z = CONTACT.z
        self.react = None
        self.hand_pos = Vec3(READY)
        self.serving = False          # free hand holds/tosses the ball until the serve is hit
        self.hold_amount = 0.0
        self.toss_t = None            # seconds since the toss started

        # Feet stay on the floor; everything else hangs off `body`.
        for side in (-1, 1):
            part(self.root, "sphere", DARK_FUR if dog else SHOE, pos=(0.09 * side, 0.05, 0.05),
                 scale=(0.13, 0.22, 0.1))
        self.body = self.root.attachNewNode("body")
        self.bean = self.body.attachNewNode("bean")
        self.bean.setPos(0, 0, BODY_CENTER)
        self.head = self.body.attachNewNode("head")
        self.head.setPos(0, 0, HEAD_Z)
        if dog:
            self.build_dog(shirt)
        else:
            self.team_parts = [part(self.bean, "capsule", shirt, scale=(BODY_WIDTH, BODY_DEPTH, BODY_WIDTH),
                                    cyl=BODY_TUBE)]
            # Shorts: a slightly wider short pill around the bottom of the body.
            part(self.bean, "capsule", pants, pos=(0, 0, BODY_BOTTOM - BODY_CENTER + 0.16),
                 scale=(BODY_WIDTH + 0.012, BODY_DEPTH + 0.012, BODY_WIDTH + 0.012), cyl=0.0)
            part(self.head, "sphere", SKIN, scale=(0.33, 0.31, 0.35))
            part(self.head, "sphere", hair, pos=(0, -0.02, 0.065), scale=(0.345, 0.32, 0.26))
            for side in (-1, 1):
                part(self.head, "sphere", EYE, pos=(0.06 * side, 0.145, 0.0), scale=(0.04, 0.03, 0.055))

        hand_color = FUR if dog else SKIN
        self.free_hand = self.body.attachNewNode("free_hand")
        part(self.free_hand, "sphere", hand_color, scale=0.11)
        self.paddle_hand = self.body.attachNewNode("paddle_hand")
        part(self.paddle_hand, "sphere", hand_color, scale=0.11)
        self.paddle = self.paddle_hand.attachNewNode("paddle")
        part(self.paddle, "box", HANDLE, pos=(0, 0, 0.07), scale=(0.028, 0.03, 0.1))
        # Blade: red rubber on the front, black rubber on the back, thin wood in between.
        part(self.paddle, "cylinder", HANDLE, pos=(0, 0, 0.19), scale=(0.158, 0.006, 0.168), axis="y")
        part(self.paddle, "cylinder", PADDLE_RED, pos=(0, 0.005, 0.19), scale=(0.16, 0.005, 0.17), axis="y")
        part(self.paddle, "cylinder", PADDLE_BLACK, pos=(0, -0.005, 0.19), scale=(0.16, 0.005, 0.17), axis="y")

    def build_dog(self, headband):
        """Fur body with a light belly, a sweatband, a snout, floppy ears and a tail."""
        part(self.bean, "capsule", FUR, scale=(BODY_WIDTH, BODY_DEPTH, BODY_WIDTH), cyl=BODY_TUBE)
        part(self.bean, "sphere", LIGHT_FUR, pos=(0, BODY_DEPTH * 0.32, -0.05), scale=(0.26, 0.12, 0.36))
        part(self.head, "sphere", FUR, scale=(0.34, 0.31, 0.32))
        # Sweatband in its team color, like a tennis player.
        self.team_parts = [part(self.head, "cylinder", headband, pos=(0, -0.005, 0.075), scale=(0.315, 0.29, 0.06))]
        part(self.head, "sphere", LIGHT_FUR, pos=(0, 0.14, -0.06), scale=(0.17, 0.15, 0.12))   # snout
        part(self.head, "sphere", NOSE, pos=(0, 0.215, -0.025), scale=(0.06, 0.045, 0.045))
        for side in (-1, 1):
            part(self.head, "sphere", EYE, pos=(0.075 * side, 0.135, 0.05), scale=(0.04, 0.03, 0.05))
        # Ears hang from a pivot at the top of the head so they can flap and droop.
        self.ears = []
        for side in (-1, 1):
            pivot = self.head.attachNewNode("ear")
            pivot.setPos(0.15 * side, -0.01, 0.08)
            part(pivot, "sphere", DARK_FUR, pos=(0.06 * side, 0, -0.1), scale=(0.08, 0.13, 0.22))
            self.ears.append((pivot, side))
        self.tail = self.body.attachNewNode("tail")
        self.tail.setPos(0, -BODY_DEPTH / 2 + 0.02, BODY_BOTTOM + 0.22)
        part(self.tail, "capsule", FUR, pos=(0, -0.07, 0.07), scale=(0.055, 0.055, 0.055), cyl=2.0).setP(-45)

    def animate_dog(self, happy, droop, hop):
        """Tail wags (faster when happy, tucked when sad); ears bounce with hops and droop when sad."""
        wag = math.sin(self.time * (9 + 16 * happy))
        self.tail.setHpr(wag * (25 + 20 * happy) * (1 - droop), -40 * droop, 0)
        for pivot, side in self.ears:
            flap = 12 * hop + 20 * happy * abs(math.sin(self.time * 12))
            pivot.setR(side * (8 + flap - 18 * droop))
            pivot.setP(25 * droop)

    # --- commands from the game ---------------------------------------------

    def swing(self, strength=0.6, face=0.0, contact_height=None):
        """Start a swing. `contact_height` (m above the floor) makes the hand meet the ball."""
        self.swing_t = 0.0
        self.swing_strength = strength
        self.swing_face = face
        self.swing_start_pos = Vec3(self.hand_pos)
        z = CONTACT.z if contact_height is None else contact_height / SCALE
        self.contact_z = min(max(z, 0.88), 1.45)

    def set_team_color(self, color):
        """Repaint the shirt (or the dog's sweatband)."""
        for np in self.team_parts:
            np.setColor(*color)

    def celebrate(self):
        self.react = ["celebrate", 0.0]

    def slump(self):
        self.react = ["slump", 0.0]

    def toss(self):
        """Free hand pushes the ball up, then comes back to the same holding spot."""
        self.toss_t = 0.0

    def set_windup(self, amount):
        self.windup_target = amount

    @property
    def x(self):
        return self.root.getX()

    def free_hand_world(self):
        """Free hand position in Panda3D world coordinates (for holding the serve)."""
        return self.free_hand.getPos(self.root.getParent())

    # --- per frame ------------------------------------------------------------

    def mirror(self, v):
        return Vec3(v.x * self.handed, v.y, v.z)

    def update(self, dt, look_at=None):
        self.time += dt

        # Hoppy side-steps toward the ball, leaning into the movement.
        x = self.root.getX()
        new_x = approach(x, self.target_x, SLIDE_SPEED, dt)
        self.velocity = approach(self.velocity, (new_x - x) / max(dt, 1e-4), 15, dt)
        self.root.setX(new_x)
        moving = min(abs(self.velocity) / 1.2, 1.0)
        self.step_phase += dt * (7 + 9 * moving)
        hop_wave = abs(math.sin(self.step_phase))
        hop = hop_wave * 0.07 * moving
        idle = math.sin(self.time * 4.0)
        lean_target = max(-12.0, min(12.0, -self.velocity * 5 * self.facing))
        self.lean = approach(self.lean, lean_target, 10, dt)

        # Paddle hand: rest pose blends from ready to wound up.
        self.windup = approach(self.windup, self.windup_target, 8, dt)
        rest = READY + (WINDUP - READY) * self.windup
        rest_h = PADDLE_H["ready"] + (PADDLE_H["windup"] - PADDLE_H["ready"]) * self.windup
        hand, paddle_h, paddle_p = rest + Vec3(0, 0, idle * 0.012), rest_h, 0.0
        self.hold_amount = approach(self.hold_amount, 1.0 if self.serving else 0.0, 10, dt)
        free = FREE_HAND + (SERVE_HOLD - FREE_HAND) * self.hold_amount + Vec3(0, 0, -idle * 0.01)
        if self.toss_t is not None:
            self.toss_t += dt
            k = self.toss_t / TOSS_TIME
            if k >= 1:
                self.toss_t = None
            else:
                free += TOSS_LIFT * math.sin(math.pi * k)     # up, then back down to the same spot
        twist = -10 * self.windup

        if self.swing_t is not None:
            self.swing_t += dt
            t = self.swing_t
            back_time = 0.07
            through_time = 0.2 - 0.08 * self.swing_strength
            hold, recover = 0.12, 0.32
            lift = self.swing_face * 0.12
            contact = Vec3(CONTACT.x, CONTACT.y, self.contact_z)
            follow = Vec3(FOLLOW.x, FOLLOW.y, max(FOLLOW.z, self.contact_z + 0.2) + lift * 2)
            if t < back_time:
                k = ease_out(t / back_time)
                hand = self.swing_start_pos + (BACKSWING - self.swing_start_pos) * k
                paddle_h = rest_h + (PADDLE_H["back"] - rest_h) * k
                twist = -18 * k
            elif t < back_time + through_time:
                u = ease_out((t - back_time) / through_time)
                hand = swing_curve(BACKSWING, contact, follow, u)
                paddle_h = (PADDLE_H["back"] + (PADDLE_H["contact"] - PADDLE_H["back"]) * min(u * 2, 1)
                            + (PADDLE_H["follow"] - PADDLE_H["contact"]) * max(u * 2 - 1, 0))
                paddle_p = -self.swing_face * 35 * math.sin(math.pi * u)
                twist = -18 + 48 * u
                free = FREE_HAND + Vec3(-0.06, -0.12, 0.08) * math.sin(math.pi * u)
            elif t < back_time + through_time + hold:
                hand, paddle_h, twist = follow, PADDLE_H["follow"], 30
            elif t < back_time + through_time + hold + recover:
                k = smooth((t - back_time - through_time - hold) / recover)
                hand = follow + (rest - follow) * k
                paddle_h = PADDLE_H["follow"] + (rest_h - PADDLE_H["follow"]) * k
                twist = 30 * (1 - k)
            else:
                self.swing_t = None
        self.hand_pos = hand

        # Body: hop with squash & stretch, lean, idle breathing, twist with the swing.
        body_z = hop
        squash = 1 + 0.06 * (hop_wave - 0.5) * moving + 0.015 * idle
        body_p = -6 - 5 * self.windup
        head_p = 0.0
        happy = droop = 0.0                 # for the dog's ears and tail

        if self.react:
            kind, t = self.react
            self.react[1] += dt
            if kind == "celebrate" and t < 0.9:
                jump = abs(math.sin(t * math.pi / 0.45))
                body_z += jump * 0.3
                squash = 1 + 0.08 * (jump - 0.5)
                happy = 1.0
                hand = Vec3(0.32, 0.1, 1.4 + 0.1 * jump)
                free = Vec3(-0.32, 0.1, 1.4 + 0.1 * jump)
                paddle_h = 0
            elif kind == "slump" and t < 1.2:
                droop = smooth(t / 0.3) * smooth((1.2 - t) / 0.3)
                head_p -= 30 * droop
                body_p += 10 * droop
                hand = hand + (Vec3(0.34, 0.15, 0.75) - hand) * droop
                free = free + (Vec3(-0.34, 0.15, 0.72) - free) * droop
            else:
                self.react = None

        h = self.handed
        self.body.setPos(0, 0, body_z)
        self.body.setHpr(twist * h, body_p, self.lean)
        self.bean.setScale(1 / math.sqrt(squash), 1 / math.sqrt(squash), squash)
        self.paddle_hand.setPos(self.mirror(hand))
        self.paddle.setHpr(paddle_h * h, paddle_p, -25 * h)
        self.free_hand.setPos(self.mirror(free))

        # Head turns to watch the ball.
        if look_at is not None:
            local = self.root.getRelativePoint(self.root.getParent(), Point3(look_at)) - Point3(0, 0, HEAD_Z)
            yaw = math.degrees(math.atan2(-local.x, max(local.y, 0.1))) - twist * h
            pitch = math.degrees(math.atan2(local.z, max(math.hypot(local.x, local.y), 0.1)))
            self.head.setHpr(max(-55, min(55, yaw)), max(-35, min(25, pitch)) + head_p, 0)
        else:
            self.head.setHpr(0, head_p, 0)
        if self.dog:
            self.animate_dog(happy, droop, hop_wave * moving)

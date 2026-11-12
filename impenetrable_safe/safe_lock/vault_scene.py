"""The vault animations for the safe game's screen.

A round vault door seen from the front, set in a stone wall, with sacks of
gold inside. Everything is drawn here with code (no pictures needed).

The door's front is split into 3 parts (top, middle, bottom): one per digit.

Scenes:
  waiting      far away from the door, torches flickering, "VAULT 687"
  get ready    walking up to the door
  code         close-up of the door + 3 boxes for the digits. A correct digit
               unlocks its part of the door: the bars slide in, the gold knobs
               light up, the side bolts pull back, sparkles fly
  wrong        shake + red flash, the bars slam back shut, "LOCKED", boxes erased
  open         the door swings open and shows the treasure
  closing      the door swings shut, the boxes are erased, back to waiting
"""

import math
import cv2
import numpy as np

VAULT_NUMBER = 687                       # on the plaque above the door

# The scene is drawn big, then the "camera" looks at part of it.
CW, CH = 1760, 1120
DOOR_C = (880, 500)                      # door center
DOOR_AX = (150, 290)                     # door half-width, half-height (gold rim included)
RIM = 16                                 # gold rim thickness
FRAME = 44                               # stone frame around the door
FLOOR_Y = 870
TORCHES = [(DOOR_C[0] - 430, 430), (DOOR_C[0] + 430, 430)]
N_ROWS = 15                              # rows of curly bars on the door

GOLD = (40, 190, 255)
BRIGHT_GOLD = (120, 230, 255)
DARK_GOLD = (20, 100, 150)
SILVER = (170, 172, 175)
IRON = (34, 33, 32)
RED = (40, 40, 230)
FONT = cv2.FONT_HERSHEY_TRIPLEX

WIDE = (DOOR_C[0], 560, CH)              # camera: center x, center y, how much height it sees
CLOSE = (DOOR_C[0], 495, 820)


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def ease(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)   # starts slow, speeds up, ends slow


def lerp(a, b, t):
    return a + (b - a) * t


class VaultScene:
    def __init__(self, width, height):
        self.w, self.h = width, height
        rng = np.random.default_rng(3)
        self.background = self._build_background(rng)
        self.door, self.door_mask, self.face_mask = self._build_door(rng)
        self.inside = self._build_inside(rng)

        bx, by = DOOR_C[0] - DOOR_AX[0], DOOR_C[1] - DOOR_AX[1]
        self.door_xy = (bx, by)
        face_ax, face_ay = DOOR_AX[0] - RIM, DOOR_AX[1] - RIM
        top, bottom = DOOR_C[1] - face_ay + 30, DOOR_C[1] + face_ay - 30
        self.rows = [top + (bottom - top) * i / (N_ROWS - 1) for i in range(N_ROWS)]
        third = (2 * face_ay) / 3
        self.band_centers = [(DOOR_C[0], int(DOOR_C[1] - face_ay + third * (i + 0.5))) for i in range(3)]

        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        r = np.sqrt(((xx - width / 2) / (width / 2)) ** 2 + ((yy - height / 2) / (height / 2)) ** 2)
        self.vignette = np.clip(1.15 - 0.5 * r ** 2, 0.3, 1.0)[..., None]

        g = np.zeros((300, 300), np.float32)
        cv2.circle(g, (150, 150), 40, 1.0, -1)
        self.torch_glow = (cv2.GaussianBlur(g, (0, 0), 45) * 3)[..., None] * np.float32([40, 110, 210])

        self.sparks = rng.uniform(-1, 1, (20, 2))
        self.dust = rng.uniform(0, 1, (40, 3))

    # ================= building the still parts (done once) =================

    def _build_background(self, rng):
        bg = np.zeros((CH, CW, 3), np.float32)
        # stone blocks in rows, every other row shifted half a block
        stone = np.float32([135, 155, 172])
        h, w = 64, 150
        for row, y in enumerate(range(0, FLOOR_Y, h)):
            shift = (w // 2) * (row % 2)
            for x in range(-shift, CW, w):
                shade = rng.uniform(0.82, 1.08)
                tint = rng.uniform(-6, 6, 3)
                bg[y + 3:y + h - 2, max(0, x + 3):x + w - 2] = stone * shade + tint
        # mottled texture so the stone isn't flat
        blotch = cv2.resize(rng.uniform(0.85, 1.12, (CH // 20, CW // 20)).astype(np.float32), (CW, CH),
                            interpolation=cv2.INTER_CUBIC)[..., None]
        bg = bg * blotch + rng.normal(0, 5, (CH, CW, 1)).astype(np.float32)
        bg[np.all(bg < 25, axis=2)] = (70, 78, 86)    # mortar between blocks

        # shadowed recess behind the door
        cv2.ellipse(bg, DOOR_C, (DOOR_AX[0] + FRAME + 70, DOOR_AX[1] + FRAME + 50), 0, 0, 360,
                    (60, 66, 72), -1, cv2.LINE_AA)
        # stone frame: wedge-shaped blocks around the door
        outer = (DOOR_AX[0] + FRAME, DOOR_AX[1] + FRAME)
        cv2.ellipse(bg, DOOR_C, outer, 0, 0, 360, (112, 126, 140), -1, cv2.LINE_AA)
        for deg in range(0, 360, 14):
            a = math.radians(deg)
            p0 = (int(DOOR_C[0] + DOOR_AX[0] * math.cos(a)), int(DOOR_C[1] + DOOR_AX[1] * math.sin(a)))
            p1 = (int(DOOR_C[0] + outer[0] * math.cos(a)), int(DOOR_C[1] + outer[1] * math.sin(a)))
            cv2.line(bg, p0, p1, (70, 78, 86), 3, cv2.LINE_AA)
        cv2.ellipse(bg, DOOR_C, outer, 0, 0, 360, (60, 66, 72), 4, cv2.LINE_AA)

        # floor: dark metal grate going off into the distance
        floor = bg[FLOOR_Y:]
        floor[:] = (36, 38, 42)
        vx, vy = DOOR_C[0], DOOR_C[1] - 100           # where the lines meet
        for x in range(-2000, CW + 2000, 90):
            cv2.line(bg, (vx + (x - vx) * (FLOOR_Y - vy) // (CH - vy), FLOOR_Y), (x, CH), (22, 23, 26), 3)
        for i in range(12):
            y = int(FLOOR_Y + (CH - FLOOR_Y) * (i / 11) ** 1.6)
            cv2.line(bg, (0, y), (CW, y), (22, 23, 26), 3)
        cv2.line(bg, (0, FLOOR_Y), (CW, FLOOR_Y), (55, 60, 66), 4)

        # brass plaque with the vault number, above the door
        px, py = DOOR_C[0], DOOR_C[1] - DOOR_AX[1] - FRAME - 50
        cv2.rectangle(bg, (px - 80, py - 30), (px + 80, py + 30), DARK_GOLD, -1)
        cv2.rectangle(bg, (px - 74, py - 24), (px + 74, py + 24), GOLD, -1)
        cv2.rectangle(bg, (px - 74, py - 24), (px + 74, py + 24), DARK_GOLD, 2)

        # torch holders
        for tx, ty in TORCHES:
            cv2.rectangle(bg, (tx - 5, ty), (tx + 5, ty + 70), (30, 30, 32), -1)
            pts = np.int32([[tx - 22, ty - 8], [tx + 22, ty - 8], [tx + 12, ty + 12], [tx - 12, ty + 12]])
            cv2.fillPoly(bg, [pts], (40, 40, 44), cv2.LINE_AA)

        # light: brighter around the door, darker at the edges
        yy, xx = np.mgrid[0:CH, 0:CW].astype(np.float32)
        r = np.sqrt(((xx - DOOR_C[0]) / 900) ** 2 + ((yy - DOOR_C[1]) / 700) ** 2)
        bg *= np.clip(1.15 - 0.6 * r, 0.35, 1.15)[..., None]
        bg = np.clip(bg, 0, 255).astype(np.uint8)
        label = str(VAULT_NUMBER)                     # engraved on the plaque
        tw = cv2.getTextSize(label, FONT, 1.4, 3)[0][0]
        cv2.putText(bg, label, (px - tw // 2, py + 16), FONT, 1.4, (20, 50, 80), 3, cv2.LINE_AA)
        return bg

    def _build_door(self, rng):
        ax, ay = DOOR_AX
        w, h = 2 * ax + 2, 2 * ay + 2
        c = (ax + 1, ay + 1)
        door = np.zeros((h, w, 3), np.float32)
        # gold rim, shaded like a rounded edge
        for k in range(RIM):
            shade = 0.55 + 0.6 * math.sin(math.pi * k / RIM)
            cv2.ellipse(door, c, (ax - k, ay - k), 0, 0, 360, tuple(v * shade for v in GOLD), 2, cv2.LINE_AA)
        # dark iron face with a little texture and light in the middle
        face = np.zeros((h, w), np.float32)
        cv2.ellipse(face, c, (ax - RIM, ay - RIM), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        light = np.clip(1.3 - np.sqrt(((xx - c[0]) / ax) ** 2 + ((yy - c[1]) / ay) ** 2) * 0.6, 0.6, 1.3)
        iron = (np.float32(IRON) + rng.normal(0, 4, (h, w, 1))) * light[..., None]
        door = door * (1 - face[..., None]) + iron * face[..., None]
        # rivets around the rim
        for deg in range(0, 360, 12):
            a = math.radians(deg)
            p = (int(c[0] + (ax - RIM / 2) * math.cos(a)), int(c[1] + (ay - RIM / 2) * math.sin(a)))
            cv2.circle(door, p, 4, DARK_GOLD, -1, cv2.LINE_AA)
            cv2.circle(door, (p[0] - 1, p[1] - 1), 2, BRIGHT_GOLD, -1, cv2.LINE_AA)
        # center spine
        cv2.rectangle(door, (c[0] - 7, RIM + 6), (c[0] + 7, h - RIM - 6), (25, 25, 25), -1)
        cv2.rectangle(door, (c[0] - 5, RIM + 8), (c[0] + 5, h - RIM - 8), SILVER, -1)
        mask = np.zeros((h, w), np.float32)
        cv2.ellipse(mask, c, (ax, ay), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        return np.clip(door, 0, 255).astype(np.uint8), mask[..., None], face[..., None]

    @staticmethod
    def _sack(img, x, y, size, rng):
        """One sack of gold: a round cloth bag, tied at the top, a coin peeking out."""
        cloth = np.float32([55, 100, 145]) * rng.uniform(0.85, 1.1)     # brown sack cloth
        dark = tuple(float(v) for v in cloth * 0.55)
        w, h = int(size * 0.62), int(size * 0.5)
        cv2.ellipse(img, (x, y), (w + 2, h + 2), 0, 0, 360, dark, -1, cv2.LINE_AA)      # outline
        cv2.ellipse(img, (x, y), (w, h), 0, 0, 360, tuple(float(v) for v in cloth), -1, cv2.LINE_AA)
        cv2.ellipse(img, (x - w // 3, y - h // 3), (w // 3, h // 4), -20, 0, 360,
                    tuple(float(min(255, v * 1.25)) for v in cloth), -1, cv2.LINE_AA)   # light on it
        # fold lines running down from the tie
        for fx in (-w // 3, w // 4):
            cv2.ellipse(img, (x + fx, y - h // 6), (max(2, w // 8), int(h * 0.7)), 0, 270, 360 + 60 * (fx > 0),
                        dark, 2, cv2.LINE_AA)
        # the neck and the string
        neck = (x, y - h)
        cv2.ellipse(img, (neck[0], neck[1] - 6), (w // 3, int(size * 0.18)), 0, 0, 360,
                    tuple(float(v) for v in cloth * 0.9), -1, cv2.LINE_AA)
        cv2.line(img, (neck[0] - w // 3, neck[1] + 2), (neck[0] + w // 3, neck[1] + 2), (40, 60, 80), 3, cv2.LINE_AA)
        # a gold coin peeking out of the top
        cv2.circle(img, (neck[0] + 3, neck[1] - int(size * 0.25)), max(3, size // 9), GOLD, -1, cv2.LINE_AA)
        cv2.circle(img, (neck[0] + 3, neck[1] - int(size * 0.25)), max(3, size // 9), DARK_GOLD, 1, cv2.LINE_AA)

    @staticmethod
    def _coin(img, x, y, r, rng):
        shade = rng.uniform(0.75, 1.1)
        cv2.ellipse(img, (x, y), (r, int(r * 0.55)), 0, 0, 360, DARK_GOLD, -1, cv2.LINE_AA)
        cv2.ellipse(img, (x, y - 1), (r - 1, int(r * 0.5)), 0, 0, 360,
                    tuple(float(min(255, v * shade)) for v in GOLD), -1, cv2.LINE_AA)
        cv2.ellipse(img, (x - r // 3, y - 2), (max(1, r // 3), max(1, r // 6)), 0, 0, 360, BRIGHT_GOLD, -1)

    def _build_inside(self, rng):
        """What's behind the door: stone shelves full of sacks of gold, coins on the floor."""
        ax, ay = DOOR_AX
        w, h = 2 * ax + 2, 2 * ay + 2
        inside = np.full((h, w, 3), (40, 48, 58), np.float32)                # back wall, in shadow
        for row, y in enumerate(range(0, h, 40)):                             # faint stone blocks
            for x in range(-40 * (row % 2), w, 80):
                cv2.rectangle(inside, (x + 2, y + 2), (x + 78, y + 38), (48, 57, 68), -1)
        shelves = [int(h * 0.36), int(h * 0.64)]
        for sy in shelves:
            cv2.rectangle(inside, (0, sy), (w, sy + 14), (70, 85, 100), -1)
            cv2.line(inside, (0, sy + 14), (w, sy + 14), (25, 30, 36), 3)
        # sacks sitting on each shelf (back row smaller, front row bigger)
        for sy in shelves + [h - 30]:
            for size, back in [(46, True), (58, False)]:
                step = int(size * 1.35)
                for x in range(int(rng.uniform(10, 30)), w + step, step):
                    jitter = int(rng.uniform(-8, 8))
                    base = sy - (12 if back else 0)
                    self._sack(inside, x + jitter + (step // 2 if back else 0),
                               base - int(size * 0.5), size, rng)
        # coins spilled on the floor and on the shelves
        for _ in range(140):
            x = int(rng.uniform(10, w - 10))
            y = int(h - 10 - abs(rng.normal(0, 22)))
            self._coin(inside, x, y, int(rng.uniform(5, 9)), rng)
        for sy in shelves:
            for _ in range(12):
                self._coin(inside, int(rng.uniform(10, w - 10)), sy + 2, 5, rng)
        # warm golden glow inside
        glow = np.zeros((h, w), np.float32)
        cv2.ellipse(glow, (w // 2, int(h * 0.6)), (ax, ay // 2), 0, 0, 360, 1.0, -1)
        glow = cv2.GaussianBlur(glow, (0, 0), 50)[..., None]
        return np.clip(inside + glow * np.float32(GOLD) * 0.25, 0, 255).astype(np.uint8)

    # ================= the moving parts (every frame) =================

    def _door_now(self, unlocked, glow):
        """The door with its curly bars. unlocked[i] (0-1) = how far part i's bars
        have slid in; glow[i] = how bright part i's gold knobs shine."""
        door = self.door.copy()
        h, w = door.shape[:2]
        cx = w // 2
        cy = h // 2
        face_ax, face_ay = DOOR_AX[0] - RIM, DOOR_AX[1] - RIM
        shine = np.zeros((h, w), np.float32)
        gap = self.rows[1] - self.rows[0]
        for r, row_y in enumerate(self.rows):
            part = min(2, r * 3 // N_ROWS)
            y = row_y - (DOOR_C[1] - DOOR_AX[1])                 # door-local y
            half = face_ax * math.sqrt(max(0.0, 1 - ((y - cy) / face_ay) ** 2)) - 10
            if half < 30:
                continue
            slide = 1 - 0.72 * unlocked[part]                   # bars slide toward the middle
            # a long bar on the row, and a shorter one between rows (wiggling the other way)
            for length, dy, phase in [((half - 10) * slide, 0.0, r * 1.3),
                                      ((half - 10) * 0.62 * slide, gap * 0.5, r * 1.3 + math.pi)]:
                if r == N_ROWS - 1 and dy:
                    continue
                for side in (-1, 1):
                    pts = []
                    for k in range(25):
                        t = k / 24
                        x = cx + side * (10 + t * length)
                        yy = y + dy + gap * 0.2 * math.sin(t * 2 * math.pi + phase) * side
                        pts.append((x, yy))
                    pts = np.int32(pts)
                    thick = 4 if dy == 0 else 3
                    cv2.polylines(door, [pts], False, (18, 18, 18), thick + 4, cv2.LINE_AA)
                    cv2.polylines(door, [pts], False, SILVER, thick, cv2.LINE_AA)
                    cv2.polylines(door, [pts - [0, 1]], False, (225, 228, 230), 1, cv2.LINE_AA)
                    # little curls along the bar
                    for k, curl in [(8, 7), (16, 6)]:
                        mx, my = pts[k]
                        up = side * (1 if k == 8 else -1)
                        cv2.ellipse(door, (int(mx), int(my - up * curl)), (curl, curl), 0, 90 * up,
                                    90 * up + 270, SILVER, 2, cv2.LINE_AA)
                    # gold knobs: a big one at the end, small ones along the bar
                    knobs = [(pts[-1], 7 if dy == 0 else 5), (pts[12], 4)]
                    if dy == 0:
                        knobs.append((pts[20], 4))
                    for (kx, ky), size in knobs:
                        bright = clamp(0.55 + 0.45 * glow[part])
                        cv2.circle(door, (int(kx), int(ky)), size + 1, DARK_GOLD, -1, cv2.LINE_AA)
                        cv2.circle(door, (int(kx), int(ky)), size,
                                   tuple(int(v * bright) for v in BRIGHT_GOLD), -1, cv2.LINE_AA)
                        cv2.circle(door, (int(kx) - 1, int(ky) - 1), max(1, size // 3), (230, 250, 255), -1)
                        if glow[part] > 0:
                            cv2.circle(shine, (int(kx), int(ky)), size + 5, glow[part], -1)
        if shine.any():
            shine = cv2.GaussianBlur(shine, (0, 0), 9)[..., None]
            door = np.clip(door + shine * np.float32(GOLD) * 1.6, 0, 255).astype(np.uint8)
        return door

    def _bolts(self, canvas, unlocked):
        """Metal bolts on both sides of the door, sticking into the stone frame.
        They pull back into the door when their part is unlocked."""
        for i, (_, by) in enumerate(self.band_centers):
            for dy in (-34, 34):
                y = by + dy
                half = DOOR_AX[0] * math.sqrt(max(0.0, 1 - ((y - DOOR_C[1]) / DOOR_AX[1]) ** 2))
                out = int(30 * (1 - unlocked[i]))
                for side in (-1, 1):
                    edge = int(DOOR_C[0] + side * (half - 8))
                    x0, x1 = sorted((edge, edge + side * (out + 8)))
                    cv2.rectangle(canvas, (x0, y - 9), (x1, y + 9), (25, 25, 25), -1)
                    cv2.rectangle(canvas, (x0 + 1, y - 7), (x1 - 1, y + 7), (190, 192, 195), -1)

    def _scene(self, now, unlocked, glow, swing=0.0):
        canvas = self.background.copy()
        bx, by = self.door_xy
        h, w = self.door.shape[:2]
        spot = canvas[by:by + h, bx:bx + w]
        door = self._door_now(unlocked, glow)
        if swing <= 0:
            spot[:] = spot * (1 - self.door_mask) + door * self.door_mask
            self._bolts(canvas, unlocked)
        else:
            spot[:] = spot * (1 - self.door_mask) + self.inside * self.door_mask
            nw = int(w * math.cos(swing * math.pi / 2))     # turning on its left hinge: thinner and thinner
            if nw >= 3:                                     # fully open: the door is out of sight
                d = cv2.resize(door, (nw, h)) * (1 - 0.5 * swing)
                m = cv2.resize(self.door_mask, (nw, h))[..., None]
                spot[:, :nw] = spot[:, :nw] * (1 - m) + d * m
        # torches
        for i, (tx, ty) in enumerate(TORCHES):
            f = 0.85 + 0.1 * math.sin(now * 9 + i * 2) + 0.05 * math.sin(now * 23 + i)
            area = canvas[ty - 180:ty + 120, tx - 150:tx + 150]
            area[:] = np.clip(area + self.torch_glow * f, 0, 255)
            cv2.ellipse(canvas, (tx, ty - 28), (int(15 * f), int(34 * f)), 0, 0, 360, (0, 110, 240), -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (tx, ty - 22), (int(9 * f), int(22 * f)), 0, 0, 360, (40, 210, 255), -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (tx, ty - 16), (4, int(10 * f)), 0, 0, 360, (220, 250, 255), -1, cv2.LINE_AA)
        return canvas

    # ================= camera =================

    def _camera(self, cx, cy, view_h, shake=(0.0, 0.0)):
        z = self.h / view_h
        return np.float32([[z, 0, self.w / 2 - z * cx + shake[0]], [0, z, self.h / 2 - z * cy + shake[1]]])

    def _shoot(self, canvas, cam, brightness=1.0):
        img = cv2.warpAffine(canvas, cam, (self.w, self.h), flags=cv2.INTER_AREA,
                             borderMode=cv2.BORDER_REPLICATE)
        return np.clip(img * self.vignette * brightness, 0, 255).astype(np.uint8)

    @staticmethod
    def _to_screen(cam, x, y):
        return int(cam[0, 0] * x + cam[0, 2]), int(cam[1, 1] * y + cam[1, 2])

    # ================= things drawn on top =================

    def _sparkles(self, img, cam, now, entered_at):
        z = cam[0, 0]
        for i, at in enumerate(entered_at[:3]):
            t = now - at
            if t >= 1.2:
                continue
            x, y = self._to_screen(cam, *self.band_centers[i])
            fade = 1 - t / 1.2
            ring = int((40 + t * 220) * z)
            cv2.ellipse(img, (x, y), (ring, int(ring * 0.6)), 0, 0, 360,
                        tuple(int(c * fade) for c in GOLD), max(1, int(4 * fade)), cv2.LINE_AA)
            for sx, sy in self.sparks:
                px = int(x + sx * t * 300 * z)
                py = int(y + sy * t * 180 * z + 120 * t * t)
                cv2.circle(img, (px, py), max(1, int(4 * fade)), (200, 245, 255), -1, cv2.LINE_AA)

    def _boxes(self, img, digits, now, entered_at, color=GOLD, alpha=1.0):
        """The 3 digit boxes at the bottom. digits = what to show in each box."""
        bw, bh, gap = 70, 82, 20
        x0 = (self.w - (3 * bw + 2 * gap)) // 2
        y0 = self.h - bh - 16
        layer = img.copy()
        for i in range(3):
            x = x0 + i * (bw + gap)
            cv2.rectangle(layer, (x, y0), (x + bw, y0 + bh), (20, 20, 22), -1)
            filled = i < len(digits)
            pop = 0.0
            if filled and i < len(entered_at):
                pop = 1 - clamp((now - entered_at[i]) / 0.4)
            border = BRIGHT_GOLD if pop > 0 else (color if filled else DARK_GOLD)
            cv2.rectangle(layer, (x, y0), (x + bw, y0 + bh), border, 3 + int(3 * pop))
            if filled:
                size = 2.0 + 0.8 * pop
                tw, th = cv2.getTextSize(digits[i], FONT, size, 4)[0]
                tx, ty = x + (bw - tw) // 2, y0 + (bh + th) // 2
                cv2.putText(layer, digits[i], (tx, ty), FONT, size, (0, 0, 0), 9, cv2.LINE_AA)
                cv2.putText(layer, digits[i], (tx, ty), FONT, size, color, 4, cv2.LINE_AA)
        img[:] = cv2.addWeighted(layer, alpha, img, 1 - alpha, 0)

    def _title(self, img, msg, y, size, color, thick=2):
        tw = cv2.getTextSize(msg, FONT, size, thick)[0][0]
        x = (self.w - tw) // 2
        cv2.putText(img, msg, (x, y), FONT, size, (0, 0, 0), thick + 6, cv2.LINE_AA)
        cv2.putText(img, msg, (x, y), FONT, size, color, thick, cv2.LINE_AA)

    # ================= pick the scene for what the safe is doing =================

    def render(self, safe, now, ready_s):
        t = now - safe.since
        entered_at = safe.entered_at
        none = [0.0, 0.0, 0.0]

        def unlocked_parts():
            return [ease((now - entered_at[i]) / 0.7) if i < len(entered_at) else 0.0 for i in range(3)]

        def glowing_parts(next_part):
            g = []
            for i in range(3):
                if i < len(entered_at):
                    since = now - entered_at[i]
                    g.append(0.7 + 0.15 * math.sin(now * 3 + i) + 1.5 * max(0.0, 1 - since / 0.8))
                elif i == next_part:
                    g.append(0.15 + 0.12 * math.sin(now * 5))
                else:
                    g.append(0.0)
            return g

        if safe.state == "ready":                                  # walking up to the door
            p = ease(t / ready_s)
            bob = math.sin(t * 2 * math.pi * 1.8) * 12 * (1 - p)
            sway = math.sin(t * math.pi * 1.8) * 8 * (1 - p)
            cam = self._camera(lerp(WIDE[0], CLOSE[0], p) + sway, lerp(WIDE[1], CLOSE[1], p) + bob,
                               lerp(WIDE[2], CLOSE[2], p))
            img = self._shoot(self._scene(now, none, none), cam, clamp(t / 0.7) * lerp(0.75, 1.0, p))
            self._boxes(img, "", now, [], alpha=clamp((t - ready_s + 1.0) / 0.8))
            return img

        if safe.state == "digit":                                  # entering the code
            cam = self._camera(*CLOSE)
            next_part = -1 if safe.need_clear else len(entered_at)
            img = self._shoot(self._scene(now, unlocked_parts(), glowing_parts(next_part)), cam)
            self._sparkles(img, cam, now, entered_at)
            self._boxes(img, "".join(safe.entered), now, entered_at)
            return img

        if safe.state == "open":                                   # the door swings open
            swing = ease((t - 0.6) / 1.8)
            push = ease((t - 2.6) / 3.0)
            cam = self._camera(CLOSE[0], lerp(CLOSE[1], DOOR_C[1] + 40, push), lerp(CLOSE[2], 640, push))
            img = self._shoot(self._scene(now, [1.0] * 3, glowing_parts(-1), swing), cam)
            self._sparkles(img, cam, now, entered_at)
            if swing > 0:   # gold dust floating out of the vault
                x0, y0 = self._to_screen(cam, DOOR_C[0] - DOOR_AX[0], DOOR_C[1] - DOOR_AX[1])
                x1, y1 = self._to_screen(cam, DOOR_C[0] + DOOR_AX[0], DOOR_C[1] + DOOR_AX[1])
                for dx, speed, size in self.dust:
                    y = y1 - ((now * (25 + 50 * speed) + dx * 500) % (y1 - y0))
                    x = x0 + dx * (x1 - x0) + 10 * math.sin(now * 2 + dx * 10)
                    cv2.circle(img, (int(x), int(y)), 1 + int(2 * size * swing), (120, 230, 255), -1, cv2.LINE_AA)
            if t > 2.0:
                self._title(img, "OPEN", 70, 2.2, GOLD, thick=4)
            self._boxes(img, "".join(safe.entered), now, entered_at)
            return img

        # ---- locked ----
        if safe.tone == "bad":                                     # a digit was wrong
            amount = 26 * (1 - clamp(t / 0.6))
            cam = self._camera(*CLOSE, shake=(amount * math.sin(t * 70), amount * math.cos(t * 55)))
            snap = 1 - ease(t / 0.15)                              # unlocked bars slam back
            parts = [snap if i < len(entered_at) else 0.0 for i in range(3)]
            img = self._shoot(self._scene(now, parts, none), cam, lerp(1.0, 0.75, clamp(t / 0.8)))
            flash = 0.55 * (1 - clamp(t / 0.5))
            if flash > 0:
                img[:] = cv2.addWeighted(np.full_like(img, RED), flash, img, 1 - flash, 0)
            k = ease((t - 0.25) / 0.2)
            if k > 0:
                self._title(img, "LOCKED", self.h // 2 + 25, lerp(4.0, 2.4, k), RED, thick=6)
            # the boxes flash red, then get erased
            self._boxes(img, "".join(safe.entered) if t < 0.7 else "", now, [], color=RED)
            return img

        if safe.prev == "open" and t < 2.0:                        # closing the door
            swing = 1 - ease(t / 1.2)
            parts = [1 - ease((t - 1.2) / 0.3)] * 3                # then the bars slam locked
            img = self._shoot(self._scene(now, parts, none, swing), self._camera(*CLOSE),
                              1 - clamp((t - 1.5) / 0.5))
            self._boxes(img, "".join(safe.entered) if t < 0.8 else "", now, [], alpha=1 - clamp((t - 1.3) / 0.4))
            return img

        # waiting, far from the door
        fade = clamp((t - 2.0) / 0.6) if safe.prev == "open" else 1.0
        img = self._shoot(self._scene(now, none, none), self._camera(*WIDE), 0.75 * fade)
        self._title(img, f"VAULT {VAULT_NUMBER}", self.h - 90, 2.0, GOLD, thick=3)
        self._title(img, "Only those who know the code may enter", self.h - 45, 0.75, (225, 225, 225))
        return img

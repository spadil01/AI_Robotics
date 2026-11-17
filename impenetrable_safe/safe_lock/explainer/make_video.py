"""Make a silent stop-motion style video that explains the safe's neural network.

Everything shown is real: a real hand picture from data/, the real trained
network (model.pt), its real pattern maps and its real answer. Paper cut-out
look: cards jiggle a little every drawing, and things move in small jumps.

Run:  python explainer/make_video.py
Out:  explainer/how_it_works.mp4  (+ the scene start times, for the narration)
"""

import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
LOCK = HERE.parent
sys.path.insert(0, str(LOCK))
from finger_net import LABELS, load, prepare, to_tensor  # noqa: E402
from train import shake_up  # noqa: E402

W, H = 1280, 720
FPS = 12              # video frames per second
HOLD = 2              # each drawing is shown for 2 frames ("on twos", like real stop motion)
OUT = HERE / "how_it_works.mp4"
EXAMPLE = LOCK / "data" / "ryder" / "3"      # the example hand: 3 fingers
NEW_PERSON = LOCK / "data" / "khabir" / "4"
FINDER = 8            # which 1st-layer pattern finder to show (8 = an edge finder)

FONT_PATH = "/System/Library/Fonts/MarkerFelt.ttc"
INK = (40, 35, 30)
RED = (200, 50, 40)
BLUE = (40, 90, 170)
GREEN = (40, 140, 60)
GOLD = (210, 150, 30)
PAPER = (232, 220, 196)


def font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size, index=1)
    except OSError:
        return ImageFont.load_default()


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def lerp(a, b, t):
    return a + (b - a) * t


def to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def colorize(m, size):
    """One pattern map (2D numbers) -> a picture: bright = the pattern is there."""
    m = m - m.min()
    m = m / (m.max() + 1e-6)
    img = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return to_pil(cv2.resize(img, size, interpolation=cv2.INTER_NEAREST))


# ---------------------------------------------------------------- real data

def gather():
    """Run the real network on the example picture and keep everything we show."""
    d = {}
    f = sorted(EXAMPLE.glob("*.jpg"))[30]
    d["photo"] = cv2.imread(str(f))                       # 256x144, as collected
    d["small"] = prepare(d["photo"])                      # 128x72, what the network eats
    net = load(LOCK / "model.pt")
    x = to_tensor([d["small"]])
    maps = []
    with torch.no_grad():
        first = net.features[0][:3](x)[0].numpy()         # 1st pattern finders (before shrinking)
        h = x
        for blk in net.features:
            h = blk(h)
            maps.append(h[0].numpy())
        pooled = h.mean(dim=(2, 3))[0].numpy()
        chances = torch.softmax(net.head(h), 1)[0].numpy()
    d["first"], d["maps"], d["pooled"], d["chances"] = first, maps, pooled, chances
    # the pattern finder we show: #8 finds edges (outlines of fingers and of the board)
    d["finder"] = FINDER
    # the 4x3 pixels we zoom into in scene 3: where colors change most (edge of a finger)
    sm = d["small"].astype(np.float32)
    best, d["patch"] = -1, (70, 30)
    for y in range(10, 60, 2):
        for x in range(20, 110, 2):
            v = sm[y:y + 3, x:x + 4].reshape(-1, 3).std(0).sum()
            if v > best:
                best, d["patch"] = v, (x, y)
    w = net.features[0][0].weight[d["finder"]].detach().numpy()   # (3 colors, 3, 3)
    d["finder_w"] = w.mean(0)
    torch.manual_seed(4)
    aug = shake_up(to_tensor([d["small"]] * 6))
    d["aug"] = [((a.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8) for a in aug]
    d["new_person"] = cv2.imread(str(sorted(NEW_PERSON.glob("*.jpg"))[40]))
    d["others"] = [cv2.imread(str(sorted((LOCK / "data" / p / lab).glob("*.jpg"))[k]))
                   for k, (p, lab) in enumerate([("sergio", "1"), ("will", "5"), ("victor", "2"),
                                                 ("ryder", "0"), ("khabir", "4"), ("sergio", "none"),
                                                 ("will", "3"), ("victor", "5")])]
    return d


# ---------------------------------------------------------------- paper + cards

class Page:
    """One drawing. Every card and word gets its own tiny random wobble."""

    def __init__(self, backgrounds, drawing):
        self.img = backgrounds[drawing % len(backgrounds)].copy()
        self.drawing = drawing
        self.draw = ImageDraw.Draw(self.img)

    def wobble(self, key, amount=2.0):
        rng = np.random.default_rng(abs(hash((key, self.drawing))) % (2 ** 32))
        return rng.uniform(-amount, amount, 3)

    def card(self, pic, cx, cy, width, key, angle=0.0, border=8, shadow=True):
        pic = pic.convert("RGB")
        h = max(1, round(width * pic.height / pic.width))
        pic = pic.resize((max(1, int(width)), h), Image.NEAREST if pic.width < width / 2 else Image.LANCZOS)
        framed = Image.new("RGB", (pic.width + 2 * border, pic.height + 2 * border), (250, 248, 242))
        framed.paste(pic, (border, border))
        dx, dy, da = self.wobble(key)
        rgba = framed.convert("RGBA").rotate(angle + da * 0.4, resample=Image.BICUBIC, expand=True)
        x, y = int(cx - rgba.width / 2 + dx), int(cy - rgba.height / 2 + dy)
        if shadow:
            sh = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
            sh.putalpha(rgba.getchannel("A").point(lambda a: a * 0.35))
            sh = sh.filter(ImageFilter.GaussianBlur(5))
            self.img.paste((30, 20, 10), (x + 6, y + 8), sh)
        self.img.paste(rgba, (x, y), rgba)
        return x, y, rgba.width, rgba.height

    def text(self, s, x, y, size, color=INK, key=None, anchor="la"):
        dx, dy, _ = self.wobble(key or s, 1.5)
        self.draw.text((x + dx, y + dy), s, font=font(size), fill=color, anchor=anchor)

    def title(self, s, p=1.0):
        """Scene title, written on letter by letter."""
        shown = s[:max(1, int(len(s) * min(1.0, p * 2.5)))]
        self.text(shown, 50, 40, 46, INK, key="title")
        self.draw.line([(50, 100), (50 + 18 * len(shown), 100)], fill=RED, width=4)

    def arrow(self, x0, y0, x1, y1, color=INK, key="arrow"):
        dx, dy, _ = self.wobble(key, 1.5)
        x0, y0, x1, y1 = x0 + dx, y0 + dy, x1 + dx, y1 + dy
        self.draw.line([(x0, y0), (x1, y1)], fill=color, width=6)
        a = math.atan2(y1 - y0, x1 - x0)
        for side in (-1, 1):
            self.draw.line([(x1, y1), (x1 - 22 * math.cos(a + side * 0.5), y1 - 22 * math.sin(a + side * 0.5))],
                           fill=color, width=6)

    def bar(self, x, y, w, h, frac, color, key):
        dx, dy, _ = self.wobble(key, 1.0)
        self.draw.rectangle([x + dx, y + dy, x + w + dx, y + h + dy], outline=INK, width=3)
        if frac > 0:
            self.draw.rectangle([x + dx + 3, y + dy + 3, x + dx + 3 + (w - 6) * frac, y + h + dy - 3], fill=color)


def make_backgrounds(n=3):
    """A few slightly different sheets of paper (switching between them = stop-motion flicker)."""
    out = []
    for k in range(n):
        rng = np.random.default_rng(k)
        base = np.ones((H, W, 3), np.float32) * np.float32(PAPER)
        blotch = cv2.resize(rng.uniform(0.94, 1.04, (H // 40, W // 40)).astype(np.float32), (W, H),
                            interpolation=cv2.INTER_CUBIC)[..., None]
        grain = rng.normal(0, 4, (H, W, 1)).astype(np.float32)
        img = np.clip(base * blotch + grain, 0, 255).astype(np.uint8)
        for _ in range(60):   # paper fibers
            x, y = rng.integers(0, W), rng.integers(0, H)
            a = rng.uniform(0, math.pi)
            ln = rng.integers(8, 30)
            cv2.line(img, (int(x), int(y)), (int(x + ln * math.cos(a)), int(y + ln * math.sin(a))),
                     (200, 186, 160), 1, cv2.LINE_AA)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
        img = (img * np.clip(1.08 - 0.5 * r, 0.75, 1.0)[..., None]).astype(np.uint8)
        out.append(Image.fromarray(img))
    return out


# ---------------------------------------------------------------- scenes
# Each scene: (seconds, function(page, p, d)) where p goes 0 -> 1 through the scene.

def s_title(pg, p, d):
    t = "How the safe counts your fingers"
    shown = t[:max(1, int(len(t) * min(1.0, p * 2.2)))]
    pg.text(shown, W // 2, 250, 62, INK, key="t1", anchor="mm")
    if p > 0.5:
        pg.text("a neural network, one step at a time", W // 2, 330, 36, RED, key="t2", anchor="mm")
    if p > 0.65:
        for i, pic in enumerate(d["others"][:5]):
            k = ease((p - 0.65 - i * 0.04) / 0.15)
            if k > 0:
                pg.card(to_pil(pic), 200 + i * 220, lerp(800, 520, k), 190, f"o{i}", angle=(i - 2) * 4)


def s_camera(pg, p, d):
    pg.title("1. The camera takes a picture", p)
    k = ease(p / 0.5)
    pg.card(to_pil(d["photo"]), lerp(1500, W // 2, k), 400, 720, "photo", angle=lerp(12, -2, k))


def s_shrink(pg, p, d):
    pg.title("2. Shrink it to 128 x 72 pixels", p)
    k = ease(p / 0.4)
    if p < 0.55:
        pg.card(to_pil(d["photo"]), lerp(W // 2, 300, k), 400, lerp(720, 256, k), "photo", angle=-2)
    else:
        pg.card(to_pil(d["small"]), 300, 400, 256, "small", angle=-2)
        pg.text("real size", 300, 520, 30, INK, key="rs", anchor="mm")
        kk = ease((p - 0.55) / 0.3)
        if kk > 0:
            big = cv2.resize(d["small"], (640, 360), interpolation=cv2.INTER_NEAREST)
            for x in range(0, 640, 5):
                big[:, x] = (big[:, x] * 0.8).astype(np.uint8)
            for y in range(0, 360, 5):
                big[y, :] = (big[y, :] * 0.8).astype(np.uint8)
            pg.arrow(450, 400, 560, 400, key="a1")
            pg.card(to_pil(big), lerp(1400, 900, kk), 400, 600, "big", angle=1.5)
            pg.text("zoomed in: tiny squares", 900, 620, 30, BLUE, key="zz", anchor="mm")


def s_numbers(pg, p, d):
    pg.title("3. Every pixel is just 3 numbers", p)
    big = cv2.resize(d["small"], (640, 360), interpolation=cv2.INTER_NEAREST)
    gx, gy = d["patch"]                  # which pixels we zoom into (column, row in 128x72)
    cv2.rectangle(big, (gx * 5 - 4, gy * 5 - 4), ((gx + 4) * 5 + 4, (gy + 3) * 5 + 4), (40, 50, 220), 3)
    pg.card(to_pil(big), 340, 400, 560, "big", angle=-1.5)
    k = ease((p - 0.2) / 0.3)
    if k > 0:
        pg.arrow(560, 380, lerp(560, 700, k), 380, RED, key="a2")
    cells = int(12 * ease((p - 0.35) / 0.45))
    for i in range(cells):
        r, c = divmod(i, 4)
        b, g, rr = (int(v) for v in d["small"][gy + r, gx + c])
        x, y = 760 + c * 120, 220 + r * 120
        dx, dy, _ = pg.wobble(f"cell{i}", 1.5)
        pg.draw.rectangle([x + dx, y + dy, x + 110 + dx, y + 110 + dy], fill=(rr, g, b), outline=INK, width=3)
        light = (rr + g + b) / 3 > 140
        col = INK if light else (250, 248, 242)
        for j, (name, v) in enumerate([("R", rr), ("G", g), ("B", b)]):
            pg.text(f"{name} {v}", x + 14, y + 10 + j * 32, 26, col, key=f"n{i}{j}")
    if p > 0.85:
        pg.text("red, green, blue: 0 to 255", 1000, 620, 32, RED, key="rgb", anchor="mm")


def s_finder(pg, p, d):
    pg.title("4. A kernel slides over the picture", p)
    big = cv2.resize(d["small"], (512, 288), interpolation=cv2.INTER_NEAREST)   # 4 screen px per pixel
    # the window steps across in jumps (every 8 pixels), row by row
    steps = [(x, y) for y in range(0, 72, 8) for x in range(0, 128, 8)]
    i = min(len(steps) - 1, int(len(steps) * p / 0.9))
    sx, sy = steps[i]
    cv2.rectangle(big, (sx * 4 - 8, sy * 4 - 8), (sx * 4 + 20, sy * 4 + 20), (40, 50, 220), 4)
    pg.card(to_pil(big), 320, 360, 512, "big", angle=-1)
    # the finder itself: 3x3 numbers it multiplies the pixels by
    w = d["finder_w"]
    m = np.abs(w).max() + 1e-6
    pg.text("the kernel (3x3)", 320, 545, 28, INK, key="ft", anchor="mm")
    for r in range(3):
        for c in range(3):
            v = w[r, c] / m
            col = (int(255 - 160 * max(0, v)), int(255 - 160 * abs(v)), int(255 - 160 * max(0, -v)))
            x, y = 245 + c * 52, 570 + r * 40
            pg.draw.rectangle([x, y, x + 48, y + 36], fill=col, outline=INK, width=2)
    pg.arrow(600, 360, 690, 360, key="a3")
    # its answer map, filled in up to where the window is
    m = d["first"][d["finder"]]
    out = np.array(colorize(m, (512, 288)))
    rows = min(288, (sy + 8) * 4)
    mask = np.zeros_like(out)
    mask[:rows - 32] = 1
    mask[rows - 32:rows, :min(512, (sx + 8) * 4)] = 1
    shown = np.where(mask > 0, out, 245).astype(np.uint8)
    pg.card(Image.fromarray(shown), 960, 360, 512, "out", angle=1)
    pg.text("bright = \"I found my pattern here\"", 960, 545, 28, RED, key="br", anchor="mm")


def s_many(pg, p, d):
    pg.title("16 kernels -> 16 new pictures", p)
    m = d["maps"][0]
    n = int(16 * ease(p / 0.8))
    for i in range(n):
        r, c = divmod(i, 4)
        pg.card(colorize(m[i], (192, 108)), 230 + c * 270, 190 + r * 135, 200, f"m{i}",
                angle=((i * 7) % 5 - 2) * 1.2, border=5)


def s_layers(pg, p, d):
    pg.title("5. Layers: smaller pictures, more patterns", p)
    info = [("16 maps", "64 x 36", "edges"), ("32 maps", "32 x 18", "curves"),
            ("64 maps", "16 x 9", "finger parts"), ("128 maps", "8 x 4", "the whole hand")]
    pg.card(to_pil(d["small"]), 120, 330, 170, "in", angle=-2, border=5)
    for b in range(4):
        k = ease((p - 0.08 - b * 0.18) / 0.15)
        if k <= 0:
            break
        x = 330 + b * 250
        pg.arrow(x - 115, 330, x - 75, 330, key=f"la{b}")
        maps = d["maps"][b]
        order = np.argsort(-maps.reshape(maps.shape[0], -1).std(1))
        for s in range(4, -1, -1):           # a little stack of cards, the strongest on top
            pg.card(colorize(maps[order[s]], (160, 90)), x + s * 7, lerp(700, 330, k) - s * 9, 160,
                    f"l{b}{s}", angle=s * 1.5 - 3, border=4)
        name, size, what = info[b]
        pg.text(name, x, 450, 30, INK, key=f"ln{b}", anchor="mm")
        pg.text(size, x, 488, 26, BLUE, key=f"ls{b}", anchor="mm")
        pg.text(what, x, 540, 32, RED, key=f"lw{b}", anchor="mm")


def s_vote(pg, p, d):
    pg.title("6. Every answer gets a vote", p)
    pooled = d["pooled"]
    n = int(128 * ease(p / 0.3))
    v = pooled / (pooled.max() + 1e-6)
    for i in range(n):
        r, c = divmod(i, 8)
        g = int(250 - 220 * v[i])
        x, y = 90 + c * 22, 140 + r * 31
        pg.draw.rectangle([x, y, x + 18, y + 26], fill=(g, g, int(g * 0.9)), outline=INK, width=1)
    if p > 0.15:
        pg.text("128 numbers", 175, 660, 30, INK, key="128", anchor="mm")
    if p > 0.3:
        pg.arrow(290, 390, 400, 390, key="va")
    k = ease((p - 0.32) / 0.4)
    best = int(np.argmax(d["chances"]))
    for i, lab in enumerate(LABELS):
        y = 150 + i * 70
        name = "no hand" if lab == "none" else lab
        pg.text(name, 470, y + 8, 36, INK, key=f"vl{i}", anchor="lm")
        frac = float(d["chances"][i]) * k
        pg.bar(600, y - 18, 520, 52, frac, GREEN if i == best else (170, 160, 150), f"vb{i}")
        if k >= 1:
            pg.text(f"{100 * d['chances'][i]:.0f}%", 1140, y + 8, 30, INK, key=f"vp{i}", anchor="lm")
    if p > 0.8:
        y = 150 + best * 70
        dx, dy, _ = pg.wobble("circ", 2)
        pg.draw.ellipse([440 + dx, y - 32 + dy, 540 + dx, y + 44 + dy], outline=RED, width=6)
        pg.text(f"It's a {LABELS[best]}!", 860, 660, 48, RED, key="win", anchor="mm")


def s_data(pg, p, d):
    pg.title("7. How it learned: lots of examples", p)
    pics = d["others"]
    n = int(len(pics) * ease(p / 0.45))
    for i in range(n):
        pg.card(to_pil(pics[i]), 230 + (i % 4) * 30, 330 + (i // 4) * 30 + i * 4, 300, f"d{i}",
                angle=(i * 13) % 11 - 5, border=6)
    if p > 0.3:
        pg.text("8,320 pictures", 270, 560, 38, INK, key="8320", anchor="mm")
        pg.text("from 5 people", 270, 605, 32, BLUE, key="5p", anchor="mm")
    if p > 0.5:
        pg.text("+ shaken-up copies:", 870, 150, 34, RED, key="shake", anchor="mm")
        k = int(6 * ease((p - 0.5) / 0.4))
        for i in range(k):
            r, c = divmod(i, 3)
            pg.card(to_pil(d["aug"][i]), 690 + c * 185, 280 + r * 130, 170, f"g{i}",
                    angle=(i % 3 - 1) * 3, border=4)
        if p > 0.9:
            pg.text("mirrored, tilted, zoomed, darker...", 870, 560, 30, INK, key="mt", anchor="mm")


def s_learning(pg, p, d):
    pg.title("8. Guess, check, tweak, repeat", p)
    steps = [("guess", BLUE), ("check the answer", INK), ("tweak every weight a tiny bit", RED)]
    for i, (s, col) in enumerate(steps):
        if p > 0.05 + i * 0.1:
            pg.text(f"{i + 1}. {s}", 70, 170 + i * 55, 34, col, key=f"st{i}")
    # the real practice scores from training (final network)
    rounds = [1, 5, 10, 15, 20, 25]
    score = [43.1, 92.3, 97.9, 99.2, 99.7, 99.8]
    x0, y0, cw, chh = 700, 560, 500, 360
    pg.draw.line([(x0, y0), (x0 + cw, y0)], fill=INK, width=4)
    pg.draw.line([(x0, y0), (x0, y0 - chh)], fill=INK, width=4)
    pg.text("score", x0 - 12, y0 - chh // 2, 24, INK, key="ps", anchor="rm")
    pg.text("rounds of study  ->", x0 + cw // 2, y0 + 35, 26, INK, key="rr", anchor="mm")
    pg.text("100%", x0 - 12, y0 - chh, 22, INK, key="100", anchor="rm")
    pg.text("0%", x0 - 12, y0, 22, INK, key="0", anchor="rm")
    shown = int(len(rounds) * ease((p - 0.3) / 0.55)) if p > 0.3 else 0
    pts = [(x0 + cw * (r - 1) / 24, y0 - chh * s / 100) for r, s in zip(rounds, score)][:shown]
    if len(pts) > 1:
        pg.draw.line(pts, fill=GREEN, width=5)
    for (x, y), s in zip(pts, score):
        pg.draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill=GREEN, outline=INK)
        pg.text(f"{s:.0f}%", x + (30 if s < 50 else 0), y - 28, 22, INK, key=f"sc{s}", anchor="mm")
    if p > 0.6:
        pg.text("25 rounds,", 260, 420, 34, BLUE, key="25", anchor="mm")
        pg.text("about 3 minutes", 260, 465, 34, BLUE, key="3m", anchor="mm")


def s_quiz(pg, p, d):
    pg.title("9. The quiz: a hand it never saw", p)
    pg.card(to_pil(d["new_person"]), 300, 360, 400, "np", angle=-3)
    pg.text("Kabir's hand", 300, 520, 32, INK, key="kh", anchor="mm")
    if p > 0.15:
        k1 = ease((p - 0.15) / 0.3)
        pg.text("learned from 2 people", 640, 230, 32, INK, key="q1")
        pg.bar(640, 270, 520, 56, 0.80 * k1, (200, 150, 60), "qb1")
        if k1 >= 1:
            pg.text("80%", 1180, 298, 34, INK, key="q1p", anchor="lm")
    if p > 0.5:
        k2 = ease((p - 0.5) / 0.3)
        pg.text("learned from 4 people", 640, 390, 32, INK, key="q2")
        pg.bar(640, 430, 520, 56, 0.985 * k2, GREEN, "qb2")
        if k2 >= 1:
            pg.text("98.5%", 1180, 458, 34, INK, key="q2p", anchor="lm")
    if p > 0.85:
        pg.text("more people = smarter network", 900, 600, 38, RED, key="more", anchor="mm")


def s_safe(pg, p, d):
    pg.title("10. Opening the safe", p)
    pg.card(to_pil(d["photo"]), 330, 330, 440, "ph", angle=-2)
    k = ease((p - 0.1) / 0.4)
    pg.text("hold it still for 2 seconds", 330, 500, 30, INK, key="hold", anchor="mm")
    pg.bar(130, 530, 400, 50, k, GREEN, "load")
    digits = "402"     # a made-up example code (not the real one!)
    filled = 0 if p < 0.5 else (1 if p < 0.62 else (2 if p < 0.74 else 3))
    for i in range(3):
        x = 760 + i * 130
        dx, dy, _ = pg.wobble(f"box{i}", 1.5)
        pg.draw.rectangle([x + dx, 230 + dy, x + 110 + dx, 360 + dy], fill=(40, 35, 30),
                          outline=GOLD, width=6)
        if i < filled:
            pg.text(digits[i], x + 55, 295, 90, GOLD, key=f"dg{i}", anchor="mm")
    pg.text("(example code)", 955, 395, 26, INK, key="ex", anchor="mm")
    if p > 0.8:
        pg.text("Vault 687: OPEN!", 955, 450, 46, GREEN, key="open", anchor="mm")


def s_end(pg, p, d):
    pg.text("Thanks for watching!", W // 2, H // 2, 70, INK, key="end", anchor="mm")


SCENES = [
    ("Title", 9, s_title),
    ("Camera picture", 8, s_camera),
    ("Shrink to 128 x 72", 10, s_shrink),
    ("Every pixel is 3 numbers", 13, s_numbers),
    ("A kernel slides", 17, s_finder),
    ("16 kernels", 7, s_many),
    ("Layers", 17, s_layers),
    ("The vote", 17, s_vote),
    ("Lots of examples", 15, s_data),
    ("Guess, check, tweak", 19, s_learning),
    ("The quiz", 14, s_quiz),
    ("Opening the safe", 14, s_safe),
    ("The end", 4, s_end),
]


def main():
    print("running the real network on the example picture...")
    d = gather()
    print(f"example picture: network says {LABELS[int(np.argmax(d['chances']))]} "
          f"({100 * d['chances'].max():.1f}%)")
    backgrounds = make_backgrounds()
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", "-crf", "18", str(OUT)], stdin=subprocess.PIPE)
    drawing, t = 0, 0.0
    times = []
    for name, secs, fn in SCENES:
        times.append((t, name))
        n = int(secs * FPS / HOLD)
        for i in range(n):
            page = Page(backgrounds, drawing)
            fn(page, i / max(1, n - 1), d)
            raw = page.img.tobytes()
            for _ in range(HOLD):
                ff.stdin.write(raw)
            drawing += 1
        t += n * HOLD / FPS
    ff.stdin.close()
    ff.wait()
    print(f"saved {OUT.name}: {t:.0f} seconds")
    print("scene start times:")
    for start, name in times:
        print(f"  {int(start // 60)}:{int(start % 60):02d}  {name}")


if __name__ == "__main__":
    main()

"""
A scoreboard standing beside the table (like the ones at real matches),
angled toward the camera. Shows both names in their shirt colors, big
point numbers, and a yellow ball next to whoever is serving. In practice
mode it shows your current rally and your best rally instead.
"""

from panda3d.core import Point3, TextNode

from .shapes import part

FRAME = (0.12, 0.13, 0.17, 1)
FACE = (0.05, 0.06, 0.09, 1)
LEG = (0.35, 0.36, 0.4, 1)
SERVE_BALL = (1.0, 0.85, 0.2, 1)
WHITE = (1, 1, 1, 1)

POSITION = (-1.7, -0.1)       # floor spot beside the table, near the net (Panda3D x, y)
WIDTH, HEIGHT = 1.1, 0.62      # board size (m)
CENTER_Z = 1.02                # height of the board's middle
COLUMN_X = 0.27                # left/right column centers on the board
# A clean scoreboard-style font (Panda's built-in one draws 0 like the letter O).
FONT_PATHS = ["/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf",
              "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/SFNS.ttf"]


def load_font(loader):
    for path in FONT_PATHS:
        try:
            font = loader.loadFont(path)
        except Exception:
            continue
        if font and font.isValid():
            font.setPixelsPerUnit(120)          # crisp at the board's size
            return font
    return None                                 # fall back to Panda's default font


class Scoreboard:
    def __init__(self, render, loader, camera_pos, left_color, right_color):
        self.font = load_font(loader)
        self.root = render.attachNewNode("scoreboard")
        x, y = POSITION
        self.root.setPos(x, y, 0)
        # Face the camera: -y is the board's front, so look away from the camera.
        self.root.lookAt(Point3(2 * x - camera_pos.x, 2 * y - camera_pos.y, 0))

        # Stand: two legs and a frame with a dark face.
        for side in (-1, 1):
            part(self.root, "box", LEG, pos=(side * 0.4, 0, (CENTER_Z - HEIGHT / 2) / 2),
                 scale=(0.05, 0.05, CENTER_Z - HEIGHT / 2))
        part(self.root, "box", FRAME, pos=(0, 0, CENTER_Z), scale=(WIDTH + 0.06, 0.08, HEIGHT + 0.06))
        front = -0.045
        part(self.root, "box", FACE, pos=(0, front, CENTER_Z), scale=(WIDTH, 0.01, HEIGHT))
        # Name strips in each player's color, and a line between the two columns.
        top = CENTER_Z + HEIGHT / 2 - 0.08
        self.strips = [part(self.root, "box", color, pos=(side * COLUMN_X, front - 0.006, top), scale=(0.5, 0.004, 0.12))
                       for side, color in ((-1, left_color), (1, right_color))]
        # The numbers sit centered in the space between the name strips and the board's bottom edge.
        number_mid = ((top - 0.06) + (CENTER_Z - HEIGHT / 2)) / 2
        part(self.root, "box", FRAME, pos=(0, front - 0.006, number_mid), scale=(0.015, 0.004, 0.36))

        # Text is drawn without the lighting shader so the letters show (like a lit-up board).
        self.names = [self._text((side * COLUMN_X + 0.02, front - 0.012, top - 0.03), 0.075) for side in (-1, 1)]
        self.numbers = [self._text((side * COLUMN_X, front - 0.012, number_mid), 0.3) for side in (-1, 1)]
        for np in self.numbers:
            # Text is placed by its baseline (bottom of the digits), so shift it down by half a digit's height.
            node = np.node()
            node.setText("0")
            np.setZ(number_mid - 0.3 * (node.getTop() + node.getBottom()) / 2)
        self.serve_dots = [part(self.root, "sphere", SERVE_BALL, pos=(side * COLUMN_X - 0.215, front - 0.02, top),
                                scale=0.05) for side in (-1, 1)]
        self.shown = None

    def _text(self, pos, size):
        node = TextNode("scoreboard_text")
        node.setAlign(TextNode.ACenter)
        node.setTextColor(*WHITE)
        if self.font:
            node.setFont(self.font)
        np = self.root.attachNewNode(node)
        np.setPos(*pos)
        np.setScale(size)
        np.setShaderOff(1)
        np.setLightOff(1)
        np.setDepthOffset(2)
        np.setTwoSided(True)
        return np

    def show(self, left_name, right_name, left_number, right_number, serving=None):
        """serving: 0 = left, 1 = right, None = nobody (e.g. practice)."""
        state = (left_name, right_name, left_number, right_number, serving)
        if state == self.shown:
            return                      # nothing changed: don't rebuild the text every frame
        self.shown = state
        for np, text in zip(self.names, (left_name, right_name)):
            np.node().setText(text.upper()[:8])
        for np, number in zip(self.numbers, (left_number, right_number)):
            np.node().setText(str(number))
        for i, dot in enumerate(self.serve_dots):
            dot.show() if serving == i else dot.hide()

    def set_colors(self, left_color, right_color):
        for strip, color in zip(self.strips, (left_color, right_color)):
            strip.setColor(*color)

    def set_visible(self, visible):
        self.root.show() if visible else self.root.hide()

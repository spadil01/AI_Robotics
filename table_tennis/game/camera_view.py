"""
Shows the live camera picture (sent by vision_node as small JPEGs) inside
the game window, with a colored frame around it.
"""

from direct.gui.DirectGui import DirectFrame
from direct.gui.OnscreenImage import OnscreenImage
from panda3d.core import PNMImage, StringStream, Texture

ASPECT = 16 / 9


class CameraView:
    def __init__(self, base):
        self.texture = Texture("camera")
        blank = PNMImage(64, 36)
        blank.fill(0.15, 0.15, 0.18)
        self.texture.load(blank)
        self.frame = DirectFrame(parent=base.aspect2d, frameColor=(1, 1, 1, 1))
        self.frame.setBin("gui-popup", 10)   # always drawn on top of other screens/overlays
        self.image = OnscreenImage(image=self.texture, parent=self.frame)
        self.last_jpeg = None
        self.hide()

    def place(self, x, y, height, border_color=(1, 1, 1, 1)):
        """Show the picture centered at (x, y) in screen units, `height` tall."""
        w, h = height * ASPECT / 2, height / 2
        pad = 0.012
        self.frame.setPos(x, 0, y)
        self.frame["frameSize"] = (-w - pad, w + pad, -h - pad, h + pad)
        self.frame["frameColor"] = border_color
        self.image.setScale(w, 1, h)
        self.frame.show()

    def hide(self):
        self.frame.hide()

    def update(self, jpeg):
        """Load a new picture if one arrived (call every frame from the game loop)."""
        if jpeg is None or jpeg is self.last_jpeg:
            return
        self.last_jpeg = jpeg
        picture = PNMImage()
        if picture.read(StringStream(jpeg), "camera.jpg"):
            self.texture.load(picture)

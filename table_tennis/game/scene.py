"""
The 3D world: floor, table, net, lights and camera. Bright, simple and
rounded, in the spirit of Wii Sports Resort (all original shapes).

Panda3D axes: x right, y away from the player, z up. The physics uses
(x, up, away), so convert with `to_panda`.
"""

from panda3d.core import AmbientLight, DirectionalLight, Vec3

from .lighting import apply_lighting
from .physics import HALF_LENGTH, HALF_WIDTH, NET_HEIGHT, NET_OVERHANG, TABLE_TOP
from .shapes import part

SKY = (0.55, 0.78, 0.98, 1)
TABLE_BLUE = (0.14, 0.37, 0.75, 1)
LINE_WHITE = (0.96, 0.96, 0.96, 1)
MAT = (0.27, 0.59, 0.35, 1)
DARK = (0.25, 0.25, 0.28, 1)


def rgb(r, g, b, a=255):
    return (r / 255, g / 255, b / 255, a / 255)


def to_panda(p):
    """Physics (x, up, away) -> Panda3D (x, away, up)."""
    return Vec3(p[0], p[2], p[1])


def build_world(base):
    render = base.render
    base.setBackgroundColor(*SKY)

    part(render, "box", MAT, pos=(0, 0, -0.05), scale=(40, 40, 0.1))     # one green floor everywhere

    # Table top with white edges and center line
    thickness = 0.04
    part(render, "box", TABLE_BLUE, pos=(0, 0, TABLE_TOP - thickness / 2),
         scale=(2 * HALF_WIDTH, 2 * HALF_LENGTH, thickness))
    line_z = TABLE_TOP + 0.001
    for x in (-HALF_WIDTH + 0.01, HALF_WIDTH - 0.01):
        part(render, "box", LINE_WHITE, pos=(x, 0, line_z), scale=(0.02, 2 * HALF_LENGTH, 0.002))
    for y in (-HALF_LENGTH + 0.01, HALF_LENGTH - 0.01):
        part(render, "box", LINE_WHITE, pos=(0, y, line_z), scale=(2 * HALF_WIDTH, 0.02, 0.002))
    part(render, "box", LINE_WHITE, pos=(0, 0, line_z), scale=(0.006, 2 * HALF_LENGTH, 0.002))
    for x in (-HALF_WIDTH + 0.12, HALF_WIDTH - 0.12):
        for y in (-HALF_LENGTH + 0.25, HALF_LENGTH - 0.25):
            part(render, "box", DARK, pos=(x, y, (TABLE_TOP - thickness) / 2),
                 scale=(0.06, 0.06, TABLE_TOP - thickness))

    # Net and posts
    net_width = 2 * (HALF_WIDTH + NET_OVERHANG)
    part(render, "box", (1, 1, 1, 0.55), pos=(0, 0, TABLE_TOP + NET_HEIGHT / 2),
         scale=(net_width, 0.008, NET_HEIGHT))
    part(render, "box", LINE_WHITE, pos=(0, 0, TABLE_TOP + NET_HEIGHT), scale=(net_width, 0.012, 0.015))
    for x in (-net_width / 2, net_width / 2):
        part(render, "box", DARK, pos=(x, 0, TABLE_TOP + NET_HEIGHT / 2), scale=(0.02, 0.02, NET_HEIGHT + 0.02))

    # Warm sun plus a cool fill light from the other side, and soft ambient.
    sun = DirectionalLight("sun")
    sun.setColor((0.75, 0.72, 0.65, 1))
    sun_np = render.attachNewNode(sun)
    sun_np.setHpr(35, -55, 0)
    render.setLight(sun_np)
    fill = DirectionalLight("fill")
    fill.setColor((0.2, 0.22, 0.28, 1))
    fill_np = render.attachNewNode(fill)
    fill_np.setHpr(-140, -20, 0)
    render.setLight(fill_np)
    ambient = AmbientLight("ambient")
    ambient.setColor((0.42, 0.42, 0.46, 1))
    render.setLight(render.attachNewNode(ambient))
    apply_lighting(render)

    # Behind and above the player, looking down the table (like Wii Sports Resort).
    base.disableMouse()
    base.camera.setPos(0, -5.0, 3.0)
    base.camera.lookAt(0, 0.55, 0.55)
    base.camLens.setFov(58)

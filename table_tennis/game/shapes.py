"""
Simple 3D shapes (box, sphere, cylinder) built in code, with normals so they
can be lit. All are 1 unit across and centered on their origin, so `scale`
sets their size directly.
"""

import math

from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat,
                          GeomVertexWriter, NodePath, TransparencyAttrib)


def _build(name, verts, tris):
    """verts: list of (position, normal); tris: list of vertex index triples (counter-clockwise)."""
    vdata = GeomVertexData(name, GeomVertexFormat.getV3n3(), Geom.UHStatic)
    positions = GeomVertexWriter(vdata, "vertex")
    normals = GeomVertexWriter(vdata, "normal")
    for p, n in verts:
        positions.addData3(*p)
        normals.addData3(*n)
    prim = GeomTriangles(Geom.UHStatic)
    for a, b, c in tris:
        prim.addVertices(a, b, c)
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode(name)
    node.addGeom(geom)
    return NodePath(node)


def box():
    verts, tris = [], []
    # (normal, u, v) with u x v = normal, so each face winds counter-clockwise from outside.
    faces = [((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
             ((0, 1, 0), (0, 0, 1), (1, 0, 0)), ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
             ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (0, 1, 0), (1, 0, 0))]
    for n, u, v in faces:
        start = len(verts)
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            verts.append((tuple(0.5 * (n[i] + su * u[i] + sv * v[i]) for i in range(3)), n))
        tris += [(start, start + 1, start + 2), (start, start + 2, start + 3)]
    return _build("box", verts, tris)


def sphere(segments=24, rings=14):
    verts, tris = [], []
    for r in range(rings + 1):
        phi = math.pi * r / rings
        for s in range(segments + 1):
            theta = 2 * math.pi * s / segments
            n = (math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta), math.cos(phi))
            verts.append((tuple(0.5 * c for c in n), n))
    row = segments + 1
    for r in range(rings):
        for s in range(segments):
            a, b = r * row + s, (r + 1) * row + s
            tris += [(a, b, a + 1), (a + 1, b, b + 1)]
    return _build("sphere", verts, tris)


def capsule(cyl=1.0, segments=24, rings=8):
    """Pill shape along z: radius 0.5 hemispheres joined by a tube of length `cyl`.
    Total height is 1 + cyl, centered on the origin."""
    verts, tris = [], []
    rows = []   # (phi, z offset): top cap, then bottom cap (equator duplicated for the tube)
    rows += [(math.pi / 2 * r / rings, cyl / 2) for r in range(rings + 1)]
    rows += [(math.pi / 2 + math.pi / 2 * r / rings, -cyl / 2) for r in range(rings + 1)]
    for phi, dz in rows:
        for s in range(segments + 1):
            theta = 2 * math.pi * s / segments
            n = (math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta), math.cos(phi))
            verts.append(((0.5 * n[0], 0.5 * n[1], 0.5 * n[2] + dz), n))
    row = segments + 1
    for r in range(len(rows) - 1):
        for s in range(segments):
            a, b = r * row + s, (r + 1) * row + s
            tris += [(a, b, a + 1), (a + 1, b, b + 1)]
    return _build("capsule", verts, tris)


def cylinder(segments=24, axis="z"):
    """Cylinder along `axis` ("x", "y" or "z")."""
    # Built along z, then the coordinates are cycled (a proper rotation, so
    # windings stay correct) to point along the requested axis.
    turn = {"z": lambda p: p, "x": lambda p: (p[2], p[0], p[1]), "y": lambda p: (p[1], p[2], p[0])}[axis]
    verts, tris = [], []
    ring = [(0.5 * math.cos(2 * math.pi * s / segments), 0.5 * math.sin(2 * math.pi * s / segments))
            for s in range(segments + 1)]
    for x, y in ring:                                   # side
        n = (2 * x, 2 * y, 0)
        verts += [((x, y, -0.5), n), ((x, y, 0.5), n)]
    for s in range(segments):
        a = 2 * s
        tris += [(a, a + 2, a + 3), (a, a + 3, a + 1)]
    for z, nz in ((0.5, 1), (-0.5, -1)):                # caps
        center = len(verts)
        verts.append(((0, 0, z), (0, 0, nz)))
        for x, y in ring:
            verts.append(((x, y, z), (0, 0, nz)))
        for s in range(segments):
            a, b = center + 1 + s, center + 2 + s
            tris.append((center, a, b) if nz > 0 else (center, b, a))
    verts = [(turn(p), turn(n)) for p, n in verts]
    return _build("cylinder", verts, tris)


SHAPES = {"box": box, "sphere": sphere, "cylinder": cylinder, "capsule": capsule}


def part(parent, shape, color, pos=(0, 0, 0), scale=(1, 1, 1), **shape_args):
    """Make a colored shape as a child of `parent`. Positions/scales are in Panda3D
    axes: x right, y forward, z up."""
    np = SHAPES[shape](**shape_args)
    np.reparentTo(parent)
    np.setPos(*pos)
    np.setScale(*scale) if isinstance(scale, (tuple, list)) else np.setScale(scale)
    np.setColor(*color)
    if len(color) == 4 and color[3] < 1:
        np.setTransparency(TransparencyAttrib.MAlpha)
    return np

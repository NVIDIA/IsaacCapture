"""Build a low-poly Pico 4 Ultra headset model in the OpenXR head-pose frame (metres, Y up, -Z forward).

The tracked head pose is the view origin between the eyes, so the eyes sit at the origin, the visor extends
forward (-Z) and the headband and rear battery pad wrap backwards (+Z). Dimensions follow Pico's published
figures for the Pico 4 Ultra (visor ~205 x 92 x 38 mm, pancake optics, two front RGB passthrough cameras and
one depth sensor); everything else is an approximation of the product's shape, not a scan.
"""

import sys

import numpy as np
import trimesh
from trimesh.creation import box, cylinder, torus

WHITE = [238, 238, 236, 255]
DARK = [38, 38, 42, 255]
GREY = [120, 120, 125, 255]
BLACK = [12, 12, 14, 255]


def part(mesh, color, translate=(0, 0, 0), rotate=None):
    """Colour one primitive and place it: rotate first (about the origin), then translate."""
    if rotate is not None:
        mesh.apply_transform(rotate)
    mesh.apply_translation(translate)
    mesh.visual.face_colors = np.tile(np.array(color, dtype=np.uint8), (len(mesh.faces), 1))
    return mesh


parts = []
# Visor body: 205 mm wide, 92 mm tall, 40 mm deep; its back face sits 35 mm in front of the eyes
# (pancake optics).
parts.append(part(box(extents=(0.205, 0.092, 0.040)), WHITE, (0, 0.005, -0.055)))
# Facial interface: soft black foam ring between face and visor, slightly narrower than the visor.
parts.append(part(box(extents=(0.175, 0.080, 0.023)), BLACK, (0, 0.003, -0.0235)))
# Front sensor bar: two RGB passthrough cameras and a central depth sensor on the visor's front face.
rot_x = trimesh.transformations.rotation_matrix(np.pi / 2, (1, 0, 0))
for x in (-0.032, 0.032):
    parts.append(part(cylinder(radius=0.0095, height=0.004, sections=24), DARK, (x, -0.012, -0.0765), rot_x))
parts.append(part(box(extents=(0.016, 0.010, 0.003)), DARK, (0, -0.012, -0.0765)))
# Headband: an elliptical ring (85 mm half-width, 105 mm half-depth) around the head, 30 mm tall, 10 mm thick.
ring = torus(major_radius=1.0, minor_radius=0.045, major_sections=48, minor_sections=10)
# Torus frame: x half-width, y half-depth, z the band's height once rotated to encircle the head.
ring.apply_scale((0.085, 0.105, 0.3))
ring.apply_transform(rot_x)  # torus axis to Y so the band encircles the head
parts.append(part(ring, GREY, (0, 0.012, 0.03)))
# Rear battery pad with its adjustment dial.
parts.append(part(box(extents=(0.120, 0.070, 0.038)), WHITE, (0, 0.012, 0.137)))
parts.append(part(cylinder(radius=0.022, height=0.006, sections=32), DARK, (0, 0.012, 0.159), rot_x))
# Top strap: a thin arc over the head.
top = torus(major_radius=1.0, minor_radius=0.012, major_sections=48, minor_sections=8)
top.apply_scale((0.11, 0.105, 0.5))
top.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (0, 0, 1)))  # ring in the Y-Z plane
top = top.slice_plane((0, 0.02, 0), (0, 1, 0))  # keep the upper half only
parts.append(part(top, GREY, (0, 0.012, 0.03)))

mesh = trimesh.util.concatenate(parts)
out = sys.argv[1]
mesh.export(out)
b = mesh.bounds
print("vertices", len(mesh.vertices), "faces", len(mesh.faces))
print("bounds", np.round(b[0], 3).tolist(), np.round(b[1], 3).tolist())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Pyrender-based mesh-on-image renderer for visualization.

Uses OSMesa backend for headless rendering on cluster.
"""

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np
import pyrender
import trimesh


def _create_raymond_lights():
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
    nodes = []
    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)
        z = np.array([xp, yp, zp])
        z /= np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(
            pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                matrix=matrix,
            )
        )
    return nodes


class Renderer:
    """Render SOMA mesh overlaid on an image using pyrender."""

    def __init__(self, focal_length, faces=None):
        self.focal_length = focal_length
        self.faces = faces
        self._offscreen_renderer = None
        self._viewport_size = None

    def __call__(
        self,
        vertices,
        cam_t,
        image,
        mesh_base_color=(1.0, 1.0, 0.9),
        camera_center=None,
    ):
        """Render mesh on image.

        Args:
            vertices: (V, 3) numpy array
            cam_t: (3,) camera translation
            image: (H, W, 3) uint8 image
            mesh_base_color: RGB tuple
            camera_center: (2,) pixel center, defaults to image center
        Returns:
            (H, W, 3) uint8 overlaid image
        """
        image_f = image.astype(np.float32) / 255.0
        h, w = image_f.shape[:2]
        if camera_center is None:
            camera_center = [w / 2.0, h / 2.0]
        focal = np.asarray(self.focal_length, dtype=np.float64).reshape(-1)
        if focal.size == 1:
            fx = fy = float(focal[0])
        elif focal.size == 2:
            fx, fy = float(focal[0]), float(focal[1])
        else:
            raise ValueError(
                "focal_length must be a scalar or an (fx, fy) pair, got "
                f"shape {np.asarray(self.focal_length).shape}"
            )

        camera_translation = cam_t.copy()
        camera_translation[0] *= -1.0

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode="OPAQUE",
            baseColorFactor=(
                mesh_base_color[2],
                mesh_base_color[1],
                mesh_base_color[0],
                1.0,
            ),
        )
        mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, "mesh")

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera = pyrender.IntrinsicsCamera(
            fx=fx,
            fy=fy,
            cx=camera_center[0],
            cy=camera_center[1],
            zfar=1e12,
        )
        scene.add(camera, pose=camera_pose)
        for node in _create_raymond_lights():
            scene.add_node(node)

        if self._viewport_size != (w, h):
            if self._offscreen_renderer is not None:
                self._offscreen_renderer.delete()
            self._offscreen_renderer = pyrender.OffscreenRenderer(
                viewport_width=w,
                viewport_height=h,
            )
            self._viewport_size = (w, h)
        color, depth = self._offscreen_renderer.render(
            scene,
            flags=pyrender.RenderFlags.RGBA,
        )

        color_f = color.astype(np.float32) / 255.0
        if color_f.shape[2] == 4:
            alpha = color_f[:, :, 3:4]
            out = color_f[:, :, :3] * alpha + (1.0 - alpha) * image_f
        else:
            mask = (depth > 0)[..., np.newaxis]
            out = np.where(mask, color_f, image_f)
        return (out * 255).clip(0, 255).astype(np.uint8)

    def close(self):
        """Release the cached OffscreenRenderer."""
        if self._offscreen_renderer is not None:
            try:
                self._offscreen_renderer.delete()
            except Exception:
                # Best effort: the renderer is discarded either way.
                pass
            self._offscreen_renderer = None
            self._viewport_size = None

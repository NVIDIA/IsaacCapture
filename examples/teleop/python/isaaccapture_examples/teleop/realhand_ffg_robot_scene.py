# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mesh rendering for fixed-wrist robot-hand pose verification, without physics."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .fetch_realhand_assets import _validate_model


def visual_links(asset_root, model, side):
    urdf = _validate_model(Path(asset_root), model)
    root = ET.parse(urdf).getroot()
    visuals = {}
    for link in root.findall("link"):
        name = link.get("name")
        if not name.startswith(f"{side}_hand_"):
            continue
        parts = []
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            origin = visual.find("origin")
            matrix = np.eye(4)
            if origin is not None:
                matrix[:3, :3] = Rotation.from_euler(
                    "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                ).as_matrix()
                matrix[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            matrix[:3, :3] *= np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            color = visual.find("material/color")
            rgba = (
                tuple(np.fromstring(color.get("rgba"), sep=" "))
                if color is not None
                else (0.82, 0.85, 0.89, 1.0)
            )
            parts.append((urdf.parent / mesh.get("filename"), matrix, rgba))
        if parts:
            visuals[name] = parts
    if not visuals:
        raise ValueError(f"No {side} hand visuals in {urdf}")
    return visuals


def robot_display_transform(side):
    matrix = np.eye(4)
    # Robot palm +X points up; fingertips +Z point toward the back of the bench.
    matrix[:3, :3] = ((0, 1, 0), (0, 0, 1), (1, 0, 0))
    matrix[:3, 3] = (-0.19 if side == "left" else 0.19, -0.09, 0.15)
    return matrix


class RobotHandScene:
    def __init__(self, stage, asset_root, model):
        import trimesh
        from pxr import Gf, UsdGeom, Vt

        self._gf = Gf
        self._transforms = {}
        self._bases = {}
        self._roots = {}
        for side in ("left", "right"):
            path = f"/World/{side.title()}RobotHand"
            root = UsdGeom.Xform.Define(stage, path)
            root.MakeInvisible()
            self._roots[side] = root
            self._bases[side] = robot_display_transform(side)
            UsdGeom.Xformable(root).AddTransformOp().Set(
                Gf.Matrix4d(self._bases[side].T.tolist())
            )
            transforms = {}
            for name, parts in visual_links(asset_root, model, side).items():
                link = UsdGeom.Xform.Define(stage, f"{path}/{name}")
                transforms[name] = UsdGeom.Xformable(link).AddTransformOp()
                for index, (filename, matrix, color) in enumerate(parts):
                    data = trimesh.load_mesh(filename, force="mesh", process=False)
                    geometry = UsdGeom.Mesh.Define(
                        stage, f"{path}/{name}/Visual_{index}"
                    )
                    geometry.CreatePointsAttr(
                        Vt.Vec3fArray.FromNumpy(
                            np.asarray(data.vertices, dtype=np.float32)
                        )
                    )
                    geometry.CreateFaceVertexCountsAttr([3] * len(data.faces))
                    geometry.CreateFaceVertexIndicesAttr(
                        np.asarray(data.faces).flatten().tolist()
                    )
                    geometry.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                    geometry.CreateDoubleSidedAttr(True)
                    geometry.CreateDisplayColorAttr([Gf.Vec3f(*color[:3])])
                    UsdGeom.Xformable(geometry).AddTransformOp().Set(
                        Gf.Matrix4d(matrix.T.tolist())
                    )
            self._transforms[side] = transforms

    def update(self, robots):
        for side, transforms in self._transforms.items():
            poses = robots.get(side, {}).get("link_poses", {})
            if poses:
                self._roots[side].MakeVisible()
            for name, operation in transforms.items():
                if name in poses:
                    matrix = np.asarray(poses[name], dtype=float)
                    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                        raise ValueError(f"Invalid link transform for {name}")
                    operation.Set(self._gf.Matrix4d(matrix.T.tolist()))

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import trimesh
from trimesh import transformations as tf
from yourdfpy import URDF

from .document import Document, JOINTS, Segment, Taxel

VISUALIZATION_FROM_HAND = np.array(
    [[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    dtype=float,
)
VISUALIZATION_FROM_URDF = np.array(
    [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
    dtype=float,
)


def matrix(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if (
        array.shape != (7,)
        or not np.isfinite(array).all()
        or abs(array[3:] @ array[3:] - 1.0) > 1e-4
    ):
        raise ValueError("Invalid XYZ/XYZW pose")
    result = tf.quaternion_matrix(array[[6, 3, 4, 5]])
    result[:3, 3] = array[:3]
    return result


def pose(transform: np.ndarray) -> list[float]:
    quaternion = tf.quaternion_from_matrix(transform)
    return [
        *np.asarray(transform[:3, 3], dtype=float).tolist(),
        *quaternion[[1, 2, 3, 0]].tolist(),
    ]


def transform(position, rotation_degrees) -> np.ndarray:
    result = tf.euler_matrix(*np.deg2rad(rotation_degrees), axes="sxyz")
    result[:3, 3] = np.asarray(position, dtype=float)
    return result


def components(value: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (
        tuple(value[:3, 3]),
        tuple(np.rad2deg(tf.euler_from_matrix(value, axes="sxyz"))),
    )


def unit(value) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    norm = np.linalg.norm(value)
    if not np.isfinite(norm) or norm < 1e-10:
        raise ValueError("Expected a finite nonzero direction")
    return value / norm


def surface_frame(origin, normal) -> np.ndarray:
    z = unit(normal)
    x = np.array([1.0, 0.0, 0.0])
    x -= z * (x @ z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([0.0, 1.0, 0.0])
        x -= z * (x @ z)
    x = unit(x)
    result = np.eye(4)
    result[:3, :3] = np.column_stack((x, np.cross(z, x), z))
    result[:3, 3] = origin
    return result


def realign(document: Document, alignment: np.ndarray) -> np.ndarray:
    old = np.asarray(document.alignment, dtype=float)
    delta = alignment @ np.linalg.inv(old)
    for value in (*document.segments, *document.taxels):
        value.pose = pose(delta @ matrix(value.pose))
    document.alignment = alignment.tolist()
    return delta


def apply_delta(values, rows: list[int], delta: np.ndarray) -> None:
    for row in rows:
        if 0 <= row < len(values):
            values[row].pose = pose(delta @ matrix(values[row].pose))


def scale_positions(values, rows: list[int], factor: float) -> None:
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Scale must be finite and positive")
    points = np.asarray([values[row].pose[:3] for row in rows], dtype=float)
    center = points.mean(axis=0)
    for row in rows:
        values[row].pose[:3] = (
            center + (values[row].pose[:3] - center) * factor
        ).tolist()


def grid_poses(
    center, width: float, height: float, rows: int, columns: int, frame=None
) -> list[list[float]]:
    if rows < 1 or columns < 1:
        raise ValueError("Rows and Columns must be positive")
    if not np.isfinite(width) or not np.isfinite(height) or width < 0 or height < 0:
        raise ValueError("Grid dimensions must be finite and nonnegative")
    basis = np.eye(4) if frame is None else np.asarray(frame, dtype=float).copy()
    basis[:3, 3] = np.asarray(center, dtype=float)
    result = []
    for row in range(rows):
        y = 0.0 if rows == 1 else (row / (rows - 1) - 0.5) * height
        for column in range(columns):
            x = 0.0 if columns == 1 else (column / (columns - 1) - 0.5) * width
            current = basis.copy()
            current[:3, 3] += basis[:3, :3] @ [x, y, 0.0]
            result.append(pose(current))
    return result


class Model:
    def __init__(self, path: str):
        self.path = Path(path).resolve()
        self.handles: dict[str, object] = {}
        self._fk_cache: OrderedDict[tuple[float, ...], dict[str, np.ndarray]] = (
            OrderedDict()
        )

        def resource(fname):
            name = str(fname)
            if name.startswith("package://"):
                package, relative = name[10:].split("/", 1)
                root = self.path.parent
                while root != root.parent and root.name != package:
                    root = root.parent
                if root.name == package:
                    return str((root / relative).resolve())
                for root in self.path.parents:
                    candidate = root / relative
                    if candidate.is_file():
                        return str(candidate)
                raise ValueError(f"Cannot resolve package resource: {name}")
            elif "://" in name:
                raise ValueError(f"Only local mesh resources are supported: {name}")
            return str((self.path.parent / name).resolve())

        self.urdf = URDF.load(str(self.path), filename_handler=resource)
        self.joints = self.urdf.actuated_joints
        self.links = list(self.urdf.link_map)
        self.surfaces: dict[str, tuple[str, trimesh.Trimesh]] = {}
        scene = self.urdf.scene
        parents = {child: parent for parent, child, _ in scene.graph.to_edgelist()}
        for node in scene.graph.nodes_geometry:
            link = node
            while link not in self.links:
                link = parents[link]
            local, geometry_name = scene.graph.get(frame_to=node, frame_from=link)
            mesh = scene.geometry[geometry_name].copy()
            mesh.apply_transform(local)
            for index, part in enumerate(mesh.split(only_watertight=False)):
                self.surfaces[f"{link}/{node}/{index}"] = (link, part)

    def raw_fk(self, q: dict[str, float]) -> dict[str, np.ndarray]:
        configuration = tuple(float(q.get(joint.name, 0.0)) for joint in self.joints)
        cached = self._fk_cache.pop(configuration, None)
        if cached is None:
            self.urdf.update_cfg(
                dict(zip((j.name for j in self.joints), configuration))
            )
            cached = {link: self.urdf.get_transform(link).copy() for link in self.links}
        self._fk_cache[configuration] = cached
        if len(self._fk_cache) > 4:
            self._fk_cache.popitem(last=False)
        return cached

    @staticmethod
    def _rigid(raw: np.ndarray, alignment: np.ndarray, scale: float) -> np.ndarray:
        result = alignment.copy()
        result[:3, :3] = alignment[:3, :3] @ raw[:3, :3]
        result[:3, 3] = alignment[:3, :3] @ (raw[:3, 3] * scale) + alignment[:3, 3]
        return result

    def draw(
        self,
        scene,
        q: dict[str, float],
        alignment: np.ndarray,
        scale: float,
        highlighted: bool = False,
        opacity: float = 0.6,
    ) -> None:
        raw = self.raw_fk(q)
        for name, (link, mesh) in self.surfaces.items():
            if name not in self.handles:
                self.handles[name] = scene.add_mesh_simple(
                    "/urdf/" + name,
                    mesh.vertices,
                    mesh.faces,
                    color=(170, 175, 180),
                    opacity=opacity,
                    side="double",
                )
            current = self._rigid(raw[link], alignment, scale)
            handle = self.handles[name]
            handle.position = current[:3, 3]
            handle.wxyz = tf.quaternion_from_matrix(current)
            handle.scale = scale
            handle.opacity = opacity
            handle.color = (255, 210, 45) if highlighted else (170, 175, 180)
            handle.visible = True

    def hit(
        self,
        origin,
        direction,
        q: dict[str, float],
        alignment: np.ndarray,
        scale: float,
    ):
        nearest, transforms = None, self.raw_fk(q)
        for link, mesh in self.surfaces.values():
            raw = transforms[link]
            similarity = alignment @ np.diag([scale, scale, scale, 1.0]) @ raw
            inverse = np.linalg.inv(similarity)
            local_origin = (inverse @ np.r_[origin, 1.0])[:3]
            local_direction = inverse[:3, :3] @ direction
            points, _, faces = mesh.ray.intersects_location(
                [local_origin], [local_direction]
            )
            for point, face in zip(points, faces):
                world = (similarity @ np.r_[point, 1.0])[:3]
                distance = np.linalg.norm(world - origin)
                if nearest is None or distance < nearest[0]:
                    normal = (
                        np.linalg.inv(similarity[:3, :3]).T @ mesh.face_normals[face]
                    )
                    nearest = (distance, world, unit(normal))
        return nearest

    def remove(self) -> None:
        for handle in self.handles.values():
            handle.remove()
        self.handles.clear()


def deform(binding: Document, current: list[Segment]) -> list[Taxel]:
    bind_segments = {
        segment.joint: matrix(segment.pose) for segment in binding.segments
    }
    current_segments = {segment.joint: matrix(segment.pose) for segment in current}
    if len(bind_segments) != len(binding.segments) or len(current_segments) != len(
        current
    ):
        raise ValueError("Segment HandJoint values must be unique")
    result = []
    for taxel in binding.taxels:
        active = [
            (joint, weight)
            for joint, weight in zip(taxel.segments, taxel.weights)
            if joint is not None and weight > 0
        ]
        if not active:
            raise ValueError("Taxel has no active segment")
        transforms = []
        for joint, weight in active:
            if joint not in bind_segments or joint not in current_segments:
                raise ValueError(f"Missing segment {joint}")
            transforms.append(
                (weight, current_segments[joint] @ np.linalg.inv(bind_segments[joint]))
            )
        bind_taxel = matrix(taxel.pose)
        position = sum(
            weight * (delta @ np.r_[bind_taxel[:3, 3], 1.0])[:3]
            for weight, delta in transforms
        )
        if len({joint for joint, _ in active}) == 1:
            rotation = transforms[0][1][:3, :3] @ bind_taxel[:3, :3]
        else:
            blend = sum(weight * delta[:3, :3] for weight, delta in transforms)
            u, singular, vh = np.linalg.svd(blend)
            if singular[-1] < 1e-8:
                rotation = bind_taxel[:3, :3]
            else:
                correction = np.eye(3)
                correction[2, 2] = np.linalg.det(u @ vh)
                rotation = u @ correction @ vh @ bind_taxel[:3, :3]
        value = np.eye(4)
        value[:3, :3], value[:3, 3] = rotation, position
        result.append(Taxel(pose(value), list(taxel.segments), list(taxel.weights)))
    return result


def hand_segments(joints: list[str], hand: dict) -> list[Segment]:
    values = []
    for position, orientation, valid in zip(
        hand["positions"], hand["orientations"], hand["is_valid"]
    ):
        values.append(matrix([*position, *orientation]) if valid else None)
    if len(values) != len(JOINTS) or values[1] is None:
        raise ValueError("HandTracker WRIST pose is unavailable")
    wrist_inverse = np.linalg.inv(values[1])
    result = []
    for joint in joints:
        index = JOINTS.index(joint)
        if values[index] is None:
            raise ValueError(f"HandTracker {joint} pose is unavailable")
        result.append(
            Segment(
                joint, pose(VISUALIZATION_FROM_HAND @ wrist_inverse @ values[index])
            )
        )
    return result

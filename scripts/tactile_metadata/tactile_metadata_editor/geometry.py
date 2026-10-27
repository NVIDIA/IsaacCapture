# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import trimesh
from trimesh import transformations as tf
from yourdfpy import URDF

from .document import FIELDS, joint_number, row_error


def matrix(p):
    try:
        a = np.asarray(p, dtype=float)
    except OverflowError as exc:
        raise ValueError("Pose components exceed floating-point range") from exc
    if a.shape != (7,) or not np.isfinite(a).all() or abs(a[3:] @ a[3:] - 1) > 1e-4:
        raise ValueError("Invalid XYZ/XYZW pose")
    result = tf.quaternion_matrix(a[[6, 3, 4, 5]])
    result[:3, 3] = a[:3]
    return result


def pose(transform):
    q = tf.quaternion_from_matrix(transform)
    return [*transform[:3, 3].tolist(), *q[[1, 2, 3, 0]].tolist()]


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length < 1e-10:
        raise ValueError("Reference directions must be finite and non-collinear")
    return vector / length


def frame(origin, normal, tangent):
    z = unit(normal)
    x = unit(np.asarray(tangent) - z * (np.asarray(tangent) @ z))
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = np.column_stack((x, np.cross(z, x), z)), origin
    return result


def frames(rows, root):
    for i, p in enumerate(rows):
        try:
            yield i, root @ matrix(p)
        except (ValueError, TypeError):
            pass


class Model:
    def __init__(self, path, packages):
        self.path, self.missing, self.surfaces = Path(path).resolve(), [], {}
        self.handles, self.attachments = {}, {}
        # Instance-owned caches must not keep replaced models alive.
        self._fk_cache = OrderedDict()

        # yourdfpy invokes this resolver using the keyword fname.
        def resource(fname):
            name = fname
            if name.startswith("package://"):
                package, relative = name[10:].split("/", 1)
                resolved = (
                    Path(packages.get(package, self.path.parent / package)) / relative
                )
            elif "://" in name:
                raise ValueError(f"Only local resources are supported: {name}")
            else:
                resolved = self.path.parent / name
            if not resolved.is_file():
                self.missing.append(str(resolved))
            return str(resolved)

        self.urdf = URDF.load(str(self.path), filename_handler=resource)
        self.links, self.joints = list(self.urdf.link_map), self.urdf.actuated_joints
        scene = self.urdf.scene
        parents = {b: a for a, b, _ in scene.graph.to_edgelist()}
        for node in scene.graph.nodes_geometry:
            link = node
            while link not in self.links:
                link = parents[link]
            transform, name = scene.graph.get(frame_to=node, frame_from=link)
            mesh = scene.geometry[name].copy()
            mesh.apply_transform(transform)
            for i, part in enumerate(mesh.split(only_watertight=False)):
                self.surfaces[f"{link}/{node}/{i}"] = (link, part)

    def fk(self, cfg):
        configuration = tuple(float(cfg.get(j.name, 0.0)) for j in self.joints)
        result = self._fk_cache.pop(configuration, None)
        if result is None:
            self.urdf.update_cfg(
                dict(zip((j.name for j in self.joints), configuration))
            )
            result = {link: self.urdf.get_transform(link).copy() for link in self.links}
        self._fk_cache[configuration] = result
        if len(self._fk_cache) > 4:
            self._fk_cache.popitem(last=False)
        return result

    def root(self, d, fk):
        reference = d.editor["root"]
        local = np.asarray(reference["frame"], dtype=float)
        if (
            local.shape != (4, 4)
            or not np.isfinite(local).all()
            or not np.allclose(local[3], [0, 0, 0, 1])
            or not np.allclose(local[:3, :3].T @ local[:3, :3], np.eye(3))
            or np.linalg.det(local[:3, :3]) < 0
        ):
            raise ValueError("Invalid wrist reference frame")
        return fk[reference["link"]] @ local

    def attach(self, d, fk):
        root = self.root(d, fk)
        attachments = self.attachments[d.side] = {}
        for i, sid in enumerate(d.editor["segment_ids"]):
            link = d.editor.get("segment_links", {}).get(sid)
            try:
                attachments[sid] = (
                    np.linalg.inv(fk[link]) @ root @ matrix(d.arrays[FIELDS[0]][i])
                )
            except (KeyError, TypeError, ValueError):
                pass

    def current(self, d, fk):
        inverse = np.linalg.inv(self.root(d, fk))
        links, attachments = (
            d.editor.get("segment_links", {}),
            self.attachments.get(d.side, {}),
        )
        return [
            inverse @ fk[links[sid]] @ attachments[sid]
            if sid in attachments and links.get(sid) in fk
            else None
            for sid in d.editor["segment_ids"]
        ]

    def draw(self, scene, cfg, visible=True, highlight=None):
        fk = self.fk(cfg)
        for name, (link, mesh) in self.surfaces.items():
            if name not in self.handles:
                self.handles[name] = scene.add_mesh_simple(
                    "/model/" + name,
                    mesh.vertices,
                    mesh.faces,
                    color=(170, 175, 180),
                    opacity=0.6,
                    side="double",
                )
            h = self.handles[name]
            h.visible = visible
            h.color = (
                (255, 220, 0) if highlight and link in highlight else (170, 175, 180)
            )
            h.position, h.wxyz = fk[link][:3, 3], tf.quaternion_from_matrix(fk[link])
        return fk

    def hit(self, origin, direction, surfaces, fk):
        nearest = None
        for name in surfaces:
            link, mesh = self.surfaces[name]
            inverse = np.linalg.inv(fk[link])
            local = (inverse @ np.r_[origin, 1])[:3]
            points, _, faces = mesh.ray.intersects_location(
                [local], [inverse[:3, :3] @ direction]
            )
            for point, face in zip(points, faces):
                world = (fk[link] @ np.r_[point, 1])[:3]
                distance = np.linalg.norm(world - origin)
                if nearest is None or distance < nearest[0]:
                    nearest = (
                        distance,
                        world,
                        fk[link][:3, :3] @ mesh.face_normals[face],
                        name,
                    )
        return nearest

    def grid(self, d, fk, surface, points, rows, cols, scan, reverse, excluded):
        if (
            rows < 1
            or cols < 1
            or points[0] is None
            or (cols > 1 and points[1] is None)
            or (rows > 1 and points[2] is None)
        ):
            raise ValueError(
                "Pick required first/last centers and use positive row/column counts"
            )
        if scan not in (
            "row-major",
            "column-major",
            "row-snake",
            "column-snake",
        ) or any(
            len(v) != 2
            or any(type(k) is not int for k in v)
            or not 0 <= v[0] < rows
            or not 0 <= v[1] < cols
            for v in excluded
        ):
            raise ValueError("Invalid scan or excluded cell coordinates")
        origin = np.asarray(points[0])
        column = (
            np.asarray(points[1]) - origin if points[1] is not None else np.zeros(3)
        )
        row = np.asarray(points[2]) - origin if points[2] is not None else np.zeros(3)
        if (
            (cols > 1 and np.linalg.norm(column) < 1e-10)
            or (rows > 1 and np.linalg.norm(row) < 1e-10)
            or (rows > 1 and cols > 1 and np.linalg.norm(np.cross(column, row)) < 1e-10)
        ):
            raise ValueError(
                "Required row/column spans must be nonzero and independent"
            )
        tangent = unit(column if np.linalg.norm(column) > 0 else row)
        link, mesh = self.surfaces[surface]
        face = trimesh.proximity.closest_point(
            mesh, [(np.linalg.inv(fk[link]) @ np.r_[origin, 1])[:3]]
        )[2][0]
        normal = fk[link][:3, :3] @ mesh.face_normals[face]
        cells = [(r, c) for r in range(rows) for c in range(cols)]
        major = 1 if scan.startswith("column") else 0
        cells.sort(
            key=lambda rc: (
                rc[major],
                -rc[1 - major]
                if scan.endswith("snake") and rc[major] % 2
                else rc[1 - major],
            )
        )
        depth, inverse = (
            max(np.linalg.norm(mesh.extents), 0.01),
            np.linalg.inv(self.root(d, fk)),
        )
        result = []
        for r, c in cells:
            r, c = rows - 1 - r if reverse[0] else r, cols - 1 - c if reverse[1] else c
            if [r, c] in excluded:
                continue
            point = (
                origin
                + column * (c / (cols - 1) if cols > 1 else 0)
                + row * (r / (rows - 1) if rows > 1 else 0)
            )
            hit = self.hit(point + normal * depth, -normal, [surface], fk)
            try:
                candidate = (
                    pose(inverse @ frame(hit[1], hit[2], tangent)) if hit else None
                )
            except ValueError:
                candidate = None
            result.append((candidate, (inverse @ np.r_[point, 1])[:3]))
        return result


def transform(d, field, selected, delta, local=False):
    for i in selected:
        old = matrix(d.arrays[field][i])
        d.arrays[field][i] = pose(old @ delta if local else delta @ old)
    d.editor["approximate"] = True


def deform(d, current):
    result = []
    for i, p in enumerate(d.arrays[FIELDS[1]]):
        try:
            indices, weights = d.arrays[FIELDS[2]][i], d.arrays[FIELDS[3]][i]
            if any(
                row_error(f, v, len(current))
                for f, v in zip(FIELDS[2:], (indices, weights))
            ):
                raise ValueError("Invalid skinning row")
            transforms = [
                current[j] @ np.linalg.inv(matrix(d.arrays[FIELDS[0]][j])) @ matrix(p)
                for j, w in zip(indices, weights)
                if w > 0
            ]
            position = sum(
                w * t[:3, 3] for w, t in zip([w for w in weights if w > 0], transforms)
            )
            rigid = (
                transforms[0].copy()
                if len({j for j, w in zip(indices, weights) if w > 0}) == 1
                else None
            )
            if rigid is not None:
                rigid[:3, 3] = position
            result.append((position, rigid))
        except (TypeError, ValueError, KeyError, IndexError):
            result.append((None, None))
    return result


def hand_current(d, hand):
    values = []
    for p, q, valid in zip(hand["positions"], hand["orientations"], hand["is_valid"]):
        try:
            values.append(matrix([*p, *q]) if valid else None)
        except (ValueError, TypeError):
            values.append(None)
    if len(values) != 26 or len(hand["is_valid"]) != 26 or values[1] is None:
        raise ValueError("HandTracker WRIST or array shape is invalid")
    inverse = np.linalg.inv(values[1])
    return (
        [
            inverse @ values[joint_number(j)]
            if values[joint_number(j)] is not None
            else None
            for j in d.joints
        ],
        values,
        inverse,
    )

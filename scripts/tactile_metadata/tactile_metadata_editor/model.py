# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local URDF graph, immutable link geometry and restricted surface queries."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
from pathlib import Path
import threading
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from yourdfpy import URDF

from .domain import FrozenMap, Issue, ModelConfig
from .geometry import (
    immutable_array,
    matrix_pose,
    pose_matrix,
    transform_direction,
    transform_point,
    unit,
)


class ModelError(ValueError):
    pass


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SurfaceHit:
    geometry_key: str
    link: str
    resource_hash: str
    triangle: int
    component: int
    barycentric: tuple
    position: tuple
    normal: tuple
    distance: float

    def attachment(self):
        return FrozenMap({
            "geometry_key": self.geometry_key,
            "link": self.link,
            "resource_hash": self.resource_hash,
            "triangle": self.triangle,
            "component": self.component,
            "barycentric": self.barycentric,
        })


@dataclass(frozen=True, init=False, eq=False)
class LocalGeometry:
    """Model-owned acceleration cache; snapshot callers receive read-only arrays."""

    key: str
    link: str
    resource_hash: str
    vertices: object
    faces: object
    normals: object
    face_normals: object
    components: object
    _mesh: object
    _lock: object

    def __init__(self, key, link, resource_hash, mesh):
        for name, value in (
            ("key", key), ("link", link), ("resource_hash", resource_hash),
            ("vertices", immutable_array(mesh.vertices)),
            ("faces", immutable_array(mesh.faces, np.int64)),
            ("normals", immutable_array(mesh.vertex_normals)),
            ("face_normals", immutable_array(mesh.face_normals)),
        ):
            object.__setattr__(self, name, value)
        labels = np.empty(len(mesh.faces), dtype=np.int64)
        for index, component in enumerate(trimesh.graph.connected_components(
            mesh.face_adjacency, nodes=np.arange(len(mesh.faces)), min_len=1
        )):
            labels[component] = index
        object.__setattr__(self, "components", immutable_array(labels, np.int64))
        object.__setattr__(self, "_mesh", mesh)
        object.__setattr__(self, "_lock", threading.Lock())

    def intersections(self, origin, direction):
        with self._lock:
            locations, _, faces = self._mesh.ray.intersects_location(
                [origin], [direction], multiple_hits=True
            )
        return locations, faces


@dataclass(frozen=True)
class SurfaceInstance:
    geometry: LocalGeometry
    pose: object

    def __post_init__(self):
        object.__setattr__(self, "pose", immutable_array(self.pose))


@dataclass(frozen=True)
class SurfaceSnapshot:
    instances: tuple[SurfaceInstance, ...]

    def __post_init__(self):
        object.__setattr__(self, "instances", tuple(self.instances))

    def ray_hits(self, origin, direction, *, max_distance, allowed_links=None,
                 component_seeds=None, reference_normal=None, flip_normal=False):
        origin = np.asarray(origin, dtype=np.float64)
        direction = unit(direction)
        reference = unit(reference_normal) if reference_normal is not None else None
        hits = []
        for instance in self.instances:
            geometry, transform = instance.geometry, instance.pose
            if allowed_links is not None and geometry.link not in allowed_links:
                continue
            if component_seeds is not None and geometry.key not in component_seeds:
                continue
            inverse = np.linalg.inv(transform)
            local_origin = transform_point(inverse, origin)
            local_direction = transform_direction(inverse, direction)
            locations, faces = geometry.intersections(local_origin, local_direction)
            for location, face in zip(locations, faces):
                face = int(face)
                component = int(geometry.components[face])
                if (
                    component_seeds is not None
                    and component not in component_seeds[geometry.key]
                ):
                    continue
                position = transform_point(transform, location)
                distance = float(np.dot(position - origin, direction))
                if distance < -1e-9 or distance > max_distance + 1e-9:
                    continue
                triangle = geometry.vertices[geometry.faces[face]]
                barycentric = trimesh.triangles.points_to_barycentric(
                    triangle[None, ...], np.asarray(location)[None, ...]
                )[0]
                normal = barycentric @ geometry.normals[geometry.faces[face]]
                if np.linalg.norm(normal) < 1e-6:
                    normal = geometry.face_normals[face]
                normal = unit(transform_direction(transform, normal))
                if flip_normal:
                    normal = -normal
                face_normal = unit(
                    transform_direction(transform, geometry.face_normals[face])
                )
                if flip_normal:
                    face_normal = -face_normal
                if reference is not None and np.dot(face_normal, reference) < np.cos(
                    np.deg2rad(75)
                ):
                    continue
                hits.append(SurfaceHit(
                    geometry.key, geometry.link, geometry.resource_hash, face,
                    component, tuple(float(x) for x in barycentric),
                    tuple(float(x) for x in position),
                    tuple(float(x) for x in normal), distance,
                ))
        return tuple(hits)

    def project(self, seed, normal, *, depth, allowed_links, component_seeds,
                flip_normal=False):
        if not np.isfinite(depth) or depth <= 0:
            raise ModelError("Projection depth must be finite and positive")
        seed, normal = np.asarray(seed), unit(normal)
        hits = self.ray_hits(
            seed + depth * normal, -normal, max_distance=2 * depth,
            allowed_links=allowed_links, component_seeds=component_seeds,
            reference_normal=normal, flip_normal=flip_normal,
        )
        if not hits:
            raise ModelError("No permitted surface intersects the projection ray")
        ordered = sorted(
            hits, key=lambda hit: np.linalg.norm(np.asarray(hit.position) - seed)
        )
        best = ordered[0]
        distance = np.linalg.norm(np.asarray(best.position) - seed)
        for other in ordered[1:]:
            other_distance = np.linalg.norm(np.asarray(other.position) - seed)
            if abs(other_distance - distance) > 1e-8:
                break
            same_surface = (
                np.linalg.norm(np.asarray(other.position) - best.position) <= 1e-8
                and np.dot(other.normal, best.normal) > 1 - 1e-6
            )
            if not same_surface:
                raise ModelError(
                    "Projection is equally close to distinct permitted surfaces"
                )
        return best


class Model:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.path = Path(config.path).expanduser().resolve()
        self._lock = threading.RLock()
        self._resources = {"urdf": self.path}
        tree = ET.parse(self.path)
        for joint in tree.getroot().findall("joint"):
            if joint.get("type") not in (
                "fixed", "revolute", "continuous", "prismatic"
            ):
                raise ModelError(f"Unsupported URDF joint type: {joint.get('type')}")
        for mesh in tree.getroot().findall(".//visual/geometry/mesh"):
            self._resolve_resource(mesh.attrib["filename"])
        parsed = URDF.load(
            str(self.path), build_scene_graph=False, load_meshes=False,
            build_collision_scene_graph=False, load_collision_meshes=False,
            filename_handler=self._resolve_resource,
        )
        names = [link.name for link in parsed.robot.links]
        joint_names = [joint.name for joint in parsed.robot.joints]
        parents = {joint.child: joint.parent for joint in parsed.robot.joints}
        if len(set(names)) != len(names) or len(set(joint_names)) != len(joint_names):
            raise ModelError("URDF link and joint names must be unique")
        if len(parents) != len(parsed.robot.joints):
            raise ModelError("A URDF link cannot have multiple parent joints")
        if any(
            child not in names or parent not in names
            for child, parent in parents.items()
        ):
            raise ModelError("URDF joints reference missing links")
        if len(set(names) - set(parents)) != 1:
            raise ModelError("The URDF must have one connected root")
        for name in names:
            visited = set()
            while name in parents:
                if name in visited:
                    raise ModelError("URDF parent links form a cycle")
                visited.add(name)
                name = parents[name]
        self._visual_tokens = {}
        for link_index, link in enumerate(parsed.robot.links):
            for visual_index, visual in enumerate(link.visuals):
                # Trimesh increments trailing numbers when assigning unique names.
                token = f"tm_visual_{link_index}_{visual_index}_v"
                resource = (
                    visual.geometry.mesh.filename
                    if visual.geometry is not None and visual.geometry.mesh is not None
                    else None
                )
                self._visual_tokens[token] = (link.name, visual_index, resource)
                visual.name = token
        self._flatten_mimics(parsed)
        self._urdf = URDF(
            robot=parsed.robot, build_scene_graph=True,
            build_collision_scene_graph=False, load_meshes=True,
            load_collision_meshes=False, force_mesh=False,
            filename_handler=self._resolve_resource,
        )
        roots = set(self._urdf.link_map) - {
            joint.child for joint in self._urdf.robot.joints
        }
        if len(roots) != 1:
            raise ModelError("The URDF must have one connected root")
        self.root_link = self._urdf.base_link
        self._origins = {
            joint.name: np.array(
                joint.origin if joint.origin is not None else np.eye(4), copy=True
            )
            for joint in self._urdf.robot.joints
        }
        self._parents = {joint.child: joint for joint in self._urdf.robot.joints}
        self.resource_hashes = FrozenMap({
            key: _sha256(path) for key, path in self._resources.items()
        })
        self._geometry = self._extract_geometry()

    def with_config(self, config):
        if (
            config is None or Path(config.path).resolve() != self.path
            or config.packages != self.config.packages
        ):
            raise ModelError(
                "Reload the model after changing its resource configuration"
            )
        configured = copy.copy(self)
        configured.config = config
        return configured

    def _resolve_resource(self, fname):
        if fname.startswith("package://"):
            parts = fname[len("package://"):].split("/", 1)
            if len(parts) != 2 or parts[0] not in self.config.packages:
                raise ModelError(f"No local package mapping for {fname}")
            path = Path(self.config.packages[parts[0]]) / parts[1]
        else:
            if "://" in fname:
                raise ModelError(f"Only local model resources are supported: {fname}")
            path = self.path.parent / Path(fname).expanduser()
        path = path.resolve()
        if path.suffix.lower() not in (".stl", ".obj"):
            raise ModelError(f"Unsupported URDF mesh resource: {path}")
        if not path.is_file():
            raise ModelError(f"Missing model resource: {path}")
        self._resources[fname] = path
        return str(path)

    @staticmethod
    def _flatten_mimics(parsed):
        joints = parsed.joint_map

        def resolve(name, ancestors):
            if name in ancestors or name not in joints:
                raise ModelError(
                    "Mimic references form a cycle or name a missing joint"
                )
            joint = joints[name]
            if joint.type == "fixed":
                raise ModelError("A mimic source must be actuated")
            if joint.mimic is None:
                return name, 1.0, 0.0
            source, scale, offset = resolve(joint.mimic.joint, (*ancestors, name))
            return (
                source, joint.mimic.multiplier * scale,
                joint.mimic.multiplier * offset + joint.mimic.offset,
            )

        for joint in parsed.robot.joints:
            if joint.mimic is not None:
                source, scale, offset = resolve(joint.name, ())
                if (joint.type == "prismatic") != (joints[source].type == "prismatic"):
                    raise ModelError("Mimic joints cannot mix linear and angular units")
                joint.mimic.joint = source
                joint.mimic.multiplier = scale
                joint.mimic.offset = offset

    def _extract_geometry(self):
        scene, pieces, counts = self._urdf.scene, [], {}
        for node in scene.graph.nodes_geometry:
            _, name = scene.graph.get(node)
            token = next((
                item for item in self._visual_tokens
                if name == item or name.startswith(item + "_")
            ), None)
            if token is None:
                raise ModelError("Cannot identify a visual instance in the URDF scene")
            link, visual_index, resource = self._visual_tokens[token]
            ordinal = counts.get(token, 0)
            counts[token] = ordinal + 1
            key = f"{link}/visual:{visual_index}/instance:{ordinal}"
            transform = self._urdf.get_transform(node, frame_from=link)
            transformed = scene.geometry[name].copy()
            transformed.apply_transform(transform)
            # apply_transform fixes reflection winding; rebuild normals after scale.
            mesh = trimesh.Trimesh(
                vertices=transformed.vertices, faces=transformed.faces, process=False
            )
            contents = self.resource_hashes["urdf"]
            if resource:
                contents += self.resource_hashes[resource]
            digest = hashlib.sha256(contents.encode()).hexdigest()
            pieces.append(LocalGeometry(key, link, digest, mesh))
        return tuple(pieces)

    @property
    def links(self):
        return tuple(self._urdf.link_map)

    @property
    def joints(self):
        return tuple(FrozenMap({
            "name": joint.name, "type": joint.type,
            "lower": joint.limit.lower if joint.limit else None,
            "upper": joint.limit.upper if joint.limit else None,
            "mimic": joint.mimic.joint if joint.mimic else None,
        }) for joint in self._urdf.robot.joints)

    @property
    def actuated_joint_names(self):
        return tuple(self._urdf.actuated_joint_names)

    @property
    def triangle_count(self):
        return sum(len(piece.faces) for piece in self._geometry)

    def _configure(self, q, adjustments):
        if any(name not in self.actuated_joint_names for name in q):
            raise ModelError("Configuration contains an unknown or nonactuated joint")
        values = {}
        for name in self.actuated_joint_names:
            value = q.get(name, 0.0)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ModelError(f"Joint position must be numeric: {name}")
            try:
                values[name] = float(value)
            except OverflowError as error:
                raise ModelError(
                    f"Joint position exceeds preview range: {name}"
                ) from error
        if not all(np.isfinite(value) for value in values.values()):
            raise ModelError("Joint positions must be finite")
        if any(link not in self.links for link in adjustments):
            raise ModelError("A link adjustment references a missing link")
        for joint in self._urdf.robot.joints:
            adjustment = adjustments.get(joint.child)
            correction = (
                pose_matrix(adjustment) if adjustment is not None else np.eye(4)
            )
            joint.origin = correction @ self._origins[joint.name]
            if joint.type == "fixed":
                values[joint.name] = 0.0
        # Include fixed edges after their origin changes; never accumulate adjustments.
        self._urdf.update_cfg(values)
        adjustment = adjustments.get(self.root_link)
        return pose_matrix(adjustment) if adjustment is not None else np.eye(4)

    def link_pose(self, link, q, adjustments=None):
        if link not in self.links:
            raise KeyError(link)
        with self._lock:
            root = self._configure(
                q, self.config.link_adjustments if adjustments is None else adjustments
            )
            return immutable_array(root @ self._urdf.get_transform(link))

    def surface_snapshot(self, q, *, root_pose=None, adjustments=None):
        with self._lock:
            root = self._configure(
                q, self.config.link_adjustments if adjustments is None else adjustments
            )
            view = np.linalg.inv(root_pose) if root_pose is not None else np.eye(4)
            instances = tuple(
                SurfaceInstance(
                    piece, view @ root @ self._urdf.get_transform(piece.link)
                )
                for piece in self._geometry
            )
        return SurfaceSnapshot(instances)

    def adjustment_for_pose(self, link, q, target_model_pose, adjustments=None):
        adjustments = (
            self.config.link_adjustments if adjustments is None else adjustments
        )
        if link == self.root_link:
            return matrix_pose(target_model_pose)
        joint = self._parents[link]
        parent = self.link_pose(joint.parent, q, adjustments)
        current = np.linalg.inv(parent) @ self.link_pose(link, q, adjustments)
        previous = pose_matrix(adjustments[link]) if link in adjustments else np.eye(4)
        return matrix_pose(
            np.linalg.inv(parent) @ target_model_pose
            @ np.linalg.inv(current) @ previous
        )

    def resource_issues(self):
        issues = []
        for key, path in self._resources.items():
            if not path.is_file() or _sha256(path) != self.resource_hashes[key]:
                issues.append(Issue(
                    "error", "resource_changed", f"Model resource changed: {path}",
                    field="model",
                ))
        for key, expected in self.config.asset_hashes.items():
            if self.resource_hashes.get(key) != expected:
                issues.append(Issue(
                    "error", "resource_changed", f"Saved resource hash differs: {key}",
                    field="model",
                ))
        return tuple(issues)

    def configuration_issues(self, q):
        issues = []
        for name, value in q.items():
            if name not in self.actuated_joint_names:
                issues.append(Issue(
                    "error", "joint_name", f"Unknown free joint: {name}",
                    field="bind_configuration",
                ))
                continue
            if (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or (isinstance(value, float) and not np.isfinite(value))
            ):
                issues.append(Issue(
                    "error", "joint_value", f"Invalid joint value: {name}",
                    field="bind_configuration",
                ))
                continue
            limit = self._urdf.joint_map[name].limit
            if limit is not None and (
                (limit.lower is not None and value < limit.lower)
                or (limit.upper is not None and value > limit.upper)
            ):
                issues.append(Issue(
                    "warning", "joint_limit", f"Joint exceeds its URDF limit: {name}",
                    field="bind_configuration",
                ))
        return tuple(issues)

    def attachment_valid(self, attachment):
        """Check cached identity after resource_issues ran once for this task."""
        if not attachment:
            return False
        return any(
            piece.key == attachment.get("geometry_key")
            and piece.resource_hash == attachment.get("resource_hash")
            and type(attachment.get("triangle")) is int
            and 0 <= attachment["triangle"] < len(piece.faces)
            for piece in self._geometry
        )


def load_urdf(asset: ModelConfig) -> Model:
    return Model(asset)

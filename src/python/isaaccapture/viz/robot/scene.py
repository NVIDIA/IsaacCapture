# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A scene file, drawn as a robot twin. The only module here that names a scene backend.

The backend is a private MuJoCo shipped beside ``_robot_twin`` and reached through
``dlsym``, so whatever ``mujoco`` the caller's environment has is unrelated to it. Nothing
outside this module holds a scene handle, which is what keeps that true.

Not a scene graph: four things move (a body's pose, the joint array, a group's visibility,
a material's colour) and one is measured, a frame's offset from another, once, at load. Do
not add a per-frame ``pose_of(site)``; it would be a kinematics service on the render
thread. The thread boundary is :meth:`SceneTwin.publish` -- after
:meth:`SceneTwin.create` the scene belongs to the render thread alone.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from . import _robot_twin, quaternion
from .joint_map import JointMap

LOG = logging.getLogger(__name__)

#: ``geom_group`` values a scene is expected to use. Robot MJCFs number their visual geoms
#: 2 and their collision geoms 3, and only 2 is in the visualiser's default geomgroup mask,
#: so moving a geom to 3 is what hides it.
DRAWN_GROUP = 2
HIDDEN_GROUP = 3

#: The scene's root body, so a caller can ask for a pose in world coordinates.
WORLD_BODY = "world"


def _require(scene, obj_type, kind: str, name: str, hint: str) -> int:
    index = scene.id(obj_type, name)
    if index < 0:
        raise RuntimeError(
            f"robot twin: the scene declares no `{name}` {kind}."
            + (f" {hint}" if hint else "")
        )
    return index


class SceneTwin:
    """A compiled scene, drawn into an XR session's projection layer.

    Implements both halves of :class:`~isaaccapture.viz.robot.RobotTwinPublisher`:
    :meth:`publish` is the control thread's, everything else the render thread's. Nothing
    is integrated; the invariant that replaces a simulation step is that every published
    change is followed by exactly one forward-kinematics pass, which :meth:`render` owns.
    A written mocap row is an input to that pass, not yet a drawn pose.
    """

    def __init__(
        self,
        scene_path: Path | str,
        *,
        gl_device_index: int = -1,
    ) -> None:
        """Compile the scene. No GPU is touched until :meth:`create`.

        Args:
            scene_path: Scene file to compile. Absolute, or a nested include resolves
                against the wrong directory.
            gl_device_index: Which GPU to make the OpenGL context on. -1 takes the first
                that yields one, which is right on a single-GPU machine; :meth:`create`
                reports it when the choice disagrees with the one the compositor picked.

        Raises:
            RuntimeError: If the scene does not compile, with the backend version named.
        """
        try:
            self._scene = _robot_twin.Scene(str(scene_path))
        except Exception as error:
            raise RuntimeError(
                f"robot twin: {scene_path} did not compile against MuJoCo "
                f"{_robot_twin.mujoco_version()}, which is the version this twin is "
                f"built with: {error}"
            ) from error
        self._gl_device_index = gl_device_index
        self._joints = self._read_joint_map()

        # Named show/hide sets and materials, resolved once by declare_*.
        self._groups: dict[str, np.ndarray] = {}
        self._materials: dict[str, int] = {}

        self._gl_context = None
        self._renderer = None
        self._guide = None
        self._clutch_engaged = False

        # The only state two threads touch. Merged into, not replaced: a publisher that
        # sends only what changed must not blank what it left out. A plain dict under the
        # GIL would tear between the read and the clear, hence the lock.
        self._lock = threading.Lock()
        self._pending_clutch_engaged: bool | None = None
        self._pending_joints: np.ndarray | None = None
        self._pending_bodies: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._pending_groups: dict[str, bool] = {}
        self._pending_materials: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ load time

    @property
    def joints(self) -> JointMap:
        """Motor name to address, in the order :meth:`publish` takes them."""
        return self._joints

    @property
    def backend_version(self) -> str:
        """The scene backend compiled into this wheel. For a startup report."""
        return _robot_twin.mujoco_version()

    def declare_group(
        self,
        name: str,
        *,
        body: str | None = None,
        geoms: Sequence[str] = (),
        drawn_only: bool = False,
    ) -> None:
        """Name a set of geoms that show and hide together.

        Exactly one of ``body`` (its whole subtree) or ``geoms`` (named individually, so a
        renaming upstream is an error rather than an invisible tool). ``drawn_only`` keeps
        just the geoms the scene already draws, so showing a subtree cannot reveal a
        robot's authored-hidden collision geoms.

        Raises:
            ValueError: If neither or both of ``body`` and ``geoms`` are given.
            RuntimeError: If the scene declares no such body or geom, or the group ends
                up covering nothing -- an empty group silently never draws.
        """
        if (body is None) == (not geoms):
            raise ValueError("declare_group takes exactly one of `body` or `geoms`")
        if body is not None:
            ids = self._subtree_geoms(self.body_id(body))
        else:
            ids = np.array(
                [
                    _require(self._scene, _robot_twin.ObjType.GEOM, "geom", n, "")
                    for n in geoms
                ],
                dtype=np.int32,
            )
        if drawn_only:
            ids = ids[self._scene.geom_group[ids] == DRAWN_GROUP]
        if ids.size == 0:
            raise RuntimeError(
                f"robot twin: group `{name}` covers no geom"
                + (
                    f" in draw group {DRAWN_GROUP}; the scene's groups changed"
                    if drawn_only
                    else ""
                )
            )
        self._groups[name] = ids

    def declare_material(self, name: str, *, hint: str = "") -> np.ndarray:
        """Name a material :meth:`publish` may recolour; returns its authored rgba."""
        index = _require(
            self._scene, _robot_twin.ObjType.MATERIAL, "material", name, hint
        )
        self._materials[name] = index
        return np.array(self._scene.mat_rgba[index], dtype=float)

    def repaint(self, group: str, material: str) -> None:
        """Point every geom in ``group`` at ``material``, so it recolours in one write.
        Load time only: the renderer uploads the geom-to-material map once.
        """
        self._scene.geom_matid[self._groups[group]] = self._materials[material]

    def body_id(self, name: str) -> int:
        """The body called ``name``.

        Raises:
            RuntimeError: If the scene declares no such body.
        """
        return _require(self._scene, _robot_twin.ObjType.BODY, "body", name, "")

    def home(self, joints: Sequence[float]) -> None:
        """Pose the scene once, at load, and settle it: what :meth:`body_offset` measures
        against, and the only write to the joint array that is not a publish.
        """
        self._joints.scatter(joints, self._scene.qpos)
        self._scene.forward()

    def body_offset(
        self, of: str, *, relative_to: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(pos, quat_wxyz)`` of one body in another's frame, in the home posture.

        Measured once and composed by the caller ever after; there is deliberately no
        per-frame equivalent.
        """
        child = self.body_id(of)
        return self._relative(
            np.array(self._scene.xpos[child], dtype=float),
            np.array(self._scene.xquat[child], dtype=float),
            self.body_id(relative_to),
        )

    def site_offset(
        self, of: str, *, relative_to: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(pos, quat_wxyz)`` of a site in a body's frame, in the home posture."""
        site = _require(self._scene, _robot_twin.ObjType.SITE, "site", of, "")
        return self._relative(
            np.array(self._scene.site_xpos[site], dtype=float),
            quaternion.from_matrix(np.array(self._scene.site_xmat[site], dtype=float)),
            self.body_id(relative_to),
        )

    # ------------------------------------------------------------------ control thread

    def publish(
        self,
        joints: Sequence[float] | None = None,
        *,
        bodies: Mapping[str, tuple[Sequence[float], Sequence[float]]] | None = None,
        groups: Mapping[str, bool] | None = None,
        materials: Mapping[str, Sequence[float]] | None = None,
        clutch_engaged: bool | None = None,
    ) -> None:
        """Record a scene change. Safe from any thread; nothing is drawn or posed here.

        Merged into whatever is already pending and applied whole on the next
        :meth:`render`, so a caller may send only what moved. Latest wins.

        Args:
            joints: One value per :attr:`joints` name.
            bodies: Body name -> ``(position, quat_wxyz)`` in scene world coordinates. A
                mocap body is moved as one and any other by its frame offset; the caller
                does not choose.
            groups: Group name -> drawn, over the sets :meth:`declare_group` named.
            materials: Material name -> rgba, over the ones :meth:`declare_material`
                named.
            clutch_engaged: Which controller guide to show; None preserves the state.
        """
        # Converted outside the lock: caller code must not run with the render thread
        # blocked behind it.
        joints = None if joints is None else np.array(joints, dtype=float)
        bodies = {
            name: (np.array(pos, dtype=float), np.array(quat, dtype=float))
            for name, (pos, quat) in (bodies or {}).items()
        }
        materials = {
            name: np.array(rgba, dtype=float)
            for name, rgba in (materials or {}).items()
        }
        clutch_engaged = None if clutch_engaged is None else bool(clutch_engaged)
        with self._lock:
            if clutch_engaged is not None:
                self._pending_clutch_engaged = clutch_engaged
            if joints is not None:
                self._pending_joints = joints
            self._pending_bodies.update(bodies)
            self._pending_groups.update(groups or {})
            self._pending_materials.update(materials)

    # ------------------------------------------------------------------ render thread

    @property
    def gl_device_index(self) -> int:
        """Which GPU the OpenGL context landed on. -1 before :meth:`create`."""
        return -1 if self._gl_context is None else self._gl_context.device_index

    def create(
        self, width: int, height: int, view_count: int, *, near_z: float, far_z: float
    ) -> None:
        """Build the OpenGL context and the renderer at the compositor's resolution."""
        self._gl_context = _robot_twin.GlContext(width, height, self._gl_device_index)
        self._gl_context.make_current()
        self._scene.disable_multisampling()

        self._renderer = _robot_twin.Renderer(
            scene=self._scene,
            width=width,
            height=height,
            view_count=view_count,
            near_z=near_z,
            far_z=far_z,
        )

        from .hud import ControllerGuide

        self._guide = ControllerGuide(near_z, far_z)

    def render(self, poses: Sequence[float], fovs: Sequence[float]) -> None:
        """Apply everything published, settle the scene, then draw every view.

        Raises:
            RuntimeError: If the scene overflowed the render scene's fixed capacity.
        """
        self.settle()
        self._renderer.update_scene()
        # The scene update truncates on overflow and returns normally, with only a
        # stderr warning nobody reads in a frame loop.
        if self._renderer.ngeom >= self._renderer.maxgeom:
            raise RuntimeError(
                f"robot twin: the render scene is full: ngeom={self._renderer.ngeom} "
                f"maxgeom={self._renderer.maxgeom}. Geometry is being dropped -- "
                "raise kMaxGeom in src/viz/robot_twin/cpp/scene_renderer.cpp."
            )
        self._renderer.render(poses, fovs)
        # settle() latches one published phase for both eyes.
        self._guide.draw(self._renderer, poses, fovs, self._clutch_engaged)

    def color(self, view: int):
        return self._guide.colors[view]

    def depth(self, view: int):
        return self._guide.depths[view]

    def frustum(self, view: int):
        return self._renderer.frustum(view)

    def destroy(self) -> None:
        """Innermost first: the renderer's GL objects need a current context."""
        self._guide = None
        try:
            if self._renderer is not None:
                renderer, self._renderer = self._renderer, None
                renderer.close()
        finally:
            self._gl_context = None

    def settle(self) -> None:
        """Drain the published changes onto the scene and run forward kinematics.

        The one forward pass per frame, which :meth:`render` calls. Public so a headless
        caller can pose the scene and read it back without a GPU.
        """
        with self._lock:
            if self._pending_clutch_engaged is not None:
                self._clutch_engaged = self._pending_clutch_engaged
                self._pending_clutch_engaged = None
            joints, self._pending_joints = self._pending_joints, None
            bodies, self._pending_bodies = self._pending_bodies, {}
            groups, self._pending_groups = self._pending_groups, {}
            materials, self._pending_materials = self._pending_materials, {}

        if joints is not None:
            self._joints.scatter(joints, self._scene.qpos)
        for name, (pos, quat) in bodies.items():
            body = self.body_id(name)
            mocap = int(self._scene.body_mocapid[body])
            if mocap >= 0:
                self._scene.mocap_pos[mocap] = pos
                self._scene.mocap_quat[mocap] = quat
            else:
                self._scene.body_pos[body] = pos
                self._scene.body_quat[body] = quat
        for name, drawn in groups.items():
            self._scene.geom_group[self._groups[name]] = (
                DRAWN_GROUP if drawn else HIDDEN_GROUP
            )
        for name, rgba in materials.items():
            self._scene.mat_rgba[self._materials[name]] = rgba

        self._scene.forward()

    # ------------------------------------------------------------------ internals

    def _read_joint_map(self) -> JointMap:
        """Every hinge in the scene, by name, mapped to its position address.

        Slide joints are left out rather than addressed: a snapshot is one value per name
        in radians, and a slide's position is metres, so the same array would pose it
        plausibly and wrongly. Out of the map nothing writes them and they hold their
        authored pose. A ball or free joint still raises -- it occupies four or seven
        slots, so no per-name snapshot could fill it either way.

        Raises:
            RuntimeError: If the scene carries a ball or free joint, or one with no name
                -- an unnamed joint cannot be published or asserted.
        """
        names: list[str] = []
        addresses: list[int] = []
        skipped: list[str] = []
        types = self._scene.jnt_type
        for joint in range(self._scene.njnt):
            name = self._scene.name(_robot_twin.ObjType.JOINT, joint)
            if not name:
                raise RuntimeError(
                    f"robot twin: joint {joint} has no name, so nothing can address it."
                )
            if types[joint] == int(_robot_twin.JointType.SLIDE):
                skipped.append(name)
                continue
            if types[joint] != int(_robot_twin.JointType.HINGE):
                raise RuntimeError(
                    f"robot twin: joint `{name}` is neither a hinge nor a slide; the twin "
                    "poses hinges and holds slides."
                )
            names.append(name)
            addresses.append(int(self._scene.jnt_qposadr[joint]))
        if skipped:
            LOG.info(
                "robot twin: holding %d slide joint(s) at their authored pose: %s",
                len(skipped),
                ", ".join(skipped),
            )
        return JointMap(names, addresses, width=int(self._scene.nq), skipped=skipped)

    def _subtree_geoms(self, root: int) -> np.ndarray:
        """Every geom on ``root`` and its descendants, by geom id. ``body_rootid`` is the
        top of a body's kinematic tree, so this holds only while ``root`` is a child of
        world.
        """
        roots = self._scene.body_rootid[self._scene.geom_bodyid]
        return np.where(roots == root)[0].astype(np.int32)

    def _relative(
        self, pos: np.ndarray, quat: np.ndarray, parent: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """A world pose expressed in ``parent``'s frame."""
        inverse = quaternion.conjugate(np.array(self._scene.xquat[parent], dtype=float))
        local_pos = quaternion.rotate(
            pos - np.array(self._scene.xpos[parent], dtype=float), inverse
        )
        return local_pos, quaternion.multiply(inverse, quat)

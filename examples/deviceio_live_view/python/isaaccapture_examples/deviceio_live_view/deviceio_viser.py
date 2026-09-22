# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeviceIO-to-viser rendering helpers for the live human-tracking viewer.

Viz classes: ``HandViz``, ``ControllerViz``, ``BodyViz``, ``HeadViz``,
``HumanDeviceIOViz``.

Rendering helpers: ``HAND_BONES``, ``BODY_BONES``, ``controller_state``.
"""

import numpy as np
import viser

from isaaccapture.retargeting_engine.tensor_types import HandInputIndex
from isaaccapture.retargeting_engine.tensor_types.indices import (
    ControllerInputIndex,
    HeadInputIndex,
)

from .body_pipeline import BodyViewLayout
from .full_body_pose import FULL_BODY_POSE_BONES, FULL_BODY_POSE_LAYOUT

BODY_JOINT_NAMES = list(FULL_BODY_POSE_LAYOUT.joint_names)
BODY_BONES = FULL_BODY_POSE_BONES

# ---------------------------------------------------------------------------
# Color palette shared across all viz scripts
# ---------------------------------------------------------------------------

LEFT_COLOR: tuple[float, float, float] = (0.25, 0.85, 0.35)
RIGHT_COLOR: tuple[float, float, float] = (0.35, 0.55, 0.95)
INVALID_COLOR: tuple[float, float, float] = (1.0, 0.0, 0.0)
TRACKED_COLOR: tuple[float, float, float] = (0.25, 0.85, 0.35)


# Duplicated verbatim in mcap_record_replay/common.py: each example package
# is self-contained (see examples/README.md), so this stays a copy rather
# than a cross-example dependency. Keep the two in sync by hand.
class GroundGrid:
    """The ground plane and the default camera, anchored to what is tracked.

    The session asks OpenXR for a stage (floor-relative) space, but a runtime
    that cannot supply one falls back to a head-relative origin: y=0 then sits
    at eye height and the skeleton hangs below a grid drawn at zero. Following
    the lowest tracked joint puts the grid on the floor in either space, and
    the camera is framed against that floor rather than against y=0 -- aiming
    at a fixed height leaves the subject at the bottom of the viewport in a
    head-relative space.
    """

    def __init__(self, server, handle, smoothing: float = 0.05):
        self._server = server
        self._handle = handle
        self._smoothing = smoothing
        self._y: float | None = None

        @server.on_client_connect
        def _(client) -> None:
            self._frame(client)

    def _frame(self, client) -> None:
        """Stand back from the floor at eye height, looking at torso height."""
        floor = 0.0 if self._y is None else self._y
        client.camera.position = (0.0, floor + 1.5, 2.5)
        client.camera.look_at = (0.0, floor + 0.9, 0.0)

    def follow(self, positions: np.ndarray, valid: np.ndarray) -> None:
        points = np.asarray(positions, dtype=np.float32)[np.asarray(valid, dtype=bool)]
        if points.size == 0:
            return
        lowest = float(np.min(points[:, 1]))
        # Ease toward it: a single mistracked frame should not drop the floor.
        first = self._y is None
        self._y = lowest if first else self._y + self._smoothing * (lowest - self._y)
        self._handle.position = (0.0, self._y, 0.0)

        # Re-aim once, when the floor is first known. Doing it every frame would
        # fight the mouse.
        if first:
            for client in self._server.get_clients().values():
                self._frame(client)


def setup_scene(server) -> GroundGrid:
    """Up axis, ground grid and a starting camera, shared by every viewer here.

    viser's ``add_grid`` defaults to the XY plane, which stands up as a wall
    once the up direction is +y -- it has to be ``xz`` to lie on the ground.
    Returns the grid so a caller with tracked joints can keep it on the floor.
    """
    server.scene.set_up_direction("+y")
    grid = server.scene.add_grid(
        name="/grid",
        width=6.0,
        height=6.0,
        plane="xz",
        cell_size=0.25,
        section_size=1.0,
    )

    return GroundGrid(server, grid)


# OpenXR hand-joint connectivity (parent → child) for skeleton rendering.
# Indices follow XR_HAND_JOINT_*_EXT: 0=PALM, 1=WRIST, thumb has 4 joints
# (no intermediate), the other 4 fingers have 5 joints each — 26 total.
HAND_BONES: tuple[tuple[int, int], ...] = (
    # Thumb
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    # Index
    (1, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    # Middle
    (1, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    # Ring
    (1, 16),
    (16, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    # Little
    (1, 21),
    (21, 22),
    (22, 23),
    (23, 24),
    (24, 25),
)


def _bone_segments(positions: np.ndarray) -> np.ndarray:
    """Return (N, 2, 3) segment array for the parent→child hand bones."""
    return np.stack(
        [np.stack([positions[a], positions[b]], axis=0) for a, b in HAND_BONES],
        axis=0,
    ).astype(np.float32)


def _valid_bone_segments(
    positions: np.ndarray,
    valid: np.ndarray,
    bones: tuple[tuple[int, int], ...] = BODY_BONES,
) -> np.ndarray:
    """Return (N, 2, 3) segment array for body bones whose both endpoints are valid."""
    segments: list[np.ndarray] = []
    for a, b in bones:
        if valid[a] and valid[b]:
            segments.append(np.stack([positions[a], positions[b]], axis=0))
    if not segments:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.stack(segments, axis=0).astype(np.float32)


def _segment(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    return np.stack([start, end], axis=0).astype(np.float32)


def controller_state(controller) -> dict:
    """Extract a plain-dict snapshot from a controller TensorGroup."""
    if controller.is_none:
        return {
            "aim_pos": None,
            "grip_pos": None,
            "aim_valid": False,
            "grip_valid": False,
            "trigger": 0.0,
            "squeeze": 0.0,
            "thumbstick_xy": (0.0, 0.0),
            "primary_click": False,
            "secondary_click": False,
            "thumbstick_click": False,
            "menu_click": False,
            "tracked": False,
        }

    aim_valid = bool(controller[ControllerInputIndex.AIM_IS_VALID])
    grip_valid = bool(controller[ControllerInputIndex.GRIP_IS_VALID])
    return {
        "aim_pos": np.asarray(
            controller[ControllerInputIndex.AIM_POSITION], dtype=np.float32
        ),
        "grip_pos": np.asarray(
            controller[ControllerInputIndex.GRIP_POSITION], dtype=np.float32
        ),
        "aim_valid": aim_valid,
        "grip_valid": grip_valid,
        "trigger": float(controller[ControllerInputIndex.TRIGGER_VALUE]),
        "squeeze": float(controller[ControllerInputIndex.SQUEEZE_VALUE]),
        "thumbstick_xy": (
            float(controller[ControllerInputIndex.THUMBSTICK_X]),
            float(controller[ControllerInputIndex.THUMBSTICK_Y]),
        ),
        "primary_click": float(controller[ControllerInputIndex.PRIMARY_CLICK]) > 0.5,
        "secondary_click": float(controller[ControllerInputIndex.SECONDARY_CLICK])
        > 0.5,
        "thumbstick_click": float(controller[ControllerInputIndex.THUMBSTICK_CLICK])
        > 0.5,
        "menu_click": float(controller[ControllerInputIndex.MENU_CLICK]) > 0.5,
        "tracked": aim_valid or grip_valid,
    }


class HandViz:
    """Per-hand viser handles (joint cloud + skeleton segments)."""

    def __init__(
        self,
        server: viser.ViserServer,
        name: str,
        color: tuple[float, float, float],
    ):
        self.color = np.array(color, dtype=np.float32)
        zero_pts = np.zeros((26, 3), dtype=np.float32)
        zero_segs = np.zeros((len(HAND_BONES), 2, 3), dtype=np.float32)

        self.points = server.scene.add_point_cloud(
            name=f"/{name}/joints",
            points=zero_pts,
            colors=np.tile(self.color, (26, 1)),
            point_size=0.008,
        )
        self.bones = server.scene.add_line_segments(
            name=f"/{name}/bones",
            points=zero_segs,
            colors=np.tile(self.color, (len(HAND_BONES), 2, 1)),
            line_width=2.0,
        )

    def update(self, positions: np.ndarray, valid: bool) -> None:
        if valid:
            self.points.points = positions.astype(np.float32)
            self.points.colors = np.tile(self.color, (positions.shape[0], 1))
            self.bones.points = _bone_segments(positions)
        else:
            zero_pts = np.zeros_like(positions, dtype=np.float32)
            self.points.points = zero_pts
            self.points.colors = np.tile(INVALID_COLOR, (positions.shape[0], 1))
            self.bones.points = np.zeros((len(HAND_BONES), 2, 3), dtype=np.float32)


class ControllerViz:
    """Per-controller viser handles (3D pose + live input-state HUD)."""

    def __init__(
        self,
        server: viser.ViserServer,
        name: str,
        color: tuple[float, float, float],
    ):
        self.color = np.array(color, dtype=np.float32)
        zero_pt = np.zeros((1, 3), dtype=np.float32)
        zero_seg = np.zeros((0, 2, 3), dtype=np.float32)
        zero_seg_colors = np.zeros((0, 2, 3), dtype=np.float32)

        self.aim = server.scene.add_point_cloud(
            name=f"/{name}/aim",
            points=zero_pt,
            colors=np.tile(self.color, (1, 1)),
            point_size=0.015,
        )
        self.grip = server.scene.add_point_cloud(
            name=f"/{name}/grip",
            points=zero_pt,
            colors=np.tile(self.color, (1, 1)),
            point_size=0.015,
        )
        self.ray = server.scene.add_line_segments(
            name=f"/{name}/ray",
            points=zero_seg,
            colors=zero_seg_colors,
            line_width=2.0,
        )

        with server.gui.add_folder(name):
            self.hud_tracking = server.gui.add_checkbox("tracked", False, disabled=True)
            self.hud_aim_valid = server.gui.add_checkbox(
                "aim_valid", False, disabled=True
            )
            self.hud_grip_valid = server.gui.add_checkbox(
                "grip_valid", False, disabled=True
            )
            self.hud_stick = server.gui.add_vector2(
                "thumbstick_xy",
                initial_value=(0.0, 0.0),
                min=(-1.0, -1.0),
                max=(1.0, 1.0),
                disabled=True,
            )
            self.hud_trigger_value = server.gui.add_number(
                "trigger",
                initial_value=0.0,
                min=0.0,
                max=1.0,
                step=0.01,
                disabled=True,
            )
            self.hud_trigger = server.gui.add_progress_bar(0.0)
            self.hud_squeeze_value = server.gui.add_number(
                "squeeze",
                initial_value=0.0,
                min=0.0,
                max=1.0,
                step=0.01,
                disabled=True,
            )
            self.hud_squeeze = server.gui.add_progress_bar(0.0)
            self.hud_primary = server.gui.add_checkbox(
                "primary_click", False, disabled=True
            )
            self.hud_secondary = server.gui.add_checkbox(
                "secondary_click", False, disabled=True
            )
            self.hud_stick_click = server.gui.add_checkbox(
                "thumbstick_click", False, disabled=True
            )
            self.hud_menu_click = server.gui.add_checkbox(
                "menu_click", False, disabled=True
            )

    def update(self, state: dict) -> None:
        aim_valid: bool = state["aim_valid"]
        grip_valid: bool = state["grip_valid"]
        aim_pos: np.ndarray | None = state["aim_pos"]
        grip_pos: np.ndarray | None = state["grip_pos"]

        self.hud_tracking.value = state["tracked"]
        self.hud_aim_valid.value = aim_valid
        self.hud_grip_valid.value = grip_valid
        self.hud_stick.value = state["thumbstick_xy"]
        self.hud_trigger.value = max(0.0, min(1.0, state["trigger"]))
        self.hud_trigger_value.value = state["trigger"]
        self.hud_squeeze.value = max(0.0, min(1.0, state["squeeze"]))
        self.hud_squeeze_value.value = state["squeeze"]
        self.hud_primary.value = state["primary_click"]
        self.hud_secondary.value = state["secondary_click"]
        self.hud_stick_click.value = state["thumbstick_click"]
        self.hud_menu_click.value = state["menu_click"]

        if aim_valid and aim_pos is not None:
            self.aim.points = aim_pos.reshape(1, 3).astype(np.float32)
            self.aim.colors = np.tile(self.color, (1, 1))
        else:
            self.aim.points = np.zeros((1, 3), dtype=np.float32)
            self.aim.colors = np.tile(INVALID_COLOR, (1, 1))

        if grip_valid and grip_pos is not None:
            self.grip.points = grip_pos.reshape(1, 3).astype(np.float32)
            self.grip.colors = np.tile(self.color, (1, 1))
        else:
            self.grip.points = np.zeros((1, 3), dtype=np.float32)
            self.grip.colors = np.tile(INVALID_COLOR, (1, 1))

        if aim_valid and grip_valid and aim_pos is not None and grip_pos is not None:
            seg = _segment(grip_pos, aim_pos).reshape(1, 2, 3)
            self.ray.points = seg
            self.ray.colors = np.tile(self.color, (1, 2, 1))
        else:
            self.ray.points = np.zeros((0, 2, 3), dtype=np.float32)
            self.ray.colors = np.zeros((0, 2, 3), dtype=np.float32)


class BodyViz:
    """Viser handles for a body skeleton (joint cloud + skeleton segments)."""

    def __init__(self, server: viser.ViserServer, layout: BodyViewLayout):
        self.layout = layout
        self.color = np.array(TRACKED_COLOR, dtype=np.float32)
        zero_pts = np.zeros((len(layout.joint_names), 3), dtype=np.float32)
        zero_segs = np.zeros((0, 2, 3), dtype=np.float32)

        self.points = server.scene.add_point_cloud(
            name="/body/joints",
            points=zero_pts,
            colors=np.tile(self.color, (len(layout.joint_names), 1)),
            point_size=0.01,
        )
        self.bones = server.scene.add_line_segments(
            name="/body/bones",
            points=zero_segs,
            colors=np.zeros((0, 2, 3), dtype=np.float32),
            line_width=2.0,
        )

    def update(self, positions: np.ndarray | None, valid: np.ndarray | None) -> None:
        if positions is None or valid is None:
            zero_pts = np.zeros((len(self.layout.joint_names), 3), dtype=np.float32)
            self.points.points = zero_pts
            self.points.colors = np.tile(
                INVALID_COLOR, (len(self.layout.joint_names), 1)
            )
            self.bones.points = np.zeros((0, 2, 3), dtype=np.float32)
            self.bones.colors = np.zeros((0, 2, 3), dtype=np.float32)
            return

        positions = positions.astype(np.float32)
        valid_bool = valid.astype(bool)
        self.points.points = positions

        point_colors = np.tile(self.color, (positions.shape[0], 1))
        point_colors[~valid_bool] = INVALID_COLOR
        self.points.colors = point_colors

        segs = _valid_bone_segments(positions, valid_bool, self.layout.bones)
        self.bones.points = segs
        self.bones.colors = np.tile(self.color, (segs.shape[0], 2, 1))


def _xyzw_to_wxyz(orientation: np.ndarray) -> tuple[float, float, float, float]:
    quat = np.asarray(orientation, dtype=np.float32).reshape(4)
    return (float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]))


class HeadViz:
    """Head pose frame (hidden when tracking is inactive)."""

    def __init__(
        self,
        server: viser.ViserServer,
        name: str = "head",
        axes_length: float = 0.12,
    ):
        self.frame = server.scene.add_frame(
            name=f"/{name}/frame",
            axes_length=axes_length,
            axes_radius=0.004,
            visible=False,
        )

    def update_if_active(self, head) -> bool:
        if head.is_none or not bool(head[HeadInputIndex.IS_VALID]):
            self.frame.visible = False
            return False

        position = np.asarray(head[HeadInputIndex.POSITION], dtype=np.float32)
        orientation = np.asarray(head[HeadInputIndex.ORIENTATION], dtype=np.float32)
        self.frame.position = (
            float(position[0]),
            float(position[1]),
            float(position[2]),
        )
        self.frame.wxyz = _xyzw_to_wxyz(orientation)
        self.frame.visible = True
        return True


class HumanDeviceIOViz:
    """Aggregate viser handles for all human DeviceIO trackers.

    Inactive or absent trackers are hidden instead of drawn in the invalid color.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        ground: GroundGrid | None = None,
        body_layout: BodyViewLayout = FULL_BODY_POSE_LAYOUT,
    ):
        self._ground = ground
        self._body_layout = body_layout
        self.hand_left = HandViz(server, "hand_left", LEFT_COLOR)
        self.hand_right = HandViz(server, "hand_right", RIGHT_COLOR)
        self.head = HeadViz(server)
        self.controller_left = ControllerViz(server, "controller_left", LEFT_COLOR)
        self.controller_right = ControllerViz(server, "controller_right", RIGHT_COLOR)
        self.body = BodyViz(server, body_layout)

    def _update_hand_if_active(self, viz: HandViz, hand) -> bool:
        if hand.is_none:
            viz.points.visible = False
            viz.bones.visible = False
            return False

        positions = np.asarray(hand[HandInputIndex.JOINT_POSITIONS], dtype=np.float32)
        viz.points.visible = True
        viz.bones.visible = True
        viz.update(positions, valid=True)
        return True

    def _update_controller_if_active(self, viz: ControllerViz, controller) -> bool:
        state = controller_state(controller)
        if not state["tracked"]:
            viz.aim.visible = False
            viz.grip.visible = False
            viz.ray.visible = False
            viz.update(state)
            return False

        viz.aim.visible = True
        viz.grip.visible = True
        viz.ray.visible = True
        viz.update(state)
        return True

    def _update_body_if_active(self, body) -> tuple[bool, int]:
        if body.is_none:
            self.body.points.visible = False
            self.body.bones.visible = False
            return False, 0

        positions = np.asarray(
            body[self._body_layout.positions_index], dtype=np.float32
        )
        valid = np.asarray(body[self._body_layout.valid_index], dtype=np.uint8)
        n_valid = int(np.count_nonzero(valid))
        if n_valid == 0:
            self.body.points.visible = False
            self.body.bones.visible = False
            return False, 0

        self.body.points.visible = True
        self.body.bones.visible = True
        self.body.update(positions, valid)
        if self._ground is not None:
            self._ground.follow(positions, valid)
        return True, n_valid

    def update(self, result) -> dict[str, bool | int]:
        """Update every tracker; hide inactive ones. Returns active flags."""
        body_active, body_joints = self._update_body_if_active(result["body"])
        return {
            "hand_left": self._update_hand_if_active(
                self.hand_left, result["hand_left"]
            ),
            "hand_right": self._update_hand_if_active(
                self.hand_right, result["hand_right"]
            ),
            "head": self.head.update_if_active(result["head"]),
            "controller_left": self._update_controller_if_active(
                self.controller_left, result["controller_left"]
            ),
            "controller_right": self._update_controller_if_active(
                self.controller_right, result["controller_right"]
            ),
            "body_active": body_active,
            "body_joints": body_joints,
        }

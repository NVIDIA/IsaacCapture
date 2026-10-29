# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pipeline and visualization helpers for the record / replay / live scripts.

Pipeline builders: ``build_hand_pipeline``, ``build_controller_pipeline``,
``build_full_body_pipeline``.

Viz classes (used by replay_* and live_* scripts): ``HandViz``,
``ControllerViz``, ``FullBodyViz``.

Rendering helpers: ``HandJoints``, ``HAND_BONES``, ``BODY_BONES``.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import viser
from mcap.reader import make_reader

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.deviceio_session import DeviceIOSession, McapRecordingConfig
from isaaccapture.deviceio import McapReplayConfig
from isaaccapture.oxr import OpenXRSession
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    ControllersSource,
    FullBodySource,
    HandsSource,
)
from isaaccapture.retargeting_engine.interface import (
    BaseRetargeter,
    OutputCombiner,
)
from isaaccapture.retargeting_engine.interface.retargeter_core_types import (
    ComputeContext,
    RetargeterIO,
)
from isaaccapture.retargeting_engine.interface.tensor_group_type import (
    OptionalType,
    TensorGroupType,
)
from isaaccapture.retargeting_engine.tensor_types import (
    NUM_HAND_JOINTS,
    BoolType,
    HandInput,
    HandInputIndex,
)
from isaaccapture.retargeting_engine.tensor_types.indices import (
    BodyJointIndex,
    ControllerInputIndex,
    FullBodyInputIndex,
)
from isaaccapture.retargeting_engine.tensor_types.ndarray_types import (
    DLDataType,
    NDArrayType,
)
from isaaccapture.teleop_session_manager import (
    SessionMode,
    TeleopSession,
    TeleopSessionConfig,
)


def add_body_arguments(parser):
    from .soma_body_common import add_body_arguments as add_soma_arguments

    parser.add_argument(
        "--body-schema", choices=["full-body-pose", "soma"], default="full-body-pose"
    )
    parser.add_argument(
        "--controllers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include controllers (default: enabled for FullBodyPose, disabled for SOMA)",
    )
    add_soma_arguments(parser)


def validate_body_arguments(parser, args):
    if args.controllers is None:
        args.controllers = args.body_schema == "full-body-pose"
    if args.body_schema == "soma":
        if not args.soma_body_collection_id:
            parser.error("soma-body-collection-id must not be empty")
        # Source names also identify MCAP channel bases and must remain distinct.
        if args.controllers and args.soma_body_collection_id == "controllers":
            parser.error(
                "soma-body-collection-id 'controllers' is reserved when --controllers is enabled"
            )


def add_hand_arguments(parser):
    from .soma_hand_common import add_hand_arguments as add_soma_arguments

    parser.add_argument(
        "--hand-schema",
        choices=["openxr-hand-pose", "soma"],
        default="openxr-hand-pose",
    )
    add_soma_arguments(parser)


def validate_hand_arguments(parser, args):
    if args.hand_schema == "soma":
        from .soma_hand_common import collections

        try:
            collections(
                args.soma_left_hand_collection_id, args.soma_right_hand_collection_id
            )
        except ValueError as error:
            parser.error(str(error))


def build_body_pipeline(args):
    # Select the body independently; fingers in SOMA do not enable another hand source.
    if args.body_schema == "soma":
        from .soma_body_common import build_pipeline

        pipeline = build_pipeline(
            args.soma_body_collection_id, args.soma_body_representation
        )
        if args.controllers:
            controllers = ControllersSource(name="controllers")
            return OutputCombiner(
                {
                    "soma_body": pipeline.output("soma_body"),
                    "controller_left": controllers.output(ControllersSource.LEFT),
                    "controller_right": controllers.output(ControllersSource.RIGHT),
                }
            )
        return pipeline
    return build_full_body_pipeline(args.controllers)


def create_body_visualizer(args, server, ground=None):
    if args.body_schema == "soma":
        from .soma_body_common import SomaBodyViz

        return SomaBodyViz(server, ground)
    return FullBodyViz(server, ground)


def update_body_visualizer(args, result, viz):
    if args.body_schema == "soma":
        return viz.update(result["soma_body"])
    body = result["full_body"]
    if body.is_none:
        viz.update(None, None)
        return 0
    valid = np.asarray(body[FullBodyInputIndex.JOINT_VALID], dtype=np.uint8)
    viz.update(np.asarray(body[FullBodyInputIndex.JOINT_POSITIONS]), valid)
    return int(np.count_nonzero(valid))


def build_selected_hand_pipeline(args):
    if args.hand_schema == "soma":
        from .soma_hand_common import build_pipeline, collections

        return build_pipeline(
            collections(
                args.soma_left_hand_collection_id, args.soma_right_hand_collection_id
            ),
            args.soma_hand_representation,
        )
    return build_hand_pipeline()


def create_hand_visualizers(args, server):
    if args.hand_schema == "soma":
        from .soma_hand_common import create_visualizers

        return create_visualizers(server)
    return {
        "hand_left": HandViz(server, "hand_left", LEFT_COLOR),
        "hand_right": HandViz(server, "hand_right", RIGHT_COLOR),
    }


def update_hand_visualizers(args, result, visualizers):
    valid = []
    for side in ("left", "right"):
        viz = visualizers[f"hand_{side}"]
        if args.hand_schema == "soma":
            valid.append(bool(viz.update(result[f"hand_{side}"])))
        else:
            available = bool(result[f"{side}_valid"][0])
            viz.update(np.asarray(result[f"{side}_positions"][0]), available)
            valid.append(available)
    return valid


def record_trackers(args, path, tracker_names):
    """Record native inputs without preparing a model or evaluating a source graph."""
    trackers = [tracker for tracker, _ in tracker_names]
    config = McapRecordingConfig(str(path), tracker_names)
    extensions = DeviceIOSession.get_required_extensions(trackers)
    with CloudXRLauncher.launch_context(args):
        with OpenXRSession("McapRecordExample", extensions) as oxr:
            with DeviceIOSession.run(trackers, oxr.get_handles(), config) as session:
                start = time.monotonic()
                frames = 0
                # Only active recording handles Ctrl+C as a graceful completion.
                try:
                    while time.monotonic() - start < args.duration:
                        session.update()
                        frames += 1
                        if frames % 60 == 0:
                            print(f"[record] frame={frames}")
                        time.sleep(1 / 60)
                except KeyboardInterrupt:
                    pass


def resolve_recording(path_arg, prefix):
    if path_arg:
        path = Path(path_arg)
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist")
        return path
    recordings = Path.cwd() / "recordings"
    candidates = list(recordings.glob(f"{prefix}_*.mcap"))
    if not candidates:
        raise FileNotFoundError(f"No {prefix}_*.mcap in {recordings}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def channel_info(path, topic, record_name):
    """Return the selected channel's sample count and average capture rate."""
    count = 0
    first = last = None
    with path.open("rb") as stream:
        for schema, _, message in make_reader(stream).iter_messages(
            topics=[topic], log_time_order=False
        ):
            if schema is None or schema.name != record_name:
                raise ValueError(f"{topic}: expected {record_name}")
            first = message.log_time if first is None else min(first, message.log_time)
            last = message.log_time if last is None else max(last, message.log_time)
            count += 1
    rate = (count - 1) * 1e9 / (last - first) if count > 1 and last > first else 60.0
    return count, rate


def replay_once(path, pipeline, frame_count, update, rate_hz):
    config = TeleopSessionConfig(
        app_name="McapReplayExample",
        pipeline=pipeline,
        mode=SessionMode.REPLAY,
        mcap_config=McapReplayConfig(str(path)),
    )
    # Native replay consumes one tracked record per step. None is not an EOF signal.
    with TeleopSession(config) as session:
        start = time.monotonic()
        for frame in range(frame_count):
            update(session.step())
            if rate_hz > 0 and frame + 1 < frame_count:
                remaining = start + (frame + 1) / rate_hz - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
    return frame_count


# Duplicated verbatim in deviceio_live_view/deviceio_viser.py: each example
# package is self-contained (see examples/README.md), so this stays a copy
# rather than a cross-example dependency. Keep the two in sync by hand.
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


_ZERO_POSITIONS = np.zeros((NUM_HAND_JOINTS, 3), dtype=np.float32)


HANDS_CHANNEL = "hands"
BODY_JOINT_NAMES = [joint.name for joint in BodyJointIndex]

# ---------------------------------------------------------------------------
# Color palette shared across all viz scripts
# ---------------------------------------------------------------------------

LEFT_COLOR: tuple[float, float, float] = (0.25, 0.85, 0.35)
RIGHT_COLOR: tuple[float, float, float] = (0.35, 0.55, 0.95)
INVALID_COLOR: tuple[float, float, float] = (1.0, 0.0, 0.0)
TRACKED_COLOR: tuple[float, float, float] = (0.25, 0.85, 0.35)


def _positions_group(name: str) -> TensorGroupType:
    return TensorGroupType(
        name,
        [
            NDArrayType(
                "positions",
                shape=(NUM_HAND_JOINTS, 3),
                dtype=DLDataType.FLOAT,
                dtype_bits=32,
            )
        ],
    )


class HandJoints(BaseRetargeter):
    """Passes hand joint positions + validity through for downstream consumers.

    Zero-fills positions when a hand is not tracked so downstream code can read
    a fixed-shape array every frame.
    """

    def input_spec(self):
        return {
            HandsSource.LEFT: OptionalType(HandInput()),
            HandsSource.RIGHT: OptionalType(HandInput()),
        }

    def output_spec(self):
        return {
            "left_positions": _positions_group("left_positions"),
            "right_positions": _positions_group("right_positions"),
            "left_valid": TensorGroupType("left_valid", [BoolType("v")]),
            "right_valid": TensorGroupType("right_valid", [BoolType("v")]),
        }

    def _compute_fn(
        self, inputs: RetargeterIO, outputs: RetargeterIO, context: ComputeContext
    ) -> None:
        for side, key in (("left", HandsSource.LEFT), ("right", HandsSource.RIGHT)):
            optional_hand = inputs[key]
            if optional_hand.is_none:
                outputs[f"{side}_valid"][0] = False
                outputs[f"{side}_positions"][0] = _ZERO_POSITIONS.copy()
            else:
                outputs[f"{side}_valid"][0] = True
                outputs[f"{side}_positions"][0] = np.asarray(
                    optional_hand[HandInputIndex.JOINT_POSITIONS],
                    dtype=np.float32,
                )


def build_hand_pipeline():
    hands = HandsSource(name=HANDS_CHANNEL)
    joints = HandJoints(name="hand_joints")
    return joints.connect(
        {
            HandsSource.LEFT: hands.output(HandsSource.LEFT),
            HandsSource.RIGHT: hands.output(HandsSource.RIGHT),
        }
    )


def build_controller_pipeline():
    controllers = ControllersSource(name="controllers")
    return OutputCombiner(
        {
            "controller_left": controllers.output(ControllersSource.LEFT),
            "controller_right": controllers.output(ControllersSource.RIGHT),
        }
    )


def build_full_body_pipeline(include_controllers=True):
    full_body = FullBodySource(name="full_body")
    outputs = {"full_body": full_body.output(FullBodySource.FULL_BODY)}
    if include_controllers:
        controllers = ControllersSource(name="controllers")
        outputs.update(
            {
                "controller_left": controllers.output(ControllersSource.LEFT),
                "controller_right": controllers.output(ControllersSource.RIGHT),
            }
        )
    return OutputCombiner(outputs)


# PICO body-joint connectivity (parent → child) for skeleton rendering.
# Indices follow BodyJointIndex: 0=PELVIS, 1/2=LEFT/RIGHT_HIP, 3/6/9=SPINE1/2/3,
# 4/5=LEFT/RIGHT_KNEE, 7/8=LEFT/RIGHT_ANKLE, 10/11=LEFT/RIGHT_FOOT, 12=NECK,
# 13/14=LEFT/RIGHT_COLLAR, 15=HEAD, 16/17=LEFT/RIGHT_SHOULDER,
# 18/19=LEFT/RIGHT_ELBOW, 20/21=LEFT/RIGHT_WRIST, 22/23=LEFT/RIGHT_HAND — 24 total.
BODY_BONES: tuple[tuple[int, int], ...] = (
    # Trunk and spine
    (0, 1),
    (0, 2),
    (0, 3),
    (3, 6),
    (6, 9),
    (9, 12),
    (12, 15),
    # Left leg
    (1, 4),
    (4, 7),
    (7, 10),
    # Right leg
    (2, 5),
    (5, 8),
    (8, 11),
    # Left arm
    (12, 13),
    (13, 16),
    (16, 18),
    (18, 20),
    (20, 22),
    # Right arm
    (12, 14),
    (14, 17),
    (17, 19),
    (19, 21),
    (21, 23),
)


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


# ---------------------------------------------------------------------------
# Shared viser visualization classes
# ---------------------------------------------------------------------------


def _bone_segments(positions: np.ndarray) -> np.ndarray:
    """Return (N, 2, 3) segment array for the parent→child hand bones."""
    return np.stack(
        [np.stack([positions[a], positions[b]], axis=0) for a, b in HAND_BONES],
        axis=0,
    ).astype(np.float32)


def _valid_bone_segments(
    positions: np.ndarray, valid: np.ndarray, bones=BODY_BONES
) -> np.ndarray:
    """Return (N, 2, 3) segments whose endpoints are both available."""
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


class SkeletonViz:
    """Viser joint cloud and bones with a schema-specific layout and validity mask."""

    def __init__(
        self,
        server,
        ground=None,
        *,
        name="full_body",
        joint_count=24,
        bones=BODY_BONES,
        color=TRACKED_COLOR,
        hide_invalid=False,
    ):
        self._ground = ground
        self._joint_count = joint_count
        self._bones = bones
        self._hide_invalid = hide_invalid
        self.color = np.array(color, dtype=np.float32)
        zero_pts = np.zeros((0 if hide_invalid else joint_count, 3), dtype=np.float32)
        zero_segs = np.zeros((0, 2, 3), dtype=np.float32)

        self.points = server.scene.add_point_cloud(
            name=f"/{name}/joints",
            points=zero_pts,
            colors=np.tile(self.color, (len(zero_pts), 1)),
            point_size=0.01,
        )
        self.bones = server.scene.add_line_segments(
            name=f"/{name}/bones",
            points=zero_segs,
            colors=np.zeros((0, 2, 3), dtype=np.float32),
            line_width=2.0,
        )

    def update(self, positions: np.ndarray | None, valid: np.ndarray | None) -> int:
        if positions is None or valid is None:
            zero_pts = np.zeros(
                (0 if self._hide_invalid else self._joint_count, 3), dtype=np.float32
            )
            self.points.points = zero_pts
            self.points.colors = np.tile(INVALID_COLOR, (len(zero_pts), 1))
            self.bones.points = np.zeros((0, 2, 3), dtype=np.float32)
            self.bones.colors = np.zeros((0, 2, 3), dtype=np.float32)
            return 0

        positions = positions.astype(np.float32)
        valid_bool = valid.astype(bool)
        self.points.points = positions[valid_bool] if self._hide_invalid else positions

        point_colors = np.tile(self.color, (positions.shape[0], 1))
        point_colors[~valid_bool] = INVALID_COLOR
        self.points.colors = (
            point_colors[valid_bool] if self._hide_invalid else point_colors
        )

        segs = _valid_bone_segments(positions, valid_bool, self._bones)
        self.bones.points = segs
        self.bones.colors = np.tile(self.color, (segs.shape[0], 2, 1))

        if self._ground is not None:
            self._ground.follow(positions, valid_bool)
        return int(np.count_nonzero(valid_bool))


class FullBodyViz(SkeletonViz):
    """Render FullBodyPose's 24 joints, coloring unavailable joints red."""


class JointInputViz(SkeletonViz):
    """Render an optional evaluated joint input, hiding unavailable joints."""

    def __init__(
        self,
        server,
        name,
        joint_count,
        bones,
        positions_index,
        valid_index,
        color,
        ground=None,
    ):
        super().__init__(
            server,
            ground,
            name=name,
            joint_count=joint_count,
            bones=bones,
            color=color,
            hide_invalid=True,
        )
        self._positions_index = positions_index
        self._valid_index = valid_index

    def update(self, data):
        if data.is_none:
            return super().update(None, None)
        return super().update(
            np.asarray(data[self._positions_index], dtype=np.float32),
            np.asarray(data[self._valid_index], dtype=bool),
        )

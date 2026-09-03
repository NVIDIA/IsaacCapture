# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sharpa Avatar sample: public TeleopSession APIs + viser hand view + pinch haptic.

Single entry point. Starts CloudXR and ``avatar_hand_plugin`` unless you opt out.
Does not import plugin C++ classes or the Sharpa desktop application.

Usage (from the Isaac Teleop root, after ``src/plugins/sharpa_avatar/install.sh``)::

    python examples/sharpa_avatar/python/sharpa_avatar_sample.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PLUGIN_NAME = "avatar_hand_plugin"
PLUGIN_ROOT_ID = "sharpa_avatar"
AVATAR_GLOVE_HAPTIC_COLLECTION_ID = "avatar_glove_haptic"
AVATAR_RAW_LEFT_COLLECTION_ID = "avatar_raw_left"
AVATAR_RAW_RIGHT_COLLECTION_ID = "avatar_raw_right"
AVATAR_ROBOT_LEFT_COLLECTION_ID = "avatar_robot_left"
AVATAR_ROBOT_RIGHT_COLLECTION_ID = "avatar_robot_right"
NUM_AVATAR_JOINTS = 22
AVATAR_JOINT_NAMES = [f"joint_{i}" for i in range(NUM_AVATAR_JOINTS)]
DEFAULT_DATASETS = "human,raw,robot,haptic"
APP_NAME = "SharpaAvatarSample"
FPS = 30.0
DEFAULT_PRINT_HZ = 1.0

# Tighter than the generic optical-hand pinch demo: Avatar HUMAN tips sit
# within a few cm even when the hand is open.
MAX_DISTANCE_M = 0.028
MIN_DISTANCE_M = 0.008
PINCH_DEADBAND = 0.35

WRIST = 1
THUMB = (2, 3, 4, 5)
INDEX = (6, 7, 8, 9, 10)
MIDDLE = (11, 12, 13, 14, 15)
RING = (16, 17, 18, 19, 20)
LITTLE = (21, 22, 23, 24, 25)


def _finger_chain(root: int, joints: tuple[int, ...]) -> list[tuple[int, int]]:
    chain = [(root, joints[0])]
    for a, b in zip(joints[:-1], joints[1:]):
        chain.append((a, b))
    return chain


HAND_BONES = (
    _finger_chain(WRIST, THUMB)
    + _finger_chain(WRIST, INDEX)
    + _finger_chain(WRIST, MIDDLE)
    + _finger_chain(WRIST, RING)
    + _finger_chain(WRIST, LITTLE)
)
HAND_COLOURS = {
    "left": (0.30, 0.85, 1.00),
    "right": (1.00, 0.65, 0.30),
}
NUM_HAND_JOINTS = 26
# Display-only: no headset, both gloves share the OpenXR origin. Split
# left/right and stand fingers up (Avatar local +X is not OpenXR -Z).
# Coordinates are MuJoCo Z-up before the viser Y-up permute.
_HAND_DISPLAY_ORIGIN = {
    "left": np.array([-0.20, 0.0, 0.28], dtype=np.float64),
    "right": np.array([0.20, 0.0, 0.28], dtype=np.float64),
}


def _die(message: str, code: int = 2) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(code)


def _import_teleop():
    try:
        from isaacteleop.cloudxr import CloudXRLauncher
        from isaacteleop.retargeting_engine.deviceio_source_nodes import (
            HandsSource,
            JointStateSource,
        )
        from isaacteleop.retargeting_engine.interface import OutputCombiner
        from isaacteleop.retargeting_engine.tensor_types import (
            HandInputIndex,
            HandJointIndex,
        )
        from isaacteleop.teleop_session_manager import (
            PluginConfig,
            TeleopSession,
            TeleopSessionConfig,
        )
    except ImportError as exc:
        _die(
            "Isaac Teleop public APIs are missing "
            f"({exc}). Activate the Isaac Teleop venv and install the wheel:\n"
            "  pip install 'isaacteleop[retargeters,cloudxr]' "
            "--find-links=./install/wheels/ --no-index --force-reinstall"
        )
    return (
        CloudXRLauncher,
        HandsSource,
        JointStateSource,
        OutputCombiner,
        HandInputIndex,
        HandJointIndex,
        PluginConfig,
        TeleopSession,
        TeleopSessionConfig,
    )


def _import_viser():
    try:
        import viser
    except ImportError as exc:
        _die(
            "viser is required for the hand-shape view "
            f"({exc}). This Isaac Teleop venv has no pip; install with:\n"
            "  uv pip install viser\n"
            "  .venv/bin/python examples/sharpa_avatar/python/sharpa_avatar_sample.py\n"
            "Or pass --no-viz for terminal + haptic only."
        )
    return viser


def plugin_search_paths() -> list[Path]:
    """Directories that contain ``sharpa_avatar/avatar_hand_plugin``.

    ``src/plugins`` is not used: it has ``plugin.yaml`` but no binary, and the
    plugin manager last-write-wins, so it would exec ``./avatar_hand_plugin``
    from the source tree and fail immediately.
    """
    here = Path(__file__).resolve()
    roots = [Path.cwd(), *here.parents[:4]]
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            (
                root / "plugins",
                root / "install" / "plugins",
                root / "build" / "src" / "plugins",
            )
        )
    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        binary = path / PLUGIN_ROOT_ID / PLUGIN_NAME
        if not binary.is_file():
            continue
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def _plugin_installed(search_paths: list[Path]) -> bool:
    return any((path / PLUGIN_ROOT_ID / PLUGIN_NAME).is_file() for path in search_paths)


def _xr_pos_to_mj(p: np.ndarray) -> np.ndarray:
    # OpenXR Y-up, -Z forward → MuJoCo Z-up (Rx+90): (x, y, z) -> (x, -z, y).
    return np.array([p[0], -p[2], p[1]], dtype=np.float64)


def _anchor_joint(positions: list[np.ndarray | None]) -> np.ndarray | None:
    for idx in (WRIST, 0, MIDDLE[0], INDEX[0]):
        if positions[idx] is not None:
            return positions[idx]
    for pos in positions:
        if pos is not None:
            return pos
    return None


def _upright_rotation(centered: list[np.ndarray | None]) -> np.ndarray:
    """Map wrist→middle onto +Z so the hand stands up in any incoming frame."""
    tip = None
    for idx in (MIDDLE[-1], MIDDLE[0], INDEX[-1], INDEX[0]):
        if centered[idx] is not None:
            tip = centered[idx]
            break
    if tip is None:
        return np.eye(3)
    z_axis = np.asarray(tip, dtype=np.float64)
    norm_z = np.linalg.norm(z_axis)
    if norm_z < 1e-4:
        return np.eye(3)
    z_axis = z_axis / norm_z
    x_ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    index_j = centered[INDEX[0]]
    little_j = centered[LITTLE[0]]
    if index_j is not None and little_j is not None:
        x_ref = np.asarray(index_j - little_j, dtype=np.float64)
    x_axis = x_ref - float(np.dot(x_ref, z_axis)) * z_axis
    norm_x = np.linalg.norm(x_axis)
    if norm_x < 1e-4:
        x_axis = np.cross(np.array([0.0, 1.0, 0.0]), z_axis)
        norm_x = np.linalg.norm(x_axis)
        if norm_x < 1e-4:
            x_axis = np.cross(np.array([1.0, 0.0, 0.0]), z_axis)
            norm_x = np.linalg.norm(x_axis)
    x_axis = x_axis / norm_x
    y_axis = np.cross(z_axis, x_axis)
    return np.stack((x_axis, y_axis, z_axis), axis=0)


def _place_hand(
    positions: list[np.ndarray | None], *, side: str, upright: bool
) -> list[np.ndarray | None]:
    origin = _HAND_DISPLAY_ORIGIN[side]
    if not upright:
        return positions
    anchor = _anchor_joint(positions)
    if anchor is None:
        return [None if p is None else (p + origin) for p in positions]
    centered = [None if p is None else (p - anchor) for p in positions]
    rotation = _upright_rotation(centered)
    if side == "left":
        # Yaw 180 about MuJoCo +Z so the left palm faces the camera.
        rotation = np.diag([-1.0, -1.0, 1.0]) @ rotation
    return [None if p is None else (rotation @ p + origin) for p in centered]


def _mj_to_viser(p: np.ndarray) -> np.ndarray:
    # Display placement is Z-up; viser live_hand uses OpenXR Y-up.
    return np.array([p[0], p[2], -p[1]], dtype=np.float32)


def _bone_segments(positions: np.ndarray, valid: np.ndarray) -> np.ndarray:
    segs = [
        (positions[a], positions[b])
        for a, b in HAND_BONES
        if bool(valid[a]) and bool(valid[b])
    ]
    if not segs:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.asarray(segs, dtype=np.float32)


class _HandViz:
    """Joint cloud + bone segments, same idea as examples/mcap_record_replay live_hand."""

    def __init__(self, server, name: str, color: tuple[float, float, float]):
        self._color = np.array(color, dtype=np.float32)
        zero_pts = np.zeros((0, 3), dtype=np.float32)
        self.points = server.scene.add_point_cloud(
            name=f"/{name}/joints",
            points=zero_pts,
            colors=np.zeros((0, 3), dtype=np.float32),
            point_size=0.008,
        )
        self.bones = server.scene.add_line_segments(
            name=f"/{name}/bones",
            points=np.zeros((0, 2, 3), dtype=np.float32),
            colors=np.zeros((0, 2, 3), dtype=np.float32),
            line_width=2.0,
        )

    def update(self, positions: np.ndarray, valid: np.ndarray) -> int:
        mask = np.asarray(valid, dtype=bool)
        n = int(mask.sum())
        if n == 0:
            self.points.points = np.zeros((0, 3), dtype=np.float32)
            self.points.colors = np.zeros((0, 3), dtype=np.float32)
            self.bones.points = np.zeros((0, 2, 3), dtype=np.float32)
            self.bones.colors = np.zeros((0, 2, 3), dtype=np.float32)
            return 0
        pts = positions[mask].astype(np.float32)
        self.points.points = pts
        self.points.colors = np.tile(self._color, (n, 1))
        segs = _bone_segments(positions, mask)
        self.bones.points = segs
        self.bones.colors = np.tile(self._color, (int(segs.shape[0]), 2, 1))
        return n


def _hand_points(group, hand_input_index, *, side: str, upright: bool):
    pts = np.zeros((NUM_HAND_JOINTS, 3), dtype=np.float32)
    valid_mask = np.zeros(NUM_HAND_JOINTS, dtype=bool)
    if group is None or group.is_none:
        return pts, valid_mask, 0
    joint_positions = np.asarray(group[hand_input_index.JOINT_POSITIONS])
    joint_valid = np.asarray(group[hand_input_index.JOINT_VALID])
    positions: list[np.ndarray | None] = [None] * NUM_HAND_JOINTS
    n = 0
    for i in range(min(NUM_HAND_JOINTS, joint_positions.shape[0])):
        if not bool(joint_valid[i]):
            continue
        n += 1
        positions[i] = _xr_pos_to_mj(joint_positions[i])
    positions = _place_hand(positions, side=side, upright=upright)
    for i, pos in enumerate(positions):
        if pos is None:
            continue
        valid_mask[i] = True
        pts[i] = _mj_to_viser(pos)
    return pts, valid_mask, n


def _fmt_optional(group, *, kind: str, hand_input_index) -> str:
    if group is None or group.is_none:
        return "offline"
    try:
        if kind == "hand":
            valid = int(np.asarray(group[hand_input_index.JOINT_VALID]).sum())
            return f"ok joints={valid}"
        values = np.array(
            [float(np.asarray(group[i]).ravel()[0]) for i in range(len(group))],
            dtype=np.float64,
        )
        preview = " ".join(f"{v:+.3f}" for v in values[:4])
        return f"ok n={values.size} [{preview} ...]"
    except Exception as exc:  # noqa: BLE001 — diagnostic print only
        return f"error:{type(exc).__name__}"


def _build_haptic_pipeline(hands, mapping, sinks, hands_source_cls) -> None:
    try:
        from isaacteleop.haptic_devices.glove import haptic_glove_device
        from isaacteleop.retargeters.tactile_retargeters import (
            TactileVectorToFingerPower,
        )
        from isaacteleop.retargeting_engine.deviceio_source_nodes import HapticSink
        from isaacteleop.retargeting_engine.interface import BaseRetargeter
        from isaacteleop.retargeting_engine.interface.retargeter_core_types import (
            ComputeContext,
            RetargeterIO,
            RetargeterIOType,
        )
        from isaacteleop.retargeting_engine.interface.tensor_group_type import (
            OptionalType,
        )
        from isaacteleop.retargeting_engine.tensor_types import (
            FingerIndex,
            HandInput,
            HandInputIndex,
            HandJointIndex,
            NUM_HAPTIC_FINGERS,
            TactileVector,
        )
    except ImportError as exc:
        _die(
            "Pinch haptic needs isaacteleop haptic / retargeter extras "
            f"({exc}). Install 'isaacteleop[retargeters]' or pass --no-haptic."
        )

    try:
        device = haptic_glove_device(AVATAR_GLOVE_HAPTIC_COLLECTION_ID)
    except Exception as exc:  # noqa: BLE001
        _die(
            "Could not create the public haptic glove device "
            f"({type(exc).__name__}: {exc}). Upgrade isaacteleop or pass --no-haptic."
        )

    finger_tip_joints = {
        FingerIndex.INDEX: HandJointIndex.INDEX_TIP,
        FingerIndex.MIDDLE: HandJointIndex.MIDDLE_TIP,
        FingerIndex.RING: HandJointIndex.RING_TIP,
        FingerIndex.PINKY: HandJointIndex.LITTLE_TIP,
    }

    class PinchProximityToTactile(BaseRetargeter):
        INPUT_HAND = "hand"
        OUTPUT_TACTILE = "tactile"

        def input_spec(self) -> RetargeterIOType:
            return {self.INPUT_HAND: OptionalType(HandInput())}

        def output_spec(self) -> RetargeterIOType:
            return {self.OUTPUT_TACTILE: TactileVector(NUM_HAPTIC_FINGERS)}

        def _compute_fn(
            self, inputs: RetargeterIO, outputs: RetargeterIO, context: ComputeContext
        ) -> None:
            proximity = np.zeros(NUM_HAPTIC_FINGERS, dtype=np.float32)
            hand = inputs[self.INPUT_HAND]
            if not hand.is_none:
                joint_positions = np.asarray(hand[HandInputIndex.JOINT_POSITIONS])
                joint_valid = np.asarray(hand[HandInputIndex.JOINT_VALID])
                if bool(joint_valid[HandJointIndex.THUMB_TIP]):
                    thumb_tip = joint_positions[HandJointIndex.THUMB_TIP]
                    span = MAX_DISTANCE_M - MIN_DISTANCE_M
                    for finger, tip_joint in finger_tip_joints.items():
                        if bool(joint_valid[tip_joint]):
                            distance = float(
                                np.linalg.norm(joint_positions[tip_joint] - thumb_tip)
                            )
                            proximity[finger] = float(
                                np.clip((MAX_DISTANCE_M - distance) / span, 0.0, 1.0)
                            )
            outputs[self.OUTPUT_TACTILE][0] = proximity

    sink = HapticSink("avatar_haptic_sink", device)
    sink_inputs = {}
    for side, hand_out in (
        ("left", hands.output(hands_source_cls.LEFT)),
        ("right", hands.output(hands_source_cls.RIGHT)),
    ):
        proximity = PinchProximityToTactile(f"{side}_pinch").connect(
            {PinchProximityToTactile.INPUT_HAND: hand_out}
        )
        powers = TactileVectorToFingerPower(
            f"{side}_powers",
            num_taxels=NUM_HAPTIC_FINGERS,
            num_fingers=NUM_HAPTIC_FINGERS,
            deadband=PINCH_DEADBAND,
        ).connect(
            {
                TactileVectorToFingerPower.INPUT_TACTILE: proximity.output(
                    PinchProximityToTactile.OUTPUT_TACTILE
                )
            }
        )
        sink_inputs[side] = powers.output(TactileVectorToFingerPower.OUTPUT_POWERS)
        mapping[f"haptic_{side}"] = sink_inputs[side]
    sinks.append(sink.connect(sink_inputs))


def main() -> int:
    (
        CloudXRLauncher,
        HandsSource,
        JointStateSource,
        OutputCombiner,
        HandInputIndex,
        _HandJointIndex,
        PluginConfig,
        TeleopSession,
        TeleopSessionConfig,
    ) = _import_teleop()

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    CloudXRLauncher.add_launcher_arguments(parser)
    parser.add_argument("--hz", type=float, default=FPS, help="Session step rate.")
    parser.add_argument(
        "--print-hz",
        type=float,
        default=DEFAULT_PRINT_HZ,
        help="Terminal status rate. 0 silences prints.",
    )
    parser.add_argument(
        "--haptic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pinch-to-vibrate via HapticSink (default: on).",
    )
    parser.add_argument(
        "--launch-plugin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start avatar_hand_plugin via TeleopSession (default: on).",
    )
    parser.add_argument(
        "--transport",
        default="",
        help="wired or wireless (default: plugin USB dongle config).",
    )
    parser.add_argument(
        "--plugin-search-path",
        type=Path,
        action="append",
        default=None,
        help="Directory containing the sharpa_avatar plugin folder (repeatable).",
    )
    parser.add_argument(
        "--sdk-config",
        default="",
        help="Optional sdk_config.json passed to the plugin.",
    )
    parser.add_argument(
        "--datasets",
        default=DEFAULT_DATASETS,
        help="Passed as --datasets= when launching the plugin.",
    )
    parser.add_argument(
        "--world-frame",
        action="store_true",
        help="Draw OpenXR world poses. Default stands the hand upright and "
        "splits left/right (no headset, both wrists share the origin).",
    )
    parser.add_argument(
        "--viz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Browser viser hand view (default: on). --no-viz is terminal + haptic only.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Viser HTTP bind address (default: 127.0.0.1).",
    )
    parser.add_argument("--port", type=int, default=8080, help="Viser HTTP port.")
    args = parser.parse_args()

    search_paths = args.plugin_search_path or plugin_search_paths()
    if args.launch_plugin and not _plugin_installed(search_paths):
        _die(
            "Sharpa Avatar plugin binary was not found under "
            f"{PLUGIN_ROOT_ID}/{PLUGIN_NAME}.\n"
            "  Build it with:  ./src/plugins/sharpa_avatar/install.sh\n"
            "  That installs install/plugins/sharpa_avatar/avatar_hand_plugin "
            "(do not point at src/plugins — that tree has plugin.yaml only).\n"
            "  Or pass --plugin-search-path to the directory that contains "
            "sharpa_avatar/avatar_hand_plugin."
        )

    hands = HandsSource(name="hands")
    joint_sources = [
        JointStateSource(
            name=name,
            collection_id=collection_id,
            joint_names=AVATAR_JOINT_NAMES,
        )
        for name, collection_id in (
            ("avatar_raw_left", AVATAR_RAW_LEFT_COLLECTION_ID),
            ("avatar_raw_right", AVATAR_RAW_RIGHT_COLLECTION_ID),
            ("avatar_robot_left", AVATAR_ROBOT_LEFT_COLLECTION_ID),
            ("avatar_robot_right", AVATAR_ROBOT_RIGHT_COLLECTION_ID),
        )
    ]
    mapping = {
        "hand_left": hands.output(HandsSource.LEFT),
        "hand_right": hands.output(HandsSource.RIGHT),
        "raw_left": joint_sources[0].output(JointStateSource.JOINTS),
        "raw_right": joint_sources[1].output(JointStateSource.JOINTS),
        "robot_left": joint_sources[2].output(JointStateSource.JOINTS),
        "robot_right": joint_sources[3].output(JointStateSource.JOINTS),
    }
    sinks: list = []
    if args.haptic:
        _build_haptic_pipeline(hands, mapping, sinks, HandsSource)

    plugins = []
    if args.launch_plugin:
        plugin_args = [f"--datasets={args.datasets}"]
        if args.sdk_config:
            plugin_args.insert(0, args.sdk_config)
        if args.transport:
            plugin_args.append(f"--transport={args.transport}")
        plugins = [
            PluginConfig(
                plugin_name=PLUGIN_NAME,
                plugin_root_id=PLUGIN_ROOT_ID,
                search_paths=list(search_paths),
                plugin_args=plugin_args,
            )
        ]

    config = TeleopSessionConfig(
        app_name=APP_NAME,
        pipeline=OutputCombiner(mapping),
        sinks=sinks,
        plugins=plugins,
    )

    viser = None
    server = None
    viz_left = None
    viz_right = None
    if args.viz:
        viser = _import_viser()
        try:
            server = viser.ViserServer(host=args.host, port=args.port)
            server.scene.set_up_direction("+y")
            server.scene.add_grid(
                name="/grid",
                width=2.0,
                height=2.0,
                plane="xz",
                cell_size=0.1,
            )
            viz_left = _HandViz(server, "hand_left", HAND_COLOURS["left"])
            viz_right = _HandViz(server, "hand_right", HAND_COLOURS["right"])
        except Exception as exc:  # noqa: BLE001
            _die(
                "Could not start the viser server "
                f"({type(exc).__name__}: {exc}). Pass --no-viz for terminal only."
            )

    step_period = 1.0 / max(args.hz, 1.0)
    print_period = (1.0 / args.print_hz) if args.print_hz > 0 else None
    next_print = time.monotonic()
    last_offline_hint = 0.0
    upright = not args.world_frame

    print("=" * 72)
    print("Sharpa Avatar sample (TeleopSession + viser)")
    print("  hands:   /hand/left, /hand/right")
    print(
        f"  raw:     {AVATAR_RAW_LEFT_COLLECTION_ID}, {AVATAR_RAW_RIGHT_COLLECTION_ID}"
    )
    print(
        f"  robot:   {AVATAR_ROBOT_LEFT_COLLECTION_ID}, {AVATAR_ROBOT_RIGHT_COLLECTION_ID}"
    )
    print(
        f"  haptic:  {AVATAR_GLOVE_HAPTIC_COLLECTION_ID} "
        f"({'pinch ON' if args.haptic else 'off'})"
    )
    if args.world_frame:
        print("  display: OpenXR world frame (no left/right split)")
    else:
        print("  display: left x=-0.20 (palm to camera)  right x=+0.20  fingers up")
    if args.viz:
        print(f"  viser:   http://{args.host}:{args.port}  (Ctrl+C to exit)")
    else:
        print("  viser:   off (--no-viz); terminal + haptic only. Ctrl+C to exit.")
    print("=" * 72)

    try:
        with CloudXRLauncher.launch_context(args), TeleopSession(config) as session:
            while True:
                result = session.step()
                n_l = n_r = 0
                if viz_left is not None and viz_right is not None:
                    pts, mask, n_l = _hand_points(
                        result.get("hand_left"),
                        HandInputIndex,
                        side="left",
                        upright=upright,
                    )
                    viz_left.update(pts, mask)
                    pts, mask, n_r = _hand_points(
                        result.get("hand_right"),
                        HandInputIndex,
                        side="right",
                        upright=upright,
                    )
                    viz_right.update(pts, mask)
                else:
                    left = result.get("hand_left")
                    right = result.get("hand_right")
                    if left is not None and not left.is_none:
                        n_l = int(np.asarray(left[HandInputIndex.JOINT_VALID]).sum())
                    if right is not None and not right.is_none:
                        n_r = int(np.asarray(right[HandInputIndex.JOINT_VALID]).sum())

                now = time.monotonic()
                if print_period is not None and now >= next_print:
                    next_print = now + print_period
                    print(
                        f"[{session.get_elapsed_time():6.1f}s] "
                        f"HUMAN L={_fmt_optional(result.get('hand_left'), kind='hand', hand_input_index=HandInputIndex)}"
                        f"({n_l}) | "
                        f"R={_fmt_optional(result.get('hand_right'), kind='hand', hand_input_index=HandInputIndex)}"
                        f"({n_r})\n"
                        f"           RAW   L={_fmt_optional(result.get('raw_left'), kind='joints', hand_input_index=HandInputIndex)} | "
                        f"R={_fmt_optional(result.get('raw_right'), kind='joints', hand_input_index=HandInputIndex)}\n"
                        f"           ROBOT L={_fmt_optional(result.get('robot_left'), kind='joints', hand_input_index=HandInputIndex)} | "
                        f"R={_fmt_optional(result.get('robot_right'), kind='joints', hand_input_index=HandInputIndex)}"
                    )

                any_online = any(
                    g is not None and not g.is_none
                    for g in (
                        result.get("hand_left"),
                        result.get("hand_right"),
                        result.get("raw_left"),
                        result.get("raw_right"),
                        result.get("robot_left"),
                        result.get("robot_right"),
                    )
                )
                if not any_online and (now - last_offline_hint) >= 5.0:
                    last_offline_hint = now
                    print(
                        "  (waiting for glove data — plugin/CloudXR should start "
                        "with this process; gloves must be on; do not also run "
                        "avatar-backend or avatar_hand_tracker_printer)",
                        flush=True,
                    )
                time.sleep(step_period)
    except KeyboardInterrupt:
        print("\nExiting.")
        return 0
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        _die(
            f"{type(exc).__name__}: {exc}\n"
            "  If this is a plugin/SDK error: run ./src/plugins/sharpa_avatar/install.sh "
            "with AVATAR_SDK_ROOT or /opt/avatar-sdk present.\n"
            "  If CloudXR failed: python -m isaacteleop.cloudxr.service start "
            "and source ~/.cloudxr/run/cloudxr.env, or keep the sample's default launcher."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ROS parameter declaration, resolution, and validation for teleop_ros2_node.

Calling any ``_load_*`` helper has the side effect of registering its ROS
parameter on the supplied ``Node`` via ``Node.declare_parameter``. Helpers
declare each parameter at most once, read it back, validate it, optionally log
a startup message, and return the resolved value(s). The public entry point
``create_node_parameters`` orchestrates the helpers in dependency order and
assembles a frozen ``NodeParameters`` snapshot.
"""

import itertools
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from .constants import (
    HAND_RETARGETERS,
    HAND_TRACKING_PROVIDERS,
    TELEOP_MODES,
    TRACKED_HAND_RETARGETERS,
    WUJI_HAND_MODELS,
    EE_POSE_FRAMES,
    HandRetargeter,
    HandTrackingProvider,
    TeleopMode,
    EePoseFrame,
    resolve_hand_retargeter,
)
from isaaccapture.cloudxr.oob_teleop_env import TELEOP_CLIENT_ROUTE_ENV
from isaaccapture.deviceio import McapReplayConfig
from isaaccapture.retargeting_engine.deviceio_source_nodes.pedals_source import (
    DEFAULT_PEDAL_COLLECTION_ID,
)
from isaaccapture.teleop_session_manager import SessionMode
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from rclpy.parameter import Parameter
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class CloudXRParams:
    """CloudXR launcher settings, mirroring ``CloudXRLauncher.__init__`` kwargs.

    Field names match the launcher constructor so the resolved snapshot maps
    cleanly onto explicit keyword arguments at the call site.
    """

    install_dir: str
    env_config: str | None
    client_route: str
    accept_eula: bool
    setup_oob: bool
    usb_local: bool


@dataclass(frozen=True)
class NodeParameters:
    """Resolved snapshot of every ROS parameter consumed by TeleopRos2Node."""

    mode: TeleopMode
    sleep_period_s: float
    hand_retargeter: HandRetargeter
    resolved_hand_retargeter: HandRetargeter
    hand_tracking_provider: HandTrackingProvider
    use_external_hand_tracking_plugin: bool
    plugin_search_paths: tuple[Path, ...]
    wuji_hand_model: str
    config_asset_root: Path
    session_mode: SessionMode
    mcap_config: McapReplayConfig | None
    cloudxr_params: CloudXRParams
    pedal_collection_id: str
    world_frame: str
    right_wrist_frame: str
    left_wrist_frame: str
    head_frame: str
    transform_translation: list[float] | None
    transform_rotation: Rotation | None
    left_finger_joint_name_aliases: list[str] | None
    right_finger_joint_name_aliases: list[str] | None
    ee_poses_frame: EePoseFrame


def _load_cloudxr(node: Node) -> CloudXRParams:
    node.declare_parameter(
        "cloudxr_install_dir",
        "~/.cloudxr",
        ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description=(
                "CloudXR install directory used by the in-process "
                "CloudXRLauncher (runtime + WSS proxy). Defaults to ~/.cloudxr."
            ),
        ),
    )
    node.declare_parameter(
        "cloudxr_env_config",
        "",
        ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description=(
                "Optional CloudXR env file (KEY=value per line) passed to "
                "CloudXRLauncher to override default CloudXR env vars. Empty "
                "uses the built-in defaults."
            ),
        ),
    )
    node.declare_parameter(
        "cloudxr_client_route",
        os.environ.get(TELEOP_CLIENT_ROUTE_ENV, "/real/ros2"),
        ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description=(
                "WebXR client route exported as TELEOP_CLIENT_ROUTE before CloudXR "
                "launch. Defaults to the inherited environment value when set, "
                "otherwise /real/ros2. Empty suppresses the URL route."
            ),
        ),
    )
    node.declare_parameter(
        "cloudxr_accept_eula",
        False,
        ParameterDescriptor(
            type=ParameterType.PARAMETER_BOOL,
            description=(
                "Accept the NVIDIA CloudXR EULA non-interactively. Required for "
                "container/non-interactive runs; otherwise the launcher prompts "
                "on stdin the first time."
            ),
        ),
    )
    node.declare_parameter(
        "cloudxr_setup_oob",
        False,
        ParameterDescriptor(
            type=ParameterType.PARAMETER_BOOL,
            description=(
                "Enable the OOB (out-of-band) teleop control hub and USB adb "
                "automation in the WSS proxy. The hub shares the proxy TLS "
                "port. Only takes effect for live sessions, not MCAP replay."
            ),
        ),
    )
    node.declare_parameter(
        "cloudxr_usb_local",
        False,
        ParameterDescriptor(
            type=ParameterType.PARAMETER_BOOL,
            description=(
                "Route teleop traffic over the USB headset loopback via "
                "'adb reverse' (also starts coturn and serves the WebXR "
                "static files). Requires cloudxr_setup_oob. Only takes effect "
                "for live sessions, not MCAP replay."
            ),
        ),
    )

    install_dir = (
        node.get_parameter("cloudxr_install_dir")
        .get_parameter_value()
        .string_value.strip()
    ) or "~/.cloudxr"

    env_config_str = (
        node.get_parameter("cloudxr_env_config")
        .get_parameter_value()
        .string_value.strip()
    )
    env_config: str | None = None
    if env_config_str:
        env_config_path = Path(env_config_str).expanduser()
        if not env_config_path.is_file():
            raise FileNotFoundError(
                f"cloudxr_env_config file not found: {env_config_path}"
            )
        env_config = str(env_config_path)

    client_route = (
        node.get_parameter("cloudxr_client_route")
        .get_parameter_value()
        .string_value.strip()
    )
    accept_eula = (
        node.get_parameter("cloudxr_accept_eula").get_parameter_value().bool_value
    )
    setup_oob = node.get_parameter("cloudxr_setup_oob").get_parameter_value().bool_value
    usb_local = node.get_parameter("cloudxr_usb_local").get_parameter_value().bool_value
    # CloudXRLauncher allows usb_local only alongside setup_oob; reject the
    # invalid combination here so it surfaces as a clear parameter error.
    if usb_local and not setup_oob:
        raise ValueError(
            "Parameter 'cloudxr_usb_local' requires 'cloudxr_setup_oob' to be true"
        )
    return CloudXRParams(
        install_dir=install_dir,
        env_config=env_config,
        client_route=client_route,
        accept_eula=accept_eula,
        setup_oob=setup_oob,
        usb_local=usb_local,
    )


def _load_config_asset_root(node: Node) -> Path:
    node.declare_parameter(
        "config_asset_root",
        "",
        ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description=(
                "Directory containing teleop_ros2 configs/ and assets/. "
                "Leave empty to use the installed or source tree root."
            ),
        ),
    )
    config_asset_root_str = (
        node.get_parameter("config_asset_root")
        .get_parameter_value()
        .string_value.strip()
    )
    if config_asset_root_str:
        config_asset_root = Path(config_asset_root_str).expanduser().resolve()
        if not config_asset_root.is_dir():
            raise FileNotFoundError(
                f"config_asset_root directory not found: {config_asset_root}"
            )
    else:
        # Four levels up is the example root, where configs/ and assets/ sit --
        # in the source tree and in the install tree alike.
        config_asset_root = Path(__file__).resolve().parents[3]
    node.get_logger().info(f"Config/asset root: {config_asset_root}")
    return config_asset_root


def _load_ee_poses_frame(node: Node) -> EePoseFrame:
    node.declare_parameter(
        "ee_poses_frame",
        EePoseFrame.WORLD.value,
        ParameterDescriptor(
            description=(
                "Reference frame of the published end effector poses. "
                "'world' (default) publishes absolute poses in world_frame; "
                "'head' publishes poses relative to head_frame and marks both "
                "poses invalid while the head pose is unavailable."
            )
        ),
    )
    raw_frame = node.get_parameter("ee_poses_frame").get_parameter_value().string_value
    try:
        ee_poses_frame = EePoseFrame(raw_frame)
    except ValueError as exc:
        raise ValueError(
            f"Parameter 'ee_poses_frame' must be one of {EE_POSE_FRAMES}, "
            f"got {raw_frame!r}"
        ) from exc
    node.get_logger().info(f"EE poses frame: {ee_poses_frame}")
    return ee_poses_frame


def _load_finger_joint_name_aliases(node: Node, side: str) -> list[str] | None:
    # A bare [] default is inferred by rclpy as BYTE_ARRAY on Humble. Declare
    # by type first, then initialize the unset default to [] explicitly.
    param_name = f"{side}_finger_joint_names"
    param = node.declare_parameter(
        param_name,
        Parameter.Type.STRING_ARRAY,
        ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING_ARRAY,
            description=(
                f"Optional {side}-hand joint names for xr_teleop/finger_joints. "
                "Empty means the selected mode's default names."
            ),
            additional_constraints=(
                "Leave empty to use the selected mode's default joint names. "
                "In modes that publish xr_teleop/finger_joints, provide ROS "
                "JointState name aliases matching the selected retargeter output count."
            ),
        ),
    )
    if param.type_ == Parameter.Type.NOT_SET:
        node.set_parameters([Parameter(param_name, Parameter.Type.STRING_ARRAY, [])])

    names = node.get_parameter(param_name).value
    for index, joint_name in enumerate(names, start=1):
        if not joint_name.strip():
            raise ValueError(
                f"Parameter '{param_name}' entry {index} must be a non-empty string"
            )
    return names or None


def _load_frames(node: Node) -> tuple[str, str, str, str]:
    node.declare_parameter(
        "world_frame",
        "world",
        ParameterDescriptor(
            description=(
                "World frame for absolute pose messages and the head TF. Also the "
                "EE message frame and wrist TF parent when ee_poses_frame is 'world'. "
                "Defaults to 'world'."
            )
        ),
    )
    node.declare_parameter(
        "right_wrist_frame",
        "right_wrist",
        ParameterDescriptor(description="TF child frame name for the right wrist."),
    )
    node.declare_parameter(
        "left_wrist_frame",
        "left_wrist",
        ParameterDescriptor(description="TF child frame name for the left wrist."),
    )
    node.declare_parameter(
        "head_frame",
        "head",
        ParameterDescriptor(
            description=(
                "TF child frame name for the head. Also the EE message frame and "
                "wrist TF parent when ee_poses_frame is 'head'."
            )
        ),
    )

    # Every frame must be non-empty and distinct to avoid ambiguous transforms
    # or cycles, regardless of the selected EE reference frame.
    frame_names = ("world_frame", "right_wrist_frame", "left_wrist_frame", "head_frame")
    frames = {
        name: node.get_parameter(name).get_parameter_value().string_value
        for name in frame_names
    }
    for name, value in frames.items():
        if not value:
            raise ValueError(f"Parameter '{name}' must not be empty")
    for (name_a, value_a), (name_b, value_b) in itertools.combinations(
        frames.items(), 2
    ):
        if value_a == value_b:
            raise ValueError(
                f"Parameters '{name_a}' and '{name_b}' must be different, got {value_a!r}"
            )
    return tuple(frames.values())


def _load_hand_retargeter(
    node: Node, mode: TeleopMode
) -> tuple[HandRetargeter, HandRetargeter]:
    node.declare_parameter(
        "hand_retargeter",
        HandRetargeter.MODE_DEFAULT.value,
        ParameterDescriptor(
            description=(
                "Hand retargeter backend. 'mode_default' resolves to "
                "'trihand' in controller_teleop and 'dexpilot' in "
                "hand_teleop. Valid values: 'mode_default', 'trihand', "
                "'pink_ik', 'dexpilot', or 'wuji'."
            )
        ),
    )
    raw_hand_retargeter = (
        node.get_parameter("hand_retargeter").get_parameter_value().string_value
    )
    try:
        hand_retargeter = HandRetargeter(raw_hand_retargeter)
    except ValueError as exc:
        raise ValueError(
            f"Parameter 'hand_retargeter' must be one of {HAND_RETARGETERS}, "
            f"got {raw_hand_retargeter!r}"
        ) from exc
    resolved_hand_retargeter = resolve_hand_retargeter(mode, hand_retargeter)
    if mode in (TeleopMode.HAND_TELEOP, TeleopMode.CONTROLLER_TELEOP):
        node.get_logger().info(f"Hand retargeter: {resolved_hand_retargeter}")
    return hand_retargeter, resolved_hand_retargeter


def _load_hand_tracking_provider(
    node: Node,
    mode: TeleopMode,
    resolved_hand_retargeter: HandRetargeter,
    session_mode: SessionMode,
) -> tuple[HandTrackingProvider, bool, tuple[Path, ...]]:
    node.declare_parameter(
        "hand_tracking_provider",
        HandTrackingProvider.NATIVE.value,
        ParameterDescriptor(
            description=(
                "Hand-tracking data provider. 'native' uses runtime-provided "
                "OpenXR hands, starts no plugin, and has no effect when tracked "
                "hands are not consumed. 'manus'/'wuji' use glove-provided "
                "OpenXR hands."
            ),
            additional_constraints=f"Must be one of {HAND_TRACKING_PROVIDERS}.",
        ),
    )
    node.declare_parameter(
        "use_external_hand_tracking_plugin",
        False,
        ParameterDescriptor(
            description=(
                "Use a MANUS or Wuji provider plugin started outside this node. "
                "Leave false to start the selected provider plugin automatically "
                "with a live session."
            )
        ),
    )

    raw_provider = (
        node.get_parameter("hand_tracking_provider")
        .get_parameter_value()
        .string_value.strip()
    )
    try:
        provider = HandTrackingProvider(raw_provider)
    except ValueError as exc:
        raise ValueError(
            f"Parameter 'hand_tracking_provider' must be one of "
            f"{HAND_TRACKING_PROVIDERS}, got {raw_provider!r}"
        ) from exc

    use_external_plugin = (
        node.get_parameter("use_external_hand_tracking_plugin")
        .get_parameter_value()
        .bool_value
    )

    consumes_tracked_hands = mode == TeleopMode.HAND_TELEOP or (
        mode == TeleopMode.CONTROLLER_TELEOP
        and resolved_hand_retargeter in TRACKED_HAND_RETARGETERS
    )
    if provider != HandTrackingProvider.NATIVE and not consumes_tracked_hands:
        raise ValueError(
            f"hand_tracking_provider:={provider} requires a tracked-hand mode; "
            f"mode:={mode} with hand_retargeter:={resolved_hand_retargeter} "
            "does not consume OpenXR hand tracking"
        )

    start_plugin = (
        provider != HandTrackingProvider.NATIVE
        and session_mode == SessionMode.LIVE
        and not use_external_plugin
    )
    search_paths: tuple[Path, ...] = ()
    if start_plugin:
        search_paths = tuple(
            dict.fromkeys(
                Path(value).expanduser().resolve()
                for value in os.environ.get("ISAAC_TELEOP_PLUGIN_PATH", "").split(
                    os.pathsep
                )
                if value.strip()
            )
        )
        if not any(path.is_dir() for path in search_paths):
            raise FileNotFoundError(
                "No existing plugin search directory was found in "
                "ISAAC_TELEOP_PLUGIN_PATH"
            )
        node.get_logger().info(f"Managed hand-tracking plugin provider: {provider}")
    elif use_external_plugin and provider != HandTrackingProvider.NATIVE:
        node.get_logger().info(f"External hand-tracking provider: {provider}")
    return provider, use_external_plugin, search_paths


def _load_mcap_replay(
    node: Node,
) -> tuple[SessionMode, McapReplayConfig | None]:
    node.declare_parameter(
        "mcap_replay_path",
        "",
        ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description=(
                "Optional MCAP file to replay through TeleopSession instead "
                "of connecting to live OpenXR/DeviceIO inputs."
            ),
        ),
    )
    mcap_replay_path = (
        node.get_parameter("mcap_replay_path")
        .get_parameter_value()
        .string_value.strip()
    )
    if not mcap_replay_path:
        return SessionMode.LIVE, None

    replay_path = Path(mcap_replay_path).expanduser().resolve()
    if not replay_path.is_file():
        raise FileNotFoundError(f"mcap_replay_path file not found: {replay_path}")
    node.get_logger().info(f"Replaying MCAP input: {replay_path}")
    return SessionMode.REPLAY, McapReplayConfig(str(replay_path))


def _load_mode(node: Node) -> TeleopMode:
    node.declare_parameter("mode", TeleopMode.CONTROLLER_TELEOP.value)
    raw_mode = node.get_parameter("mode").get_parameter_value().string_value
    try:
        mode = TeleopMode(raw_mode)
    except ValueError as exc:
        raise ValueError(
            f"Parameter 'mode' must be one of {TELEOP_MODES}, got {raw_mode!r}"
        ) from exc
    node.get_logger().info(f"Mode: {mode}")
    return mode


def _load_pedal_collection_id(node: Node) -> str:
    node.declare_parameter(
        "pedal_collection_id",
        DEFAULT_PEDAL_COLLECTION_ID,
        ParameterDescriptor(
            description=(
                "Tensor collection ID used for hand_teleop foot pedal locomotion. "
                "Must match the pedal pusher or reader collection_id."
            )
        ),
    )
    pedal_collection_id = (
        node.get_parameter("pedal_collection_id").get_parameter_value().string_value
    )
    if not pedal_collection_id:
        raise ValueError("Parameter 'pedal_collection_id' must not be empty")
    return pedal_collection_id


def _load_rate_hz(node: Node) -> float:
    node.declare_parameter("rate_hz", 60.0)
    rate_hz = node.get_parameter("rate_hz").get_parameter_value().double_value
    if rate_hz <= 0 or not math.isfinite(rate_hz):
        raise ValueError("Parameter 'rate_hz' must be > 0")
    return rate_hz


def _load_transform_rotation(node: Node) -> Rotation | None:
    node.declare_parameter(
        "transform_rotation",
        [0.0, 0.0, 0.0, 1.0],
        ParameterDescriptor(
            description=(
                "Optional rotation [qx, qy, qz, qw] used to rotate "
                "published hand/EE pose positions into the ROS world "
                "frame and re-express their orientations in that rotated "
                "basis."
            )
        ),
    )
    transform_rot_arr = (
        node.get_parameter("transform_rotation")
        .get_parameter_value()
        .double_array_value
    )
    if not transform_rot_arr:
        return None
    if len(transform_rot_arr) != 4:
        raise ValueError(
            "Parameter 'transform_rotation' must have 4 elements if provided"
        )
    if np.allclose(transform_rot_arr, [0.0, 0.0, 0.0, 1.0]):
        return None

    transform_rot_floats = [float(x) for x in transform_rot_arr]
    q_norm = np.linalg.norm(transform_rot_floats)
    if q_norm < 1e-6:
        raise ValueError(
            "Parameter 'transform_rotation' must be a valid non-zero quaternion"
        )
    if not math.isclose(q_norm, 1.0, rel_tol=1e-3):
        node.get_logger().warn(
            f"Parameter 'transform_rotation' is not a unit quaternion (norm={q_norm}). Normalizing it."
        )
    normalized_q = np.array(transform_rot_floats) / q_norm
    return Rotation.from_quat(normalized_q)


def _load_transform_translation(node: Node) -> list[float] | None:
    node.declare_parameter(
        "transform_translation",
        [0.0, 0.0, 0.0],
        ParameterDescriptor(
            description=(
                "Optional translation [x, y, z] applied to published "
                "hand/EE pose positions after rotating them into the ROS "
                "world frame."
            )
        ),
    )
    transform_trans_arr = (
        node.get_parameter("transform_translation")
        .get_parameter_value()
        .double_array_value
    )
    if not transform_trans_arr:
        return None
    if len(transform_trans_arr) != 3:
        raise ValueError(
            "Parameter 'transform_translation' must have 3 elements if provided"
        )
    if np.allclose(transform_trans_arr, [0.0, 0.0, 0.0]):
        return None
    return [float(x) for x in transform_trans_arr]


def _load_wuji_hand_model(node: Node) -> str:
    parameter_name = "wuji_hand_model"
    node.declare_parameter(
        parameter_name,
        "wuji_hand_2",
        ParameterDescriptor(
            description=(
                "Wuji model used for left- and right-hand retargeting. "
                "Valid values: 'wuji_hand' or 'wuji_hand_2'."
            ),
            additional_constraints=f"Must be one of {WUJI_HAND_MODELS}.",
        ),
    )
    model = (
        node.get_parameter(parameter_name).get_parameter_value().string_value.strip()
    )
    if model not in WUJI_HAND_MODELS:
        raise ValueError(
            f"Parameter '{parameter_name}' must be one of {WUJI_HAND_MODELS}, "
            f"got {model!r}"
        )
    return model


def create_node_parameters(node: Node) -> NodeParameters:
    """Declare every ROS parameter on ``node``, validate, and return the snapshot."""
    rate_hz = _load_rate_hz(node)
    mode = _load_mode(node)
    hand_retargeter, resolved_hand_retargeter = _load_hand_retargeter(node, mode)
    wuji_hand_model = _load_wuji_hand_model(node)
    config_asset_root = _load_config_asset_root(node)
    session_mode, mcap_config = _load_mcap_replay(node)
    (
        hand_tracking_provider,
        use_external_hand_tracking_plugin,
        plugin_search_paths,
    ) = _load_hand_tracking_provider(
        node,
        mode,
        resolved_hand_retargeter,
        session_mode,
    )
    cloudxr_params = _load_cloudxr(node)
    pedal_collection_id = _load_pedal_collection_id(node)
    world_frame, right_wrist_frame, left_wrist_frame, head_frame = _load_frames(node)
    ee_poses_frame = _load_ee_poses_frame(node)
    transform_translation = _load_transform_translation(node)
    transform_rotation = _load_transform_rotation(node)
    left_finger_joint_name_aliases = _load_finger_joint_name_aliases(node, "left")
    right_finger_joint_name_aliases = _load_finger_joint_name_aliases(node, "right")

    return NodeParameters(
        mode=mode,
        sleep_period_s=1.0 / rate_hz,
        hand_retargeter=hand_retargeter,
        resolved_hand_retargeter=resolved_hand_retargeter,
        hand_tracking_provider=hand_tracking_provider,
        use_external_hand_tracking_plugin=use_external_hand_tracking_plugin,
        plugin_search_paths=plugin_search_paths,
        wuji_hand_model=wuji_hand_model,
        config_asset_root=config_asset_root,
        session_mode=session_mode,
        mcap_config=mcap_config,
        cloudxr_params=cloudxr_params,
        pedal_collection_id=pedal_collection_id,
        world_frame=world_frame,
        right_wrist_frame=right_wrist_frame,
        left_wrist_frame=left_wrist_frame,
        head_frame=head_frame,
        transform_translation=transform_translation,
        transform_rotation=transform_rotation,
        left_finger_joint_name_aliases=left_finger_joint_name_aliases,
        right_finger_joint_name_aliases=right_finger_joint_name_aliases,
        ee_poses_frame=ee_poses_frame,
    )

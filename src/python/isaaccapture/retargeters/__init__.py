# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Retargeting Modules.

This module contains retargeters available in Isaac Teleop.
Many of these are adapted from IsaacLab (Isaac Sim).

Available Retargeters:
    - DexHandRetargeter: Uses dex_retargeting library for accurate hand tracking
    - DexBiManualRetargeter: Bimanual version of DexHandRetargeter
    - TriHandMotionControllerRetargeter: Maps VR controller inputs to G1 TriHand joints
    - TriHandBiManualMotionControllerRetargeter: Bimanual version of TriHandMotionControllerRetargeter
    - LocomotionFixedRootCmdRetargeter: Fixed root command (standing still)
    - LocomotionRootCmdRetargeter: Locomotion from controller inputs
    - FootPedalRootCmdRetargeter: Root command from 3-axis foot pedal (horizontal/vertical + rudder)
    - GripperRetargeter: Pinch-based gripper control
    - SpaceMouseToSe3RelRetargeter: SpaceMouse translation/rotation state -> relative EE delta control
    - SpaceMouseGripperRetargeter: SpaceMouse left-button toggle -> gripper open/closed
    - SpaceMouseToSe2Retargeter: SpaceMouse translation/rotation state -> base velocity command (v_x, v_y, omega_z)
    - SO101ClutchRetargeter: Clutch-rebased absolute EE pose for the SO-101 5-DOF arm --
      re-latches BOTH home position and orientation on every engage, base-frame left-composed, no
      fixed offset
    - SO101GripperRetargeter: Proportional (analog) jaw closedness for the SO-101 gripper
    - WujiHandRetargeter: Retargeting for the Wuji hand via wuji_sdk.retargeting
    - JointStateRetargeter: Generic joint-space device (leader arm, exoskeleton) -> joint or EE action
    - EePoseRateLimiter / JointRateLimiter: Safety-harness velocity bounds for EE-pose / joint streams
    - ControllerPoseSource: A controller's grip or aim pose as an ``ee_pose``, Optional so
      tracking loss survives the node
    - SharpaHandRetargeter: Pinocchio/Pink IK-based retargeting for Sharpa hand
    - SharpaBiManualRetargeter: Bimanual version of SharpaHandRetargeter
    - Se3AbsRetargeter: Absolute EE pose control
    - Se3RelRetargeter: Relative EE delta control
    - TensorReorderer: Reorders and flattens multiple inputs into a single tensor
    - Vector3FrameTransform / WorldForceAccumulator / MagnitudeReducer: Composable
      spatial primitives for tactile/haptic retargeting
    - TactileVectorToControllerPulse / TactileHeatmapToControllerPulse: Map tactile
      signals to controller haptic pulses
"""

import importlib as _importlib

# All retargeters are lazy-loaded so that importing this package has zero
# side-effects and does not require any optional dependencies to be installed.
# Each name is resolved on first access via __getattr__.
#
# Entries: name -> (module_path, attr, pip_extra_or_None)
_LAZY_IMPORTS: dict[str, tuple[str, str, str | None]] = {
    # .dex_hand_retargeter  (requires retargeters extra: nlopt, torch, …)
    "DexHandRetargeter": (".dex_hand_retargeter", "DexHandRetargeter", "retargeters"),
    "DexBiManualRetargeter": (
        ".dex_hand_retargeter",
        "DexBiManualRetargeter",
        "retargeters",
    ),
    "DexHandRetargeterConfig": (
        ".dex_hand_retargeter",
        "DexHandRetargeterConfig",
        "retargeters",
    ),
    # .G1.trihand_motion_controller
    "TriHandMotionControllerRetargeter": (
        ".G1.trihand_motion_controller",
        "TriHandMotionControllerRetargeter",
        None,
    ),
    "TriHandBiManualMotionControllerRetargeter": (
        ".G1.trihand_motion_controller",
        "TriHandBiManualMotionControllerRetargeter",
        None,
    ),
    "TriHandMotionControllerConfig": (
        ".G1.trihand_motion_controller",
        "TriHandMotionControllerConfig",
        None,
    ),
    # .locomotion_retargeter
    "LocomotionFixedRootCmdRetargeter": (
        ".locomotion_retargeter",
        "LocomotionFixedRootCmdRetargeter",
        None,
    ),
    "LocomotionFixedRootCmdRetargeterConfig": (
        ".locomotion_retargeter",
        "LocomotionFixedRootCmdRetargeterConfig",
        None,
    ),
    "LocomotionRootCmdRetargeter": (
        ".locomotion_retargeter",
        "LocomotionRootCmdRetargeter",
        None,
    ),
    "LocomotionRootCmdRetargeterConfig": (
        ".locomotion_retargeter",
        "LocomotionRootCmdRetargeterConfig",
        None,
    ),
    # .foot_pedal_retargeter
    "FootPedalRootCmdRetargeter": (
        ".foot_pedal_retargeter",
        "FootPedalRootCmdRetargeter",
        None,
    ),
    "FootPedalRootCmdRetargeterConfig": (
        ".foot_pedal_retargeter",
        "FootPedalRootCmdRetargeterConfig",
        None,
    ),
    # .gripper_retargeter
    "GripperRetargeter": (".gripper_retargeter", "GripperRetargeter", None),
    "GripperRetargeterConfig": (".gripper_retargeter", "GripperRetargeterConfig", None),
    # .spacemouse_se3_retargeter  (requires retargeters-lite extra: scipy)
    "SpaceMouseToSe3RelRetargeter": (
        ".spacemouse_se3_retargeter",
        "SpaceMouseToSe3RelRetargeter",
        "retargeters-lite",
    ),
    "SpaceMouseToSe3RelRetargeterConfig": (
        ".spacemouse_se3_retargeter",
        "SpaceMouseToSe3RelRetargeterConfig",
        "retargeters-lite",
    ),
    "SpaceMouseGripperRetargeter": (
        ".spacemouse_se3_retargeter",
        "SpaceMouseGripperRetargeter",
        "retargeters-lite",
    ),
    # .spacemouse_se2_retargeter
    "SpaceMouseToSe2Retargeter": (
        ".spacemouse_se2_retargeter",
        "SpaceMouseToSe2Retargeter",
        None,
    ),
    "SpaceMouseToSe2RetargeterConfig": (
        ".spacemouse_se2_retargeter",
        "SpaceMouseToSe2RetargeterConfig",
        None,
    ),
    # .SO101 (SO-101 5-DOF arm: clutch EE-pose, analog gripper)
    "SO101ClutchRetargeter": (
        ".SO101.clutch_retargeter",
        "SO101ClutchRetargeter",
        None,
    ),
    "SO101GripperRetargeter": (
        ".SO101.gripper_retargeter",
        "SO101GripperRetargeter",
        None,
    ),
    # .wuji_hand_retargeter  (requires wuji extra: wuji-sdk[retarget],
    # which transitively pulls nlopt, pinocchio, scipy)
    "WujiHandRetargeter": (
        ".wuji_hand_retargeter",
        "WujiHandRetargeter",
        "wuji",
    ),
    "WujiHandRetargeterConfig": (
        ".wuji_hand_retargeter",
        "WujiHandRetargeterConfig",
        "wuji",
    ),
    # .joint_space (generic joint-space devices: leader arms, exoskeletons, ...)
    "JointStateRetargeter": (
        ".joint_space.joint_state_retargeter",
        "JointStateRetargeter",
        None,
    ),
    "JointStateRetargeterConfig": (
        ".joint_space.joint_state_retargeter",
        "JointStateRetargeterConfig",
        None,
    ),
    # .ee_pose_jump_guard
    "EE_POSE_JUMP_GUARD_STATUS_KEY": (
        ".ee_pose_jump_guard",
        "EE_POSE_JUMP_GUARD_STATUS_KEY",
        None,
    ),
    "EePoseJumpGuard": (".ee_pose_jump_guard", "EePoseJumpGuard", None),
    "EePoseJumpGuardConfig": (
        ".ee_pose_jump_guard",
        "EePoseJumpGuardConfig",
        None,
    ),
    "EePoseJumpGuardDisposition": (
        ".ee_pose_jump_guard",
        "EePoseJumpGuardDisposition",
        None,
    ),
    "EePoseJumpGuardState": (
        ".ee_pose_jump_guard",
        "EePoseJumpGuardState",
        None,
    ),
    "EePoseJumpGuardStatusIndex": (
        ".ee_pose_jump_guard",
        "EePoseJumpGuardStatusIndex",
        None,
    ),
    # .rate_limiter (safety harness: per-frame velocity bounds for EE / joint streams)
    # .controller_pose
    "ControllerPoseSource": (".controller_pose", "ControllerPoseSource", None),
    "HandPose": (".controller_pose", "HandPose", None),
    "EePoseRateLimiter": (".rate_limiter", "EePoseRateLimiter", None),
    "EePoseRateLimiterDisposition": (
        ".rate_limiter",
        "EePoseRateLimiterDisposition",
        None,
    ),
    "EePoseRateLimiterStatusIndex": (
        ".rate_limiter",
        "EePoseRateLimiterStatusIndex",
        None,
    ),
    "EE_POSE_STATUS_KEY": (".rate_limiter", "EE_POSE_STATUS_KEY", None),
    "JointRateLimiter": (".rate_limiter", "JointRateLimiter", None),
    "RateLimiterConfig": (".rate_limiter", "RateLimiterConfig", None),
    # .se3_retargeter  (requires retargeters-lite extra: scipy)
    "Se3AbsRetargeter": (".se3_retargeter", "Se3AbsRetargeter", "retargeters-lite"),
    "Se3RelRetargeter": (".se3_retargeter", "Se3RelRetargeter", "retargeters-lite"),
    "Se3RetargeterConfig": (
        ".se3_retargeter",
        "Se3RetargeterConfig",
        "retargeters-lite",
    ),
    # .sharpa_hand_retargeter (bundled public V2D source; grounding installs
    # Pinocchio, Pink, and the other runtime dependencies)
    "SharpaHandRetargeter": (
        ".sharpa_hand_retargeter",
        "SharpaHandRetargeter",
        "grounding",
    ),
    "SharpaBiManualRetargeter": (
        ".sharpa_hand_retargeter",
        "SharpaBiManualRetargeter",
        "grounding",
    ),
    "SharpaHandRetargeterConfig": (
        ".sharpa_hand_retargeter",
        "SharpaHandRetargeterConfig",
        "grounding",
    ),
    # .tensor_reorderer
    "TensorReorderer": (".tensor_reorderer", "TensorReorderer", None),
    # .tactile_retargeters  (tactile / haptic device-output mappers)
    "Vector3FrameTransform": (
        ".tactile_retargeters",
        "Vector3FrameTransform",
        None,
    ),
    "WorldForceAccumulator": (
        ".tactile_retargeters",
        "WorldForceAccumulator",
        None,
    ),
    "MagnitudeReducer": (".tactile_retargeters", "MagnitudeReducer", None),
    "TactileVectorToControllerPulse": (
        ".tactile_retargeters",
        "TactileVectorToControllerPulse",
        None,
    ),
    "TactileHeatmapToControllerPulse": (
        ".tactile_retargeters",
        "TactileHeatmapToControllerPulse",
        None,
    ),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attr, extra = _LAZY_IMPORTS[name]
    try:
        mod = _importlib.import_module(module_path, __package__)
    except ModuleNotFoundError as exc:
        if extra is None:
            raise
        raise ModuleNotFoundError(
            f"{name} requires additional dependencies that are not installed.\n"
            f"Install them with:  pip install 'isaaccapture[{extra}]'"
        ) from exc

    value = getattr(mod, attr)
    globals()[name] = value
    return value


__all__ = [
    # Hand tracking retargeters (require retargeters extra)
    "DexHandRetargeter",
    "DexBiManualRetargeter",
    "DexHandRetargeterConfig",
    # Motion controller retargeters
    "TriHandMotionControllerRetargeter",
    "TriHandBiManualMotionControllerRetargeter",
    "TriHandMotionControllerConfig",
    "FootPedalRootCmdRetargeter",
    "FootPedalRootCmdRetargeterConfig",
    # Locomotion retargeters
    "LocomotionFixedRootCmdRetargeter",
    "LocomotionFixedRootCmdRetargeterConfig",
    "LocomotionRootCmdRetargeter",
    "LocomotionRootCmdRetargeterConfig",
    # Manipulator retargeters
    "GripperRetargeter",
    "GripperRetargeterConfig",
    "SpaceMouseToSe3RelRetargeter",
    "SpaceMouseToSe3RelRetargeterConfig",
    "SpaceMouseGripperRetargeter",
    "SpaceMouseToSe2Retargeter",
    "SpaceMouseToSe2RetargeterConfig",
    # SO-101 5-DOF arm retargeters
    "SO101ClutchRetargeter",
    "SO101GripperRetargeter",
    # Wuji hand retargeters (require wuji extra: wuji-sdk[retarget])
    "WujiHandRetargeter",
    "WujiHandRetargeterConfig",
    # Generic joint-space device retargeters (leader arms, exoskeletons, ...)
    "JointStateRetargeter",
    "JointStateRetargeterConfig",
    # EE-pose discontinuity guard
    "EE_POSE_JUMP_GUARD_STATUS_KEY",
    "EePoseJumpGuard",
    "EePoseJumpGuardConfig",
    "EePoseJumpGuardDisposition",
    "EePoseJumpGuardState",
    "EePoseJumpGuardStatusIndex",
    # Safety-harness rate limiters (per-frame velocity bounds)
    "ControllerPoseSource",
    "EePoseRateLimiter",
    "EePoseRateLimiterDisposition",
    "EePoseRateLimiterStatusIndex",
    "EE_POSE_STATUS_KEY",
    "HandPose",
    "JointRateLimiter",
    "RateLimiterConfig",
    "Se3AbsRetargeter",
    "Se3RelRetargeter",
    "Se3RetargeterConfig",
    # Sharpa hand retargeters (require grounding extra: robotic_grounding)
    "SharpaHandRetargeter",
    "SharpaBiManualRetargeter",
    "SharpaHandRetargeterConfig",
    # Utility retargeters
    "TensorReorderer",
    # Tactile / haptic device-output retargeters
    "Vector3FrameTransform",
    "WorldForceAccumulator",
    "MagnitudeReducer",
    "TactileVectorToControllerPulse",
    "TactileHeatmapToControllerPulse",
]

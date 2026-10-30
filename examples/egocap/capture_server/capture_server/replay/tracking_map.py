"""Pure mapping from the recorder's on-disk pose conventions to the
IsaacCapture tensor groups the production adapter reads.

Recorded conventions (from the recorder's device modules, which this
package must not import):

- ``pico_controller`` head/controller poses are 4x4 matrices in the
  headset's OpenXR base space, metres, built from ``[x, y, z, qx, qy, qz,
  qw]``. An untracked pose (all-zero quaternion) is written as the
  identity matrix, so the identity is treated here as "not tracked".
- ``manus_glove`` hand poses are ``(25, 4, 4)`` skeletons in the Manus
  retargeting frame, root-relative: node 0 (the wrist) is always the
  identity, nodes 1..24 are the OpenXR joints in canonical order WITHOUT
  the palm (thumb x4, then index/middle/ring/little x5), each expressed
  relative to the wrist and rotated into a frame where open fingers point
  +Z and the open thumb +Y, with an extra Rz on the thumb tip. A node the
  recorder marked missing is all-zero (or the identity).

Inverting that hand frame change needs the wrist's world pose, which the
recording does not hold; :func:`hand_group` anchors each hand at the
pose its caller supplies (the same side's recorded controller, see
``replay_isaac_capture``). Finger articulation is therefore real while
hand placement is that documented approximation. Whatever cannot be
derived -- the palm joint, missing nodes, a hand without an anchor -- is
reported invalid, never fabricated.

The constants below mirror the recorder's mapping module at the time of
writing (alignment ``Rz(90 deg) @ Ry(180 deg)``, thumb-tip Rz of +45 deg
left / -45 deg right, neutral tilt correction 0 deg); a recorder change to
any of them must be reflected here for replayed hands to come out right.
"""

from __future__ import annotations

import math

import numpy as np

from capture_server.acquisition.tensor_indices import (
    ControllerInputIndex,
    HandInputIndex,
    HandJointIndex,
    HeadInputIndex,
)

# SDK-shaped container and nominal joint radii: faithful reproductions of the SDK's public shapes kept in
# the mock module; none of the mock's scenario or fault machinery is used here.
from capture_server.mock.mock_isaac_capture import HAND_JOINT_RADII, OptionalTensorGroup

__all__ = [
    "MANUS_NODE_COUNT",
    "OXR_TO_MANUS_ALIGN",
    "THUMB_TIP_NODE",
    "controller_group",
    "hand_group",
    "head_group",
    "is_identity_pose",
    "quaternion_xyzw",
    "thumb_tip_rotation",
    "tilt_correction",
]

MANUS_NODE_COUNT = 25
#: Manus node index of the thumb tip, the one node the recorder gives an extra in-plane rotation.
THUMB_TIP_NODE = 4
_THUMB_TIP_RZ_DEG = {"left": 45.0, "right": -45.0}
_TILT_CORRECTION_DEG = 0.0
_TILT_SIGN = {"left": 1.0, "right": -1.0}
_IDENTITY_QUATERNION = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _rot_x(radians: float) -> np.ndarray:
    """Rotation matrix about +X."""
    c, s = math.cos(radians), math.sin(radians)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(radians: float) -> np.ndarray:
    """Rotation matrix about +Y."""
    c, s = math.cos(radians), math.sin(radians)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(radians: float) -> np.ndarray:
    """Rotation matrix about +Z."""
    c, s = math.cos(radians), math.sin(radians)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


#: The recorder's OpenXR-local -> Manus-frame alignment (open fingers -Z/thumb +-X to fingers +Z/thumb +Y).
OXR_TO_MANUS_ALIGN: np.ndarray = _rot_z(math.pi / 2.0) @ _rot_y(math.pi)


def thumb_tip_rotation(side: str) -> np.ndarray:
    """The extra Rz the recorder applies to the thumb-tip node's orientation for *side* ("left"/"right")."""
    return _rot_z(math.radians(_THUMB_TIP_RZ_DEG[side]))


def tilt_correction(side: str) -> np.ndarray:
    """The recorder's neutral-tilt correction about +X for *side*, applied on the left of every node pose."""
    return _rot_x(_TILT_SIGN[side] * math.radians(_TILT_CORRECTION_DEG))


def quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Unit quaternion ``[x, y, z, w]`` (w >= 0) of a 3x3 rotation matrix, in the SDK's component order."""
    m = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(m))
    # Shepperd's method: pick the largest diagonal term for numerical stability.
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    quat = np.asarray(q, dtype=np.float64)
    quat /= np.linalg.norm(quat)
    # A canonical sign keeps consecutive samples from flipping between q and -q.
    return (-quat if quat[3] < 0.0 else quat).astype(np.float32)


def is_identity_pose(pose: np.ndarray) -> bool:
    """Whether a 4x4 pose is the identity -- the recorder's sentinel for an untracked pose."""
    return bool(np.allclose(pose, np.eye(4)))


def _pose_is_usable(pose: np.ndarray) -> bool:
    """A finite, non-identity 4x4 pose; anything else is reported as not tracked."""
    pose = np.asarray(pose)
    return pose.shape == (4, 4) and bool(np.all(np.isfinite(pose))) and not is_identity_pose(pose)


def head_group(pose: np.ndarray) -> OptionalTensorGroup:
    """HeadInput from a recorded 4x4 head pose; the identity sentinel yields a present-but-invalid head."""
    pose = np.asarray(pose, dtype=np.float64)
    valid = _pose_is_usable(pose)
    group = OptionalTensorGroup(len(HeadInputIndex))
    group[HeadInputIndex.POSITION] = pose[:3, 3].astype(np.float32)
    group[HeadInputIndex.ORIENTATION] = quaternion_xyzw(pose[:3, :3]) if valid else _IDENTITY_QUATERNION.copy()
    group[HeadInputIndex.IS_VALID] = valid
    group[HeadInputIndex.IS_TRACKED] = valid
    return group


def controller_group(pose: np.ndarray, *, trigger: float, primary: bool, secondary: bool) -> OptionalTensorGroup:
    """ControllerInput from one recorded controller row.

    The recording holds a single pose per controller, so grip and aim share it. Trigger and the two face
    buttons were recorded; squeeze, thumbstick and menu were not and are reported as 0, not guessed.
    """
    pose = np.asarray(pose, dtype=np.float64)
    valid = _pose_is_usable(pose)
    position = pose[:3, 3].astype(np.float32)
    orientation = quaternion_xyzw(pose[:3, :3]) if valid else _IDENTITY_QUATERNION.copy()
    group = OptionalTensorGroup(len(ControllerInputIndex))
    group[ControllerInputIndex.GRIP_POSITION] = position
    group[ControllerInputIndex.GRIP_ORIENTATION] = orientation
    group[ControllerInputIndex.GRIP_IS_VALID] = valid
    group[ControllerInputIndex.AIM_POSITION] = position.copy()
    group[ControllerInputIndex.AIM_ORIENTATION] = orientation.copy()
    group[ControllerInputIndex.AIM_IS_VALID] = valid
    group[ControllerInputIndex.PRIMARY_CLICK] = 1.0 if primary else 0.0
    group[ControllerInputIndex.SECONDARY_CLICK] = 1.0 if secondary else 0.0
    group[ControllerInputIndex.THUMBSTICK_X] = 0.0
    group[ControllerInputIndex.THUMBSTICK_Y] = 0.0
    group[ControllerInputIndex.THUMBSTICK_CLICK] = 0.0
    group[ControllerInputIndex.MENU_CLICK] = 0.0
    group[ControllerInputIndex.SQUEEZE_VALUE] = 0.0
    group[ControllerInputIndex.TRIGGER_VALUE] = float(trigger)
    return group


def _missing_nodes(nodes: np.ndarray) -> np.ndarray:
    """Per-node "recorder marked this missing" mask: all-zero or identity, except the wrist, which is the
    identity by construction (root-relative) and is only missing when every other node is."""
    zero = np.all(np.isclose(nodes, 0.0), axis=(1, 2))
    identity = np.all(np.isclose(nodes, np.eye(4)), axis=(1, 2))
    missing = zero | identity
    missing[0] = bool(np.all(missing[1:]))
    return missing


def _invalid_hand_group() -> OptionalTensorGroup:
    """A present hand whose every joint is invalid: the hand exists in the recording but cannot be placed."""
    group = OptionalTensorGroup(len(HandInputIndex))
    group[HandInputIndex.JOINT_POSITIONS] = np.zeros((26, 3), dtype=np.float32)
    group[HandInputIndex.JOINT_ORIENTATIONS] = np.tile(_IDENTITY_QUATERNION, (26, 1))
    group[HandInputIndex.JOINT_RADII] = np.asarray(HAND_JOINT_RADII, dtype=np.float32)
    group[HandInputIndex.JOINT_VALID] = np.zeros(26, dtype=np.uint8)
    return group


def hand_group(nodes: np.ndarray, anchor: np.ndarray | None, side: str) -> OptionalTensorGroup | None:
    """HandInput from one recorded ``(25, 4, 4)`` Manus skeleton, placed at *anchor* (a 4x4 world pose).

    Returns None when the recorder had no hand at all (every node missing), so the caller reports the group
    absent. Without a usable anchor the hand is present but every joint is invalid: the recording cannot say
    where the hand was. The palm (OpenXR joint 0) has no Manus node and is always invalid; radii are the
    nominal OpenXR values, since the recording has none.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.shape != (MANUS_NODE_COUNT, 4, 4):
        raise ValueError(f"hand skeleton must have shape ({MANUS_NODE_COUNT}, 4, 4), got {nodes.shape}")
    if side not in _THUMB_TIP_RZ_DEG:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    missing = _missing_nodes(nodes)
    if bool(np.all(missing)) or not np.all(np.isfinite(nodes)):
        return None
    if anchor is None or not _pose_is_usable(np.asarray(anchor, dtype=np.float64)):
        return _invalid_hand_group()

    anchor = np.asarray(anchor, dtype=np.float64)
    root_rot, root_pos = anchor[:3, :3], anchor[:3, 3]
    untilt = tilt_correction(side).T
    unthumb = thumb_tip_rotation(side).T
    align = OXR_TO_MANUS_ALIGN
    positions = np.zeros((26, 3), dtype=np.float32)
    orientations = np.tile(_IDENTITY_QUATERNION, (26, 1))
    valid = np.zeros(26, dtype=np.uint8)
    for node in range(MANUS_NODE_COUNT):
        if missing[node]:
            continue
        # Undo the recorder's steps in reverse: tilt correction (left-multiplied on the whole pose), the
        # thumb-tip Rz (right-multiplied on its orientation), then the frame alignment, then re-anchor.
        rot = untilt @ nodes[node, :3, :3]
        pos = untilt @ nodes[node, :3, 3]
        if node == THUMB_TIP_NODE:
            rot = rot @ unthumb
        relative_rot = align.T @ rot @ align
        relative_pos = align.T @ pos
        world_rot = root_rot @ relative_rot
        world_pos = root_pos + root_rot @ relative_pos
        # Manus node n is OpenXR joint n + 1: the recorder dropped the palm, which has no glove node.
        joint = node + 1
        positions[joint] = world_pos.astype(np.float32)
        orientations[joint] = quaternion_xyzw(world_rot)
        valid[joint] = 1
    assert valid[HandJointIndex.PALM] == 0  # never derived, see the docstring
    group = OptionalTensorGroup(len(HandInputIndex))
    group[HandInputIndex.JOINT_POSITIONS] = positions
    group[HandInputIndex.JOINT_ORIENTATIONS] = orientations
    group[HandInputIndex.JOINT_RADII] = np.asarray(HAND_JOINT_RADII, dtype=np.float32)
    group[HandInputIndex.JOINT_VALID] = valid
    return group

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorGroupType definitions for tactile input, feedback, and haptic output.

Sim-side schemas (``TactileVector``, ``TactileHeatmap``) carry contact data
into the retargeting pipeline; device-side schemas (``FingerPowerVector``,
``ControllerHapticPulse``, ``EndEffectorForce``) describe what each
``IHapticDevice`` adapter accepts. Retargeters in
:mod:`isaaccapture.retargeters.tactile_retargeters` map sim-side to
device-side; ``HapticSink`` uses ``accepted_type()`` for connect-time type
checking.
"""

from ..interface.tensor_group_type import TensorGroupType
from .ndarray_types import NDArrayType, DLDataType


# Constants
NUM_HAPTIC_FINGERS = 5
"""Channels in a :func:`FingerPowerVector`.

Standard glove convention, in order: Thumb, Index, Middle, Ring, Pinky
(see :class:`FingerIndex`).
"""

NUM_CONTROLLER_HAPTIC_FIELDS = 3
"""Fields in a :func:`ControllerHapticPulse`: ``[amplitude, frequency_hz, duration_s]``."""

NUM_END_EFFECTOR_FORCE_AXES = 3
"""Components in an :func:`EndEffectorForce`: ``[fx, fy, fz]``."""


def _float_array(name: str, rows: int, width: int) -> NDArrayType:
    return NDArrayType(name, shape=(rows, width), dtype=DLDataType.FLOAT, dtype_bits=32)


def _int_array(name: str, shape: tuple[int, ...]) -> NDArrayType:
    return NDArrayType(name, shape=shape, dtype=DLDataType.INT, dtype_bits=32)


def TactileForces(num_taxels: int) -> TensorGroupType:
    """Per-taxel force vectors: one ``(num_taxels, 3)`` float32 array."""
    return TensorGroupType(
        f"tactile_forces_{num_taxels}",
        [_float_array("taxel_forces", num_taxels, 3)],
    )


def TactileLBSLayout(num_segments: int, num_taxels: int) -> TensorGroupType:
    """LBS and HandSkeletonRef arrays, ordered by :class:`TactileLBSLayoutIndex`.

    HandJoint identifiers and segment indices are int32. Pose rows contain
    position XYZ followed by quaternion XYZW, all in hand-root space.
    """
    return TensorGroupType(
        f"tactile_lbs_layout_{num_segments}x{num_taxels}",
        [
            _int_array("segment_joint_ids", (num_segments,)),
            _float_array("segment_bind_poses", num_segments, 7),
            _float_array("taxel_bind_poses", num_taxels, 7),
            _int_array("taxel_segment_indices", (num_taxels, 4)),
            _float_array("taxel_segment_weights", num_taxels, 4),
        ],
    )


# ============================================================================
# Sim-side types
# ============================================================================


def TactileVector(num_taxels: int) -> TensorGroupType:
    """Per-taxel scalar magnitudes (or N-element vector) [N, depending on use].

    Generic sim-side schema covering single contact magnitudes
    (``num_taxels == 1``), per-pad taxel rows, and 3-vector forces /
    positions used by the composable spatial primitives. The schema fixes
    shape and dtype; the consuming retargeter fixes semantics.

    Args:
        num_taxels: Number of scalar entries.

    Returns:
        TensorGroupType with one ``(num_taxels,) float32`` tensor.
    """
    return TensorGroupType(
        f"tactile_vector_{num_taxels}",
        [
            NDArrayType(
                "tactile_values",
                shape=(num_taxels,),
                dtype=DLDataType.FLOAT,
                dtype_bits=32,
            ),
        ],
    )


def TactileHeatmap(rows: int, cols: int, num_pads: int = 1) -> TensorGroupType:
    """2D pressure grid per pad [Pa or unitless, depending on consumer].

    Sim-side schema for heatmap-style tactile sensors (e.g. Sharpa TacMap).
    Shape is ``(num_pads, rows, cols)`` so a single pad is still a 3D array
    with leading dimension 1.

    Args:
        rows: Rows per pad.
        cols: Columns per pad.
        num_pads: Number of independent pads, e.g. 5 for one pad per finger.

    Returns:
        TensorGroupType with one ``(num_pads, rows, cols) float32`` tensor.
    """
    return TensorGroupType(
        f"tactile_heatmap_{num_pads}x{rows}x{cols}",
        [
            NDArrayType(
                "tactile_pressure",
                shape=(num_pads, rows, cols),
                dtype=DLDataType.FLOAT,
                dtype_bits=32,
            ),
        ],
    )


# ============================================================================
# Device-side types
# ============================================================================


def FingerPowerVector(num_fingers: int = NUM_HAPTIC_FINGERS) -> TensorGroupType:
    """Per-finger vibration intensities [unitless, 0..1].

    Device-side schema for vibration-glove output. Standard glove order:
    ``[Thumb, Index, Middle, Ring, Pinky]`` (see :class:`FingerIndex`).

    Consumed by a :class:`~isaaccapture.haptic_devices.push_tensor.PushTensorHapticDevice`
    (see :func:`~isaaccapture.haptic_devices.glove.haptic_glove_device`), which
    re-encodes the values into a vendor-neutral ``HapticCommand`` and pushes
    them to the glove plugin process; the plugin applies them via its vendor SDK.

    Args:
        num_fingers: Number of finger channels. Defaults to 5 (standard
            five-finger glove).

    Returns:
        TensorGroupType with one ``(num_fingers,) float32`` tensor.
    """
    return TensorGroupType(
        f"finger_power_vector_{num_fingers}",
        [
            NDArrayType(
                "finger_power",
                shape=(num_fingers,),
                dtype=DLDataType.FLOAT,
                dtype_bits=32,
            ),
        ],
    )


def ControllerHapticPulse() -> TensorGroupType:
    """One-frame motion-controller vibration pulse ``[amplitude, frequency_hz, duration_s]``.

    Fields, in order (see :class:`ControllerHapticPulseField`):

    * ``amplitude`` [unitless, 0..1] -- ``0.0`` stops any active pulse.
    * ``frequency_hz`` [Hz] -- ``0.0`` selects the backend's default frequency.
    * ``duration_s`` [s] -- ``0.0`` selects the shortest pulse the backend supports.

    Consumed by :class:`isaaccapture.haptic_devices.controller.ControllerHapticDevice`;
    the live backend maps it to the controller's vibration actuator.
    """
    return TensorGroupType(
        "controller_haptic_pulse",
        [
            NDArrayType(
                "haptic_pulse",
                shape=(NUM_CONTROLLER_HAPTIC_FIELDS,),
                dtype=DLDataType.FLOAT,
                dtype_bits=32,
            ),
        ],
    )


def EndEffectorForce() -> TensorGroupType:
    """3-DoF directional force at a single point ``[fx, fy, fz]`` [N].

    Device-side schema for grounded-haptic devices like the Haply Inverse3.
    Components are in the *device* frame -- spatial retargeters upstream of
    the :class:`HapticSink` rotate sim-frame forces into device frame via the
    optional ``world_T_haptic`` ValueInput leaf and :class:`Vector3FrameTransform`.

    Shipped even though no current device consumes it, so the schema set is
    stable when the Haply force-feedback adapter lands.
    """
    return TensorGroupType(
        "end_effector_force",
        [
            NDArrayType(
                "force",
                shape=(NUM_END_EFFECTOR_FORCE_AXES,),
                dtype=DLDataType.FLOAT,
                dtype_bits=32,
            ),
        ],
    )

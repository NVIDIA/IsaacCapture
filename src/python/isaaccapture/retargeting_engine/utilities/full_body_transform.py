# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Full Body Transform Node - Corrects skeleton orientations across headset vendors.

The CloudXR client converts a Meta Quest body skeleton into the ByteDance
24-joint layout, so the server receives ``XR_BD_body_tracking`` data whichever
headset produced it. Joint order and positions survive; per-joint orientations
do not, so consumers driven by the quaternions misbehave on a Quest while a
positions-only viewer looks correct.
"""

import numpy as np

from ..interface.base_retargeter import BaseRetargeter
from ..interface.retargeter_core_types import RetargeterIO, RetargeterIOType
from ..interface.tensor_group_type import OptionalType
from ..tensor_types import FullBodyInput, FullBodyInputIndex
from ..tensor_types.standard_types import NUM_BODY_JOINTS
from .transform_utils import _copy_tensor_group_slots_from_dlpack_input

# Parent joint per index for the ByteDance layout; -1 is the root.
#
# Index 23 (RIGHT_HAND) is parented to 22, not to 21 (RIGHT_WRIST). That is
# anatomically wrong, but the correction tables below were fitted against this
# chain: do not "fix" it without refitting them.
BODY_PARENT_INDICES = (
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    22,
)

# Per-joint (x, y, z, w), fitted from paired Quest/PICO captures of the same
# poses. See the commit that introduced this file for the fit and its residuals.
_QUEST_LEFT = (
    (-0.027511403, +0.023796476, +0.003647535, +0.999331550),  #  0 PELVIS
    (+0.044720703, +0.754457857, -0.106548888, -0.646096537),  #  1 LEFT_HIP
    (+0.034870528, -0.688311460, -0.105131198, +0.716909207),  #  2 RIGHT_HIP
    (+0.297136514, -0.640326094, +0.232699328, +0.668986852),  #  3 SPINE1
    (+0.091837609, +0.753800306, +0.000469295, -0.650654080),  #  4 LEFT_KNEE
    (+0.066386953, -0.663670653, -0.001421598, +0.745071820),  #  5 RIGHT_KNEE
    (+0.231098605, +0.450775298, -0.236681421, +0.829082005),  #  6 SPINE2
    (+0.193325733, -0.292543903, -0.473174574, +0.808176372),  #  7 LEFT_ANKLE
    (-0.080666313, -0.130154103, +0.250168677, +0.956016992),  #  8 RIGHT_ANKLE
    (+0.193829968, +0.291095981, -0.063504749, +0.934697930),  #  9 SPINE3
    (-0.000777823, -0.000129299, +0.002315641, +0.999997008),  # 10 LEFT_FOOT
    (+0.000932713, -0.003214431, -0.000214648, +0.999994376),  # 11 RIGHT_FOOT
    (-0.689556031, +0.396941447, -0.409005923, -0.446837916),  # 12 NECK
    (-0.000000128, +0.000000065, -0.000000046, +1.000000000),  # 13 LEFT_COLLAR
    (+0.000000048, +0.000000009, +0.000000005, +1.000000000),  # 14 RIGHT_COLLAR
    (-0.170184319, -0.537314121, -0.001000802, +0.826032585),  # 15 HEAD
    (-0.685903511, +0.430567306, +0.286984845, -0.511652097),  # 16 LEFT_SHOULDER
    (-0.524406506, +0.313247779, -0.418056809, +0.672385418),  # 17 RIGHT_SHOULDER
    (+0.596378466, +0.201590155, +0.000568601, +0.776977355),  # 18 LEFT_ELBOW
    (-0.794222492, +0.180394750, -0.001498366, +0.580229370),  # 19 RIGHT_ELBOW
    (+0.673952121, +0.167754280, +0.061302910, +0.716860512),  # 20 LEFT_WRIST
    (-0.680723217, -0.096405995, +0.189724830, +0.700946699),  # 21 RIGHT_WRIST
    (+0.000000000, +0.000000000, +0.000000000, +1.000000000),  # 22 LEFT_HAND
    (+0.990442079, -0.068628293, -0.102554220, +0.061622049),  # 23 RIGHT_HAND
)

_QUEST_RIGHT = (
    (+0.487294213, -0.519786601, +0.477663283, +0.514007809),  #  0 PELVIS
    (+0.764142299, -0.008799327, +0.642202808, -0.059872129),  #  1 LEFT_HIP
    (-0.016948408, -0.688923455, -0.013346544, +0.724513005),  #  2 RIGHT_HIP
    (+0.247250232, -0.643653605, +0.199053185, +0.696387241),  #  3 SPINE1
    (-0.041662636, -0.792326356, +0.045966340, +0.606935141),  #  4 LEFT_KNEE
    (+0.095325615, -0.655243992, +0.029304282, +0.748805446),  #  5 RIGHT_KNEE
    (+0.199329899, +0.444560502, -0.210752738, +0.847476746),  #  6 SPINE2
    (+0.471818974, -0.141432022, +0.225134456, +0.840653505),  #  7 LEFT_ANKLE
    (+0.109134215, -0.325263745, +0.767960190, +0.540860763),  #  8 RIGHT_ANKLE
    (+0.219801081, +0.295846962, -0.076974510, +0.926410808),  #  9 SPINE3
    (-0.091276303, -0.066270899, -0.001803263, +0.993616401),  # 10 LEFT_FOOT
    (-0.092176492, -0.067892502, -0.027139606, +0.993054653),  # 11 RIGHT_FOOT
    (+0.665550260, -0.364782028, +0.461041332, +0.459801929),  # 12 NECK
    (+0.176315143, +0.620992538, -0.115300653, +0.754974832),  # 13 LEFT_COLLAR
    (+0.597692152, -0.169738088, +0.765402789, +0.167665271),  # 14 RIGHT_COLLAR
    (-0.202281237, -0.513752587, -0.157357361, +0.818766903),  # 15 HEAD
    (+0.462741695, -0.268396624, -0.071467792, +0.841858498),  # 16 LEFT_SHOULDER
    (+0.848561982, -0.096878563, +0.239914256, -0.461517341),  # 17 RIGHT_SHOULDER
    (+0.577485645, +0.245245978, +0.004380891, +0.778681930),  # 18 LEFT_ELBOW
    (+0.755959322, -0.172070047, -0.052891017, -0.629380603),  # 19 RIGHT_ELBOW
    (+0.997600287, -0.031760763, +0.059256840, -0.016539312),  # 20 LEFT_WRIST
    (+0.080226020, -0.080786116, -0.029673977, +0.993054301),  # 21 RIGHT_WRIST
    (-0.000000001, +0.000000001, -0.000000000, +1.000000000),  # 22 LEFT_HAND
    (+0.000000000, +0.000000000, +0.000000000, +1.000000000),  # 23 RIGHT_HAND
)

_IDENTITY = tuple((0.0, 0.0, 0.0, 1.0) for _ in range(NUM_BODY_JOINTS))

# Keyed by the headset that produced the skeleton. "pico" is identity: a genuine
# PICO skeleton is already correct and applying a correction would corrupt it.
SKELETON_PROFILES = {
    "pico": (_IDENTITY, _IDENTITY),
    "quest": (_QUEST_LEFT, _QUEST_RIGHT),
}


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of xyzw quaternions, broadcasting over leading axes."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def _qinv(q: np.ndarray) -> np.ndarray:
    """Inverse of a unit xyzw quaternion."""
    out = np.array(q, dtype=np.float64, copy=True)
    out[..., :3] *= -1.0
    return out


def correct_body_orientations(orientations, profile: str = "quest") -> np.ndarray:
    """
    Remap per-joint global orientations onto the ByteDance convention.

    Each joint takes a two-sided correction in parent-relative space,
    ``corrected = LEFT.inv() * local * RIGHT``. The left factor is the
    parent-frame rotation and fixes which axis a motion emerges about; the right
    factor is the child-frame relabelling and fixes rest orientation. A
    one-sided correction cannot express both, and getting it wrong leaves static
    pose error small while driving a consumer with permuted joint axes.

    Args:
        orientations: (24, 4) global orientations, xyzw, in ByteDance joint order.
        profile: Source headset, a key of ``SKELETON_PROFILES``.

    Returns:
        (24, 4) corrected global orientations, xyzw, same layout as the input.

    Raises:
        KeyError: if ``profile`` is not a known headset.
        ValueError: if ``orientations`` is not shaped (24, 4).
    """
    if profile not in SKELETON_PROFILES:
        raise KeyError(
            f"unknown skeleton profile {profile!r}; known: {sorted(SKELETON_PROFILES)}"
        )

    q = np.asarray(orientations, dtype=np.float64)
    if q.shape != (NUM_BODY_JOINTS, 4):
        raise ValueError(
            f"expected ({NUM_BODY_JOINTS}, 4) xyzw orientations, got {q.shape}"
        )

    if profile == "pico":
        return np.array(q, copy=True)

    left_tab, right_tab = SKELETON_PROFILES[profile]
    left = np.asarray(left_tab, dtype=np.float64)
    right = np.asarray(right_tab, dtype=np.float64)

    # Global -> parent-relative.
    local = np.empty_like(q)
    for j, parent in enumerate(BODY_PARENT_INDICES):
        local[j] = q[j] if parent < 0 else _qmul(_qinv(q[parent]), q[j])

    local = _qmul(_qmul(_qinv(left), local), right)

    # Parent-relative -> global. Parents precede children here, so one forward
    # pass suffices.
    out = np.empty_like(local)
    for j, parent in enumerate(BODY_PARENT_INDICES):
        out[j] = local[j] if parent < 0 else _qmul(out[parent], local[j])

    normalized: np.ndarray = out / np.linalg.norm(out, axis=-1, keepdims=True)
    return normalized


class FullBodyTransform(BaseRetargeter):
    """
    Corrects full-body joint orientations for the headset that produced them.

    Positions and validity pass through unchanged; only orientations are
    rewritten. Unlike ``HandTransform``, this applies a per-vendor skeleton
    correction rather than a 4x4 world transform, so it takes no matrix input.

    Inputs:
        - "full_body": FullBodyInput tensor (24 joints), Optional

    Outputs:
        - "full_body": FullBodyInput tensor with corrected orientations, Optional

    Example:
        body_fix = FullBodyTransform("body_fix", profile="quest")
        corrected = body_fix.connect({
            "full_body": body_source.output("full_body"),
        })
    """

    FULL_BODY = "full_body"

    def __init__(self, name: str, profile: str = "quest") -> None:
        """
        Initialize full body transform node.

        Args:
            name: Unique name for this node.
            profile: Source headset, a key of ``SKELETON_PROFILES``.

        Raises:
            KeyError: if ``profile`` is not a known headset.
        """
        if profile not in SKELETON_PROFILES:
            raise KeyError(
                f"unknown skeleton profile {profile!r}; known: {sorted(SKELETON_PROFILES)}"
            )
        super().__init__(name)
        self._profile = profile

    @property
    def profile(self) -> str:
        """Source headset this node corrects for."""
        return self._profile

    def input_spec(self) -> RetargeterIOType:
        """Declare the full body input spec (Optional)."""
        return {self.FULL_BODY: OptionalType(FullBodyInput())}

    def output_spec(self) -> RetargeterIOType:
        """Declare the corrected full body output spec (Optional)."""
        return {self.FULL_BODY: OptionalType(FullBodyInput())}

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        """
        Correct joint orientations, passing positions and validity through.

        An absent input (Optional none) is propagated to the output.

        Args:
            inputs: Dict with a "full_body" TensorGroup.
            outputs: Dict with a "full_body" TensorGroup.
            context: ComputeContext (unused by this transform node).
        """
        inp = inputs[self.FULL_BODY]
        out = outputs[self.FULL_BODY]
        if inp.is_none:
            out.set_none()
            return

        _copy_tensor_group_slots_from_dlpack_input(inp, out)
        if self._profile == "pico":
            return

        orientations = np.from_dlpack(out[FullBodyInputIndex.JOINT_ORIENTATIONS])
        orientations[:] = correct_body_orientations(orientations, self._profile).astype(
            orientations.dtype
        )

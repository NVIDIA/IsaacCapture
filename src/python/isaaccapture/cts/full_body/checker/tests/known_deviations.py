# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where this checker knowingly disagrees with the fixture index."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Deviation:
    fixture: str
    index_verdict: str
    checker_verdict: str
    advisory_check: str
    reason: str


DEVIATIONS: tuple[Deviation, ...] = (
    Deviation(
        fixture="defect_all_tracked_flag_inconsistent.mcap",
        index_verdict="fail",
        checker_verdict="pass",
        advisory_check="consistency.all_joint_poses_tracked",
        reason=(
            "live_full_body_tracker_pico_impl.cpp assigns all_joint_poses_tracked "
            "straight from locations.allJointPosesTracked while is_valid comes "
            "independently from each joint's XR_SPACE_LOCATION_*_VALID_BIT, so the two "
            "can legitimately disagree."
        ),
    ),
    Deviation(
        fixture="defect_device_clock_copies_common.mcap",
        index_verdict="fail",
        checker_verdict="pass",
        advisory_check="timestamps.device_clock_distinct",
        reason=(
            "The Pico tracker derives sample_time_raw_device_clock from "
            "xrConvertTimespecTimeToTimeKHR, so a runtime representing XrTime as "
            "CLOCK_MONOTONIC nanoseconds makes it equal the common clock on a good "
            "recording. Unverified without hardware."
        ),
    ),
    Deviation(
        fixture="posture_device_arm_raise_saturates.mcap",
        index_verdict="fail",
        checker_verdict="retake",
        advisory_check="posture.arm_raise_range_of_motion",
        reason=(
            "A short raise and a clipped one are the same measurement. The plateau "
            "that should separate them does not: 24% of samples sit within a degree "
            "of the peak on this fixture and 23% on the clean one. The peak angle "
            "is therefore the only signal, and it cannot say whether the arm went "
            "up. Asking for a retake is the honest verdict, and a device that clips "
            "fails every retake."
        ),
    ),
)

BY_FIXTURE = {d.fixture: d for d in DEVIATIONS}


# A defect fixture may genuinely violate more than one property; the index names the
# primary cause.
EXPECTED_COLLATERAL: dict[str, set[str]] = {
    # Positions scaled by 100 move at 168 m/s and describe a 157 m skeleton.
    "defect_centimetre_units.mcap": {
        "continuity.max_joint_velocity",
        "skeleton.anthropometric_plausibility",
    },
    # Six joints pinned at the origin while their parents move.
    "defect_zero_positions_on_valid_joints.mcap": {"skeleton.bone_length_constancy"},
    # Swapping SPINE1/SPINE2, NECK/HEAD and LEFT_KNEE/LEFT_ANKLE moves joints between chains.
    "defect_joint_index_permutation.mcap": {
        "skeleton.bone_length_constancy",
        "skeleton.anthropometric_plausibility",
    },
    # Mirroring the skeleton and swapping left and right labels are the same transform
    # on a symmetric body.
    "defect_mirrored_handedness.mcap": {"skeleton.left_right_labelling"},
    "defect_left_right_swapped.mcap": {"coordinate_frame.handedness"},
}

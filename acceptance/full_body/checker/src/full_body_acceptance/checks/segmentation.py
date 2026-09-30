# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whether the labels can be trusted before anything is measured through them.

Every G4 measurement reads the recording through the reviewer's windows, so a
mislabelled session produces confident numbers about the wrong frames. These checks run
first in spirit: they ask whether the windows are well formed, aligned with the frames
they claim, and describing motion that actually happened in the order claimed.

The motion tests here deliberately look only for gross presence and ordering. Anything
finer would start inferring the segmentation from the data, and the windows exist
precisely so that measurement does not have to.
"""

from __future__ import annotations

import math

from ..frames import Frame
from ..labels import StepTimeline
from ..profile import FULL_BODY, SkeletonProfile
from acceptance_common.checks import segmentation as _shared
from acceptance_common.checks.base import (
    Attribution,
    Outcome,
    Severity,
    Status,
)
from acceptance_common.checks.segmentation import LabelCheck
from acceptance_common.vectors import relative, signed_angle_about

SAGITTAL = (1.0, 0.0, 0.0)
FRONTAL = (0.0, 0.0, 1.0)

# The joint each moving step drives hardest, and the axis it turns about. A step is
# "performed" when this angle departs from its value in the still windows.
STEP_SIGNATURE: dict[str, tuple[str, tuple[float, float, float]]] = {
    "left_arm_raise": ("LEFT_SHOULDER", FRONTAL),
    "right_arm_raise": ("RIGHT_SHOULDER", FRONTAL),
    "left_leg_raise": ("LEFT_HIP", SAGITTAL),
    "right_leg_raise": ("RIGHT_HIP", SAGITTAL),
    "squat_x2": ("LEFT_HIP", SAGITTAL),
    "march_in_place": ("LEFT_HIP", SAGITTAL),
    "clap": ("LEFT_ELBOW", (0.0, 1.0, 0.0)),
}

# Peak-to-peak degrees a signature joint must sweep for its step to count as performed.
# Position noise random-walks far enough over a window to fake a path length, which is
# why this is an angle and a peak rather than an accumulated distance.
MIN_SWEEP_DEG = 15.0


class _LabelCheck(LabelCheck):
    gate = "G4"

    def __init__(
        self,
        timeline: StepTimeline | None = None,
        profile: SkeletonProfile = FULL_BODY,
    ) -> None:
        super().__init__(timeline)
        self.profile = profile

    def _angle(self, frame: Frame, joint: str, axis) -> float | None:
        index = self.profile.index(joint)
        parent = self.profile.parents[index]
        joints = frame.joints
        if joints is None or parent < 0:
            return None
        child, base = joints[index], joints[parent]
        if not (child.is_valid and base.is_valid):
            return None
        radians = signed_angle_about(
            relative(base.orientation, child.orientation), axis
        )
        return math.degrees(radians) if math.isfinite(radians) else None


class LabelWindowsWellformed(_shared.LabelWindowsWellformed):
    gate = "G4"


class LabelAlignment(_shared.LabelAlignment):
    gate = "G4"


class _MotionPresenceCheck(_LabelCheck):
    """Peak-to-peak sweep of each step's signature joint, per labelled window."""

    depends_on = (
        "segmentation.label_windows_wellformed",
        "segmentation.label_alignment",
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.extremes: dict[str, tuple[float, float]] = {}

    def _update(self, frame: Frame) -> None:
        if self.timeline is None or frame.joints is None:
            return
        step = self.timeline.step_at(frame.sample_time_ns)
        if step is None or step.label not in STEP_SIGNATURE:
            return
        joint, axis = STEP_SIGNATURE[step.label]
        angle = self._angle(frame, joint, axis)
        if angle is None:
            return
        low, high = self.extremes.get(step.label, (angle, angle))
        self.extremes[step.label] = (min(low, angle), max(high, angle))

    def _sweeps(self) -> dict[str, float]:
        return {label: high - low for label, (low, high) in self.extremes.items()}


class LabelledStepActuallyPerformed(_MotionPresenceCheck):
    name = "segmentation.labelled_step_actually_performed"
    severity = Severity.HARD
    attribution = Attribution.PERFORMANCE
    summary = "Every labelled motion step actually contains that motion"

    def _result(self) -> Outcome:
        if self.timeline is None:
            return Outcome(Status.INSUFFICIENT_DATA, "no motion labels to verify")
        sweeps = self._sweeps()
        if not sweeps:
            return Outcome(
                Status.INSUFFICIENT_DATA,
                "no labelled step had a measurable signature joint",
            )
        missing = {
            label: sweep for label, sweep in sweeps.items() if sweep < MIN_SWEEP_DEG
        }
        measurements = {
            "sweep_deg": {
                label: round(value, 2) for label, value in sorted(sweeps.items())
            },
            "missing": sorted(missing),
        }
        if not missing:
            return Outcome(
                Status.PASS,
                f"all {len(sweeps)} labelled motion steps were performed",
                measurements,
            )
        return Outcome(
            Status.FAIL,
            f"{', '.join(sorted(missing))} was labelled but never performed; the window "
            f"holds under {MIN_SWEEP_DEG:.0f} deg of its own motion",
            measurements,
        )


class StepOrderMatchesLabels(_LabelCheck):
    name = "segmentation.step_order_matches_labels"
    severity = Severity.HARD
    attribution = Attribution.PERFORMANCE
    summary = "The motion in each window is the motion its label names"

    # A window is claimed by whichever signature sweeps most in it. Two steps swapped
    # into each other's slots therefore each name a window whose motion belongs to the
    # other, which is visible without segmenting anything.
    MIN_MARGIN_DEG = 10.0

    DISTINCT = (
        "left_arm_raise",
        "right_arm_raise",
        "left_leg_raise",
        "right_leg_raise",
    )

    depends_on = (
        "segmentation.label_windows_wellformed",
        "segmentation.label_alignment",
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sweeps: dict[str, dict[str, tuple[float, float]]] = {}

    def _update(self, frame: Frame) -> None:
        if self.timeline is None or frame.joints is None:
            return
        step = self.timeline.step_at(frame.sample_time_ns)
        if step is None or step.label not in self.DISTINCT:
            return
        window = self.sweeps.setdefault(step.label, {})
        for candidate in self.DISTINCT:
            joint, axis = STEP_SIGNATURE[candidate]
            angle = self._angle(frame, joint, axis)
            if angle is None:
                continue
            low, high = window.get(candidate, (angle, angle))
            window[candidate] = (min(low, angle), max(high, angle))

    def _result(self) -> Outcome:
        if self.timeline is None:
            return Outcome(Status.INSUFFICIENT_DATA, "no motion labels to verify")
        if not self.sweeps:
            return Outcome(
                Status.INSUFFICIENT_DATA, "none of the single-limb steps was measurable"
            )
        mismatched: dict[str, str] = {}
        observed: dict[str, str] = {}
        for label, window in self.sweeps.items():
            ranked = sorted(
                ((high - low, candidate) for candidate, (low, high) in window.items()),
                reverse=True,
            )
            if not ranked:
                continue
            best_sweep, best = ranked[0]
            claimed = window.get(label)
            claimed_sweep = (claimed[1] - claimed[0]) if claimed else 0.0
            observed[label] = best
            if best != label and best_sweep - claimed_sweep > self.MIN_MARGIN_DEG:
                mismatched[label] = best
        measurements = {"dominant_motion": observed, "mismatched": mismatched}
        if not mismatched:
            return Outcome(
                Status.PASS,
                f"each of the {len(self.sweeps)} single-limb windows contains its own "
                f"motion",
                measurements,
            )
        detail = ", ".join(
            f"{label!r} holds {found!r}" for label, found in sorted(mismatched.items())
        )
        return Outcome(
            Status.FAIL,
            f"the script was performed out of order: {detail}",
            measurements,
        )


class FallbackWithoutLabels(_LabelCheck):
    name = "segmentation.fallback_without_labels"
    gate = "G4"
    severity = Severity.ADVISORY
    summary = "Says which G4 measurements a recording without labels gives up"

    # Held poses can be found without labels, since a still window is where the whole
    # skeleton stops moving. Which held pose it is cannot be, so this reports the
    # fallback rather than measuring through it.
    STILL_SPEED_DEG_PER_S = 2.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.previous: tuple[float, ...] | None = None
        self.previous_ns: int | None = None
        self.still_frames = 0
        self.moving_frames = 0

    def _update(self, frame: Frame) -> None:
        joints = frame.joints
        if joints is None or frame.sample_time_ns is None:
            return
        angles = tuple(
            self._angle(frame, name, SAGITTAL) or 0.0
            for name in ("LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE")
        )
        if self.previous is not None and self.previous_ns is not None:
            elapsed = (frame.sample_time_ns - self.previous_ns) / 1e9
            if elapsed > 0:
                rate = max(
                    abs(now - before) / elapsed
                    for now, before in zip(angles, self.previous)
                )
                if rate <= self.STILL_SPEED_DEG_PER_S:
                    self.still_frames += 1
                else:
                    self.moving_frames += 1
        self.previous, self.previous_ns = angles, frame.sample_time_ns

    def _result(self) -> Outcome:
        total = self.still_frames + self.moving_frames
        if total == 0:
            return Outcome(Status.INSUFFICIENT_DATA, "no timestamped poses to examine")
        still = self.still_frames / total
        measurements = {
            "labels_present": self.timeline is not None,
            "still_fraction": still,
        }
        if self.timeline is not None:
            return Outcome(
                Status.PASS,
                "motion labels are present, so every G4 window is measurable",
                measurements,
            )
        return Outcome(
            Status.PASS,
            f"no motion labels: {still:.0%} of the session is a held pose, but which "
            f"held pose cannot be known, so the per-step G4 measurements are skipped "
            f"rather than guessed",
            measurements,
        )

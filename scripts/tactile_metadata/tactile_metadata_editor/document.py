# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

JOINTS = """PALM WRIST THUMB_METACARPAL THUMB_PROXIMAL THUMB_DISTAL THUMB_TIP
INDEX_METACARPAL INDEX_PROXIMAL INDEX_INTERMEDIATE INDEX_DISTAL INDEX_TIP
MIDDLE_METACARPAL MIDDLE_PROXIMAL MIDDLE_INTERMEDIATE MIDDLE_DISTAL MIDDLE_TIP
RING_METACARPAL RING_PROXIMAL RING_INTERMEDIATE RING_DISTAL RING_TIP
LITTLE_METACARPAL LITTLE_PROXIMAL LITTLE_INTERMEDIATE LITTLE_DISTAL LITTLE_TIP""".split()
IDENTITY_POSE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
IDENTITY_MATRIX = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


@dataclass
class Segment:
    joint: str
    pose: list[float] = field(default_factory=lambda: IDENTITY_POSE.copy())


@dataclass
class Taxel:
    pose: list[float] = field(default_factory=lambda: IDENTITY_POSE.copy())
    segments: list[str | None] = field(default_factory=lambda: [None] * 4)
    weights: list[float] = field(default_factory=lambda: [0.0] * 4)


def pose_error(values: object) -> str | None:
    if not isinstance(values, list) or len(values) != 7:
        return "Expected XYZ and XYZW"
    if any(
        type(value) not in (int, float) or not math.isfinite(value) for value in values
    ):
        return "Pose components must be finite numbers"
    if abs(sum(value * value for value in values[3:]) - 1.0) > 1e-4:
        return "Orientation must be a unit quaternion"
    return None


class Document:
    """Editor-native data. Poses and semantic bindings never use FlatBuffer row indices."""

    def __init__(self, side: str):
        if side not in ("left", "right"):
            raise ValueError("Side must be left or right")
        self.side = side
        self.segments: list[Segment] = []
        self.taxels: list[Taxel] = []
        self.q: dict[str, float] = {}
        self.alignment = copy.deepcopy(IDENTITY_MATRIX)

    def clone(self) -> Document:
        return copy.deepcopy(self)

    def add_segment(self, joint: str, pose: list[float]) -> int:
        if joint not in JOINTS:
            raise ValueError("Choose a HandJoint")
        self.segments.append(Segment(joint, list(pose)))
        return len(self.segments) - 1

    def add_taxels(self, poses: list[list[float]]) -> list[int]:
        first = len(self.taxels)
        self.taxels.extend(Taxel(list(pose)) for pose in poses)
        return list(range(first, len(self.taxels)))

    def delete(self, kind: str, rows: list[int]) -> None:
        values = self.segments if kind == "segments" else self.taxels
        for row in sorted(set(rows), reverse=True):
            if 0 <= row < len(values):
                values.pop(row)

    def move(self, kind: str, source: int, target: int) -> None:
        values = self.segments if kind == "segments" else self.taxels
        if not 0 <= source < len(values) or not 0 <= target < len(values):
            raise ValueError("Reorder row is outside the list")
        value = values.pop(source)
        values.insert(target, value)

    def assign(self, taxels: list[int], segment_rows: list[int]) -> None:
        if not 1 <= len(segment_rows) <= 4:
            raise ValueError("Select one to four segments")
        joints = []
        for row in segment_rows:
            if not 0 <= row < len(self.segments):
                raise ValueError("Selected segment no longer exists")
            joints.append(self.segments[row].joint)
        weights = [1.0 / len(joints)] * len(joints)
        for row in taxels:
            if 0 <= row < len(self.taxels):
                self.taxels[row].segments = joints + [None] * (4 - len(joints))
                self.taxels[row].weights = weights + [0.0] * (4 - len(joints))

    def set_weight(self, taxel_row: int, slot: int, value: float) -> None:
        taxel = self.taxels[taxel_row]
        if taxel.segments[slot] is None:
            raise ValueError("Select a segment before editing its weight")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Weight must be from 0 to 1")
        active = [i for i, segment in enumerate(taxel.segments) if segment is not None]
        if len(active) == 1:
            taxel.weights = [1.0 if i == slot else 0.0 for i in range(4)]
            return
        others = [i for i in active if i != slot]
        remainder = 1.0 - value
        total = sum(taxel.weights[i] for i in others)
        taxel.weights[slot] = value
        for i in range(4):
            if taxel.segments[i] is None:
                taxel.weights[i] = 0.0
        if total > 0.0:
            for i in others:
                taxel.weights[i] = taxel.weights[i] / total * remainder
        else:
            for i in others:
                taxel.weights[i] = remainder / len(others)

    def validate(self) -> list[dict]:
        issues: list[dict] = []

        def issue(kind: str, row: int | None, field: str, message: str) -> None:
            issues.append(dict(kind=kind, row=row, field=field, message=message))

        duplicate_joints = {
            joint
            for joint in {segment.joint for segment in self.segments}
            if sum(segment.joint == joint for segment in self.segments) > 1
        }
        joints = {segment.joint for segment in self.segments}
        for row, segment in enumerate(self.segments):
            if segment.joint not in JOINTS:
                issue("segments", row, "joint", "Invalid HandJoint")
            elif segment.joint in duplicate_joints:
                issue("segments", row, "joint", "HandJoint is already used")
            if error := pose_error(segment.pose):
                issue("segments", row, "pose", error)
        for row, taxel in enumerate(self.taxels):
            if error := pose_error(taxel.pose):
                issue("taxels", row, "pose", error)
            if len(taxel.segments) != 4 or len(taxel.weights) != 4:
                issue("taxels", row, "binding", "Expected four binding slots")
                continue
            for slot, (joint, weight) in enumerate(zip(taxel.segments, taxel.weights)):
                valid_weight = type(weight) in (int, float) and math.isfinite(weight)
                if joint is None:
                    if not valid_weight or weight != 0.0:
                        issue(
                            "taxels", row, f"weight{slot}", "None weight must be zero"
                        )
                elif joint not in joints:
                    issue("taxels", row, f"segment{slot}", f"Missing segment {joint}")
                elif not valid_weight or weight <= 0.0:
                    issue(
                        "taxels",
                        row,
                        f"weight{slot}",
                        "Selected segment weight must be greater than zero",
                    )
            if not any(
                joint in joints for joint in taxel.segments if joint is not None
            ):
                issue("taxels", row, "binding", "Taxel is not bound to a segment")
            numeric = all(
                type(value) in (int, float) and math.isfinite(value)
                for value in taxel.weights
            )
            if not numeric or abs(sum(taxel.weights) - 1.0) > 1e-5:
                issue("taxels", row, "weights", "Weights must sum to one")
        return issues


class State:
    def __init__(self):
        self.documents = {side: Document(side) for side in ("left", "right")}
        self.saved: dict[str, Document | None] = {"left": None, "right": None}
        self.paths: dict[str, str] = {}
        self.side = "left"

    @property
    def document(self) -> Document:
        return self.documents[self.side]

    def replace(self, document: Document, path: str | None = None) -> None:
        self.documents[document.side] = document
        self.saved[document.side] = document.clone()
        self.saved[document.side].q = {}
        if path is not None:
            self.paths[document.side] = path

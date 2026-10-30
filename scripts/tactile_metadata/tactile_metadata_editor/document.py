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
# Neutral right-hand OpenXR poses in wrist coordinates; left-hand mirroring is
# applied before conversion into visualization space.
# fmt: off
DEFAULT_OPENXR_SEGMENT_POSES = (
    (-0.00244, 0.00122, -0.05917, 0.010296699, 0.02060433, -0.000212213, 0.999734662),
    (0, 0, 0, 0, 0, 0, 1),
    (-0.022, -0.012, -0.028, -0.00692186, 0.30239121, 0.275661445, 0.912426661),
    (-0.047208158, -0.020249943, -0.065583071, -0.00692186, 0.30239121, 0.275661445, 0.912426661),
    (-0.065292271, -0.02616838, -0.09254484, -0.00692186, 0.30239121, 0.275661445, 0.912426661),
    (-0.080088363, -0.031010737, -0.114604468, -0.00692186, 0.30239121, 0.275661445, 0.912426661),
    (-0.00528, 0, -0.02068, 0, 0.124664128, 0, 0.992199),
    (-0.024, 0, -0.094, -0.017438446, 0.039983244, 0.00069791, 0.999047924),
    (-0.027194641, -0.00139598, -0.133847779, -0.043584497, 0.039951273, 0.00174431, 0.99824909),
    (-0.029105295, -0.003487718, -0.157679985, -0.060999707, 0.039914746, 0.00244129, 0.997336397),
    (-0.030691675, -0.005925105, -0.177467419, -0.060999707, 0.039914746, 0.00244129, 0.997336397),
    (-0.00088, 0.00044, -0.02134, 0.010296699, 0.02060433, -0.000212213, 0.999734662),
    (-0.004, 0.002, -0.097, -0.017452406, 0, 0, 0.999847695),
    (-0.004, 0.000429523, -0.141972587, -0.043619387, 0, 0, 0.999048222),
    (-0.004, -0.002010838, -0.169866039, -0.06104854, 0, 0, 0.998134798),
    (-0.004, -0.004691964, -0.191702054, -0.06104854, 0, 0, 0.998134798),
    (0.0033, 0, -0.02002, 0, -0.081592271, 0, 0.996665792),
    (0.015, 0, -0.091, -0.017444553, -0.029990932, -0.000523494, 0.999397797),
    (0.017516954, -0.001465779, -0.132898883, -0.04359976, -0.029966951, -0.001308385, 0.998598684),
    (0.019070086, -0.003731828, -0.158753338, -0.06102107, -0.029939553, -0.001831181, 0.997685671),
    (0.020319944, -0.006291084, -0.1795593, -0.06102107, -0.029939553, -0.001831181, 0.997685671),
    (0.00704, -0.00088, -0.01804, -0.022312091, -0.184914996, -0.004199362, 0.982492229),
    (0.032, -0.004, -0.082, -0.017415551, -0.064944346, -0.001133608, 0.99773626),
    (0.036145766, -0.005116784, -0.113710652, -0.043527274, -0.064892417, -0.002833264, 0.996938475),
    (0.038599442, -0.006772743, -0.132478637, -0.06091962, -0.064833086, -0.003965361, 0.996026981),
    (0.040915463, -0.008966391, -0.150193714, -0.06091962, -0.064833086, -0.003965361, 0.996026981),
)
# fmt: on


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

    def assign(self, taxels: list[int], segment_rows: list[int]) -> None:
        if not 1 <= len(segment_rows) <= 4:
            raise ValueError("Select one to four segments")
        joints = []
        for row in segment_rows:
            if not 0 <= row < len(self.segments):
                raise ValueError("Selected segment no longer exists")
            joints.append(self.segments[row].joint)
        weights = [1.0] + [0.0] * (len(joints) - 1)
        for row in taxels:
            if 0 <= row < len(self.taxels):
                self.taxels[row].segments = joints + [None] * (4 - len(joints))
                self.taxels[row].weights = weights + [0.0] * (4 - len(joints))

    def set_weight(self, taxel_row: int, slot: int, value: float) -> None:
        self.set_taxel_weight(self.taxels[taxel_row], slot, value)

    @staticmethod
    def set_taxel_weight(taxel: Taxel, slot: int, value: float) -> None:
        if taxel.segments[slot] is None:
            raise ValueError("Select a segment before editing its weight")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Weight must be from 0 to 1")
        taxel.weights[slot] = value

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
                elif not valid_weight or weight < 0.0:
                    issue(
                        "taxels",
                        row,
                        f"weight{slot}",
                        "Selected segment weight must be nonnegative",
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

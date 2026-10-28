# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import json
import math
import uuid

JOINTS = """PALM WRIST THUMB_METACARPAL THUMB_PROXIMAL THUMB_DISTAL THUMB_TIP
INDEX_METACARPAL INDEX_PROXIMAL INDEX_INTERMEDIATE INDEX_DISTAL INDEX_TIP
MIDDLE_METACARPAL MIDDLE_PROXIMAL MIDDLE_INTERMEDIATE MIDDLE_DISTAL MIDDLE_TIP
RING_METACARPAL RING_PROXIMAL RING_INTERMEDIATE RING_DISTAL RING_TIP
LITTLE_METACARPAL LITTLE_PROXIMAL LITTLE_INTERMEDIATE LITTLE_DISTAL LITTLE_TIP""".split()
FIELDS = (
    "segment_bind_poses",
    "taxel_bind_poses",
    "taxel_segment_indices",
    "taxel_segment_weights",
)
GROUPS = {
    "Skeleton binding poses": ("joints", FIELDS[0]),
    "Taxel binding poses": FIELDS[1:2],
    "Skinning": FIELDS[2:],
}
IDENTITY = [0, 0, 0, 0, 0, 0, 1]
# Neutral right-hand OpenXR poses in wrist coordinates; left is mirrored below.
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


def joint_number(joint):
    number = JOINTS.index(joint) if isinstance(joint, str) else joint
    if type(number) is not int or not 0 <= number < len(JOINTS):
        raise ValueError("Invalid HandJoint identifier")
    return number


def row_error(field, values, segments):
    width = 7 if field in FIELDS[:2] else 4
    if not isinstance(values, list) or len(values) != width:
        return f"Expected {width} components"
    if any(
        type(v) not in (int, float) or (type(v) is float and not math.isfinite(v))
        for v in values
    ):
        return "Components must be finite numbers"
    if field == FIELDS[2]:
        if any(
            type(v) is not int or not 0 <= v < min(segments, 0x100000000)
            for v in values
        ):
            return "All slots must index segment rows, including zero-weight slots"
    elif field == FIELDS[3]:
        if min(values) < 0 or abs(sum(values) - 1) > 1e-5:
            return "Weights must be nonnegative and sum to one"
    elif (
        max(map(abs, values)) > 3.4028234663852886e38
        or abs(sum(v * v for v in values[3:]) - 1) > 1e-4
    ):
        return "Expected float32 coordinates and a unit XYZW quaternion"
    return None


class Document:
    def __init__(self, side=None):
        self.side = side
        self.joints = JOINTS[:] if side in ("left", "right") else []
        self.arrays = {field: [] for field in FIELDS}
        if self.joints:
            poses = [list(pose) for pose in DEFAULT_OPENXR_SEGMENT_POSES]
            if side == "left":
                for pose in poses:
                    pose[0], pose[4], pose[5] = -pose[0], -pose[4], -pose[5]
            self.arrays[FIELDS[0]] = poses
        self.types = ("LBS", "HandSkeletonRef")
        self.editor = {"segment_ids": [], "output_indices": []}
        self.sync()

    def sync(self):
        count, ids = len(self.arrays[FIELDS[0]]), self.editor["segment_ids"]
        ids.extend(uuid.uuid4().hex for _ in range(count - len(ids)))
        del ids[count:]
        count, order = len(self.arrays[FIELDS[1]]), self.editor["output_indices"]
        order.extend(range(len(order), count))
        del order[count:]

    def set_components(self, field, selected, components, values):
        width = 7 if field in FIELDS[:2] else 4
        for i in selected:
            while len(self.arrays[field]) <= i:
                self.arrays[field].append(None)
            if not isinstance(self.arrays[field][i], list):
                self.arrays[field][i] = [None] * width
            row = self.arrays[field][i]
            row.extend([None] * max(0, width - len(row)))
            for component, value in zip(components, values):
                row[component] = value
        if field in FIELDS[:2] and selected:
            self.editor["approximate"] = True
            self.sync()

    def weight(self, selected, slot, value):
        rejected = []
        for i in selected:
            try:
                values = list(self.arrays[FIELDS[3]][i])
                values[slot] = float(value)
                if (
                    len(values) != 4
                    or min(values) < 0
                    or not all(math.isfinite(v) for v in values)
                ):
                    raise ValueError("Expected four finite nonnegative weights")
                total = sum(values)
                if total <= 0 or not math.isfinite(total):
                    raise ValueError("Weight sum must be finite and positive")
                self.arrays[FIELDS[3]][i] = [v / total for v in values]
            except (IndexError, ValueError, TypeError, OverflowError) as exc:
                rejected.append(f"Taxel row {i}: {exc}")
        return rejected

    def assign(self, selected, segments):
        if not 1 <= len(segments) <= 4 or any(
            type(i) is not int or not 0 <= i < len(self.arrays[FIELDS[0]])
            for i in segments
        ):
            raise ValueError("Choose one to four existing segment rows")
        for i in selected:
            self.set_components(
                FIELDS[2], [i], range(4), segments + [segments[0]] * (4 - len(segments))
            )
            self.set_components(
                FIELDS[3],
                [i],
                range(4),
                [1 / len(segments)] * len(segments) + [0] * (4 - len(segments)),
            )

    def unused_start(self, count):
        start = 0
        while any(start + i in self.editor["output_indices"] for i in range(count)):
            start += 1
        return start

    def add(self, poses, segments, start=-1):
        if any(len(self.arrays[f]) != len(self.arrays[FIELDS[1]]) for f in FIELDS[2:]):
            raise ValueError("Repair skinning counts before adding taxels")
        if not poses:
            raise ValueError("No candidates to apply")
        start = self.unused_start(len(poses)) if start < 0 else start
        first = len(self.arrays[FIELDS[1]])
        self.arrays[FIELDS[1]].extend(copy.deepcopy(poses))
        self.assign(range(first, first + len(poses)), segments)
        self.editor["output_indices"].extend(range(start, start + len(poses)))
        self.sync()
        self.editor["approximate"] = True

    def delete(self, selected):
        for i in sorted(set(selected), reverse=True):
            for values in (
                *[self.arrays[f] for f in FIELDS[1:]],
                self.editor["output_indices"],
            ):
                if i < len(values):
                    values.pop(i)

    def segments(self, order):
        count = len(self.arrays[FIELDS[0]])
        if (
            len(self.joints) != count
            or len(set(order)) != len(order)
            or any(type(i) is not int or not 0 <= i < count for i in order)
        ):
            raise ValueError("Repair skeleton counts and supply distinct segment rows")
        mapping = {old: new for new, old in enumerate(order)}
        self.joints = [self.joints[i] for i in order]
        self.arrays[FIELDS[0]] = [self.arrays[FIELDS[0]][i] for i in order]
        self.editor["segment_ids"] = [self.editor["segment_ids"][i] for i in order]
        for values in self.arrays[FIELDS[2]]:
            if isinstance(values, list):
                values[:] = [
                    mapping.get(i, -1) if type(i) is int and 0 <= i < count else i
                    for i in values
                ]

    def record(self, received, groups):
        for group in groups:
            for field in GROUPS[group]:
                if field == "joints":
                    self.joints = copy.deepcopy(received.joints)
                else:
                    self.arrays[field] = copy.deepcopy(received.arrays[field])
        self.sync()

    def check(self):
        problems = []

        def issue(field, message, row=None, severity="error"):
            problems.append(
                dict(severity=severity, field=field, row=row, message=message)
            )

        if self.types != ("LBS", "HandSkeletonRef"):
            issue("metadata", "Expected LBS and HandSkeletonRef discriminators")
        j, n, seen = len(self.joints), len(self.arrays[FIELDS[1]]), set()
        for field, expected in zip(FIELDS, (j, n, n, n)):
            if len(self.arrays[field]) != expected:
                issue(field, f"Expected {expected} rows")
            for i, row in enumerate(self.arrays[field]):
                error = row_error(field, row, len(self.arrays[FIELDS[0]]))
                if (
                    field == FIELDS[0]
                    and i < j
                    and self.joints[i] in ("WRIST", 1)
                    and not error
                ):
                    if (
                        any(abs(v) > 1e-6 for v in row[:6])
                        or abs(abs(row[6]) - 1) > 1e-6
                    ):
                        error = "WRIST bind pose must be identity"
                if error:
                    issue(field, error, i)
        for i, joint in enumerate(self.joints):
            try:
                number = joint_number(joint)
                if number in seen:
                    raise ValueError("Duplicate HandJoint identifier")
                seen.add(number)
            except ValueError as exc:
                issue("joint_ids", str(exc), i)
        order = self.editor["output_indices"]
        if (
            len(order) != n
            or any(type(i) is not int for i in order)
            or sorted(order) != list(range(n))
        ):
            issue("output_indices", "Taxel indices must cover 0..N-1 exactly")
        if self.editor.get("approximate"):
            issue(
                FIELDS[1],
                "Surface/manual placement requires physical calibration",
                severity="warning",
            )
        return problems


class State:
    def __init__(self, documents, model=None):
        self.documents, self.model = documents, copy.deepcopy(model or {})
        self.revision, self.saved, self.paths = 0, {}, {}

    def fingerprint(self, side):
        from .storage import metadata

        return json.dumps(metadata(self.documents[side]), sort_keys=True)

    def edit(self, action, *args, **kwargs):
        before = copy.deepcopy((self.documents, self.model))
        hashes = [self.fingerprint(s) for s in self.documents]
        try:
            result = action(*args, **kwargs)
        except Exception:
            self.documents, self.model = before
            raise
        if hashes != [self.fingerprint(s) for s in self.documents]:
            self.revision += 1
        return result

    def dirty(self):
        return any(self.saved.get(s) != self.fingerprint(s) for s in self.documents)

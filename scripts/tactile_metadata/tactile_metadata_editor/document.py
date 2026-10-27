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
        self.side, self.joints = side, []
        self.arrays = {field: [] for field in FIELDS}
        self.types = ("LBS", "HandSkeletonRef")
        self.editor = {"segment_ids": [], "output_indices": []}

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

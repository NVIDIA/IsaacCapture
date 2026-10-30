# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .document import Document, JOINTS, Segment, State, Taxel
from .geometry import VISUALIZATION_FROM_HAND, matrix, pose


def _pose_row(value: object) -> list[float]:
    if not isinstance(value, dict):
        raise ValueError("Pose row must be an object")
    position, orientation = value.get("position"), value.get("orientation")
    if not isinstance(position, dict) or not isinstance(orientation, dict):
        raise ValueError("Pose row requires position and orientation")
    return [
        *[position.get(key) for key in "xyz"],
        *[orientation.get(key) for key in "xyzw"],
    ]


def _rows(section: dict, name: str) -> list:
    value = section.get(name)
    if not isinstance(value, list):
        raise ValueError(f"{name}: expected an array")
    return value


def _joint_name(value: object) -> str:
    if isinstance(value, str):
        return value
    if type(value) is int and 0 <= value < len(JOINTS):
        return JOINTS[value]
    return f"INVALID_HAND_JOINT_{value}"


def parse(
    value: object,
    alignment: np.ndarray | list[list[float]] | None = None,
) -> Document:
    if not isinstance(value, dict):
        raise ValueError("Expected a TactileMetadata object")
    geometry, source = value.get("geometry_mapping"), value.get("segment_source")
    if not isinstance(geometry, dict) or not isinstance(source, dict):
        raise ValueError("Expected geometry_mapping and segment_source objects")
    joints = source.get("joint_ids")
    if not isinstance(joints, list):
        raise ValueError("joint_ids: expected an array")
    segment_rows = _rows(geometry, "segment_bind_poses")
    taxel_rows = _rows(geometry, "taxel_bind_poses")
    index_rows = _rows(geometry, "taxel_segment_indices")
    weight_rows = _rows(geometry, "taxel_segment_weights")
    if len(segment_rows) != len(joints):
        raise ValueError("segment_bind_poses must match joint_ids")
    if len({len(taxel_rows), len(index_rows), len(weight_rows)}) != 1:
        raise ValueError("All taxel arrays must have the same length")

    transform = np.asarray(
        VISUALIZATION_FROM_HAND if alignment is None else alignment, dtype=float
    )
    document = Document()
    document.alignment = transform.tolist()
    names = [_joint_name(joint) for joint in joints]
    document.segments = [
        Segment(joint, pose(transform @ matrix(_pose_row(row))))
        for joint, row in zip(names, segment_rows)
    ]
    for raw_pose, raw_indices, raw_weights in zip(taxel_rows, index_rows, weight_rows):
        indices = raw_indices.get("values") if isinstance(raw_indices, dict) else None
        weights = raw_weights.get("values") if isinstance(raw_weights, dict) else None
        if not isinstance(indices, list) or not isinstance(weights, list):
            raise ValueError("Taxel indices and weights require values arrays")
        if len(indices) != 4 or len(weights) != 4:
            raise ValueError("Taxel bindings require four slots")
        bindings = [
            None
            if type(weight) in (int, float) and weight == 0
            else names[index]
            if type(index) is int and 0 <= index < len(names)
            else f"INVALID_SEGMENT_INDEX_{index}"
            for index, weight in zip(indices, weights)
        ]
        document.taxels.append(
            Taxel(
                pose(transform @ matrix(_pose_row(raw_pose))),
                bindings,
                list(weights),
            )
        )
    return document


def _json_pose(values: list[float]) -> dict:
    return {
        "position": dict(zip("xyz", values[:3])),
        "orientation": dict(zip("xyzw", values[3:])),
    }


def metadata(document: Document) -> dict:
    inverse = np.linalg.inv(np.asarray(document.alignment, dtype=float))
    joints = [segment.joint for segment in document.segments]
    indices = {joint: row for row, joint in enumerate(joints)}
    segment_poses = [
        _json_pose(pose(inverse @ matrix(segment.pose)))
        for segment in document.segments
    ]
    taxel_poses = [
        _json_pose(pose(inverse @ matrix(taxel.pose))) for taxel in document.taxels
    ]
    binding_indices = [
        {
            "values": [
                indices.get(joint, 0 if joint is None else 0xFFFFFFFF)
                for joint in taxel.segments
            ]
        }
        for taxel in document.taxels
    ]
    binding_weights = [{"values": list(taxel.weights)} for taxel in document.taxels]
    return {
        "geometry_mapping_type": "LBS",
        "geometry_mapping": {
            "segment_bind_poses": segment_poses,
            "taxel_bind_poses": taxel_poses,
            "taxel_segment_indices": binding_indices,
            "taxel_segment_weights": binding_weights,
        },
        "segment_source_type": "HandSkeletonRef",
        "segment_source": {"joint_ids": joints},
    }


def load(
    path: str | Path,
    alignment: np.ndarray | list[list[float]] | None = None,
) -> Document:
    with Path(path).expanduser().resolve().open() as stream:
        return parse(json.load(stream), alignment)


def write(path: str | Path, value: object) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tactile-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save(state: State, path: str | Path) -> None:
    write(path, metadata(state.document))
    state.replace(state.document, str(Path(path).expanduser().resolve()))

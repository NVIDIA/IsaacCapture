# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path

from .document import Document, FIELDS


def parse(value, side=None):
    if not isinstance(value, dict) or any(
        not isinstance(value.get(k, {}), dict)
        for k in ("geometry_mapping", "segment_source")
    ):
        raise ValueError(
            "Expected a TactileMetadata object with object-valued sections"
        )
    d = Document(side)
    if d.side not in (None, "left", "right"):
        raise ValueError("Assign metadata to left or right explicitly")
    d.types = (value.get("geometry_mapping_type"), value.get("segment_source_type"))
    d.joints = copy.deepcopy(value.get("segment_source", {}).get("joint_ids"))
    if not isinstance(d.joints, list):
        raise ValueError("joint_ids: expected an array")
    for field in FIELDS:
        rows = value.get("geometry_mapping", {}).get(field)
        if not isinstance(rows, list):
            raise ValueError(f"{field}: expected an array")
        if field in FIELDS[:2] and any(
            isinstance(v, dict)
            and any(
                not isinstance(v.get(k, {}), dict) for k in ("position", "orientation")
            )
            for v in rows
        ):
            raise ValueError(f"{field}: position and orientation must be objects")
        d.arrays[field] = [
            (
                [v.get("position", {}).get(k) for k in "xyz"]
                + [v.get("orientation", {}).get(k) for k in "xyzw"]
                if field in FIELDS[:2]
                else v.get("values")
            )
            if isinstance(v, dict)
            else None
            for v in rows
        ]
    d.sync()
    return d


def metadata(d):
    value = dict(
        geometry_mapping_type=d.types[0],
        geometry_mapping={},
        segment_source_type=d.types[1],
        segment_source={"joint_ids": copy.deepcopy(d.joints)},
    )
    order = sorted(
        range(len(d.editor["output_indices"])),
        key=lambda i: d.editor["output_indices"][i],
    )
    aligned = all(len(d.arrays[f]) == len(order) for f in FIELDS[1:])
    for field, rows in d.arrays.items():
        if aligned and field in FIELDS[1:]:
            rows = [rows[i] for i in order]
        value["geometry_mapping"][field] = [
            {
                "position": dict(zip("xyz", r[:3])),
                "orientation": dict(zip("xyzw", r[3:])),
            }
            if field in FIELDS[:2] and r is not None
            else {"values": copy.deepcopy(r)}
            if field in FIELDS[2:]
            else None
            for r in rows
        ]
    return value


def load(path, side=None):
    with Path(path).open() as stream:
        return parse(json.load(stream), side)


def write(path, value):
    path = Path(path).resolve()
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tactile-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save(state, paths):
    if len({Path(p).resolve() for p in paths.values()}) != len(paths):
        raise ValueError("Each hand requires a different save path")
    for side, path in paths.items():
        write(path, metadata(state.documents[side]))
    for side, path in paths.items():
        state.paths[side], state.saved[side] = str(path), state.fingerprint(side)


def backup(d):
    with tempfile.NamedTemporaryFile(
        prefix="tactile-backup-", suffix=".json", delete=False
    ) as stream:
        path = stream.name
    write(path, metadata(d))
    return path


def export(documents, revision, destination, sdk=False):
    destination = Path(destination).resolve()
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise ValueError("Choose a new or empty output directory")
    with tempfile.TemporaryDirectory(
        prefix=".tactile-", dir=destination.parent
    ) as temporary:
        root, report = Path(temporary), {"revision": revision, "sides": {}}
        for side, d in documents.items():
            result = dict(local=d.check(), sdk="not checked")
            if any(v["severity"] == "error" for v in result["local"]):
                raise ValueError("Local errors prevent export")
            write(root / f"{side}.metadata.json", metadata(d))
            if sdk:
                from .sdk import check

                result["sdk"] = check(d, root / f"{side}.metadata.bin")
            report["sides"][side] = result
        if destination.exists():
            destination.rmdir()
        os.replace(root, destination)
    return report

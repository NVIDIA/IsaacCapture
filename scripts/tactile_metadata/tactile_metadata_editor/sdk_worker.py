# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File-protocol SDK worker; SDK imports are confined to this local process."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys

from .compiler import ValuePreparationError, canonical_values

_TYPES = ("HandSkeletonRef", "LBS", "TactileMetadata")
_EXIT_CODES = {"passed": 0, "failed": 1, "unavailable": 3}


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return number


def _read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(
            stream, parse_constant=_reject_constant, parse_float=_finite_float,
        )


def _write(path, value, *, binary=False):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb" if binary else "w") as stream:
        if binary:
            stream.write(value)
        else:
            json.dump(value, stream, allow_nan=False)


def _request(path):
    request = _read_json(path)
    if (
        not isinstance(request, dict) or type(request.get("version")) is not int
        or request["version"] != 1
        or request.get("operation") not in ("probe", "encode")
    ):
        raise ValueError("Invalid SDK request version or operation")
    paths = {}
    for name in ("values_file", "report_file", "binary_file"):
        value = request.get(name)
        if not isinstance(value, str):
            raise ValueError(f"{name} must be an absolute private path")
        target = Path(value)
        if not target.is_absolute() or target.resolve().parent != path.resolve().parent:
            raise ValueError(f"{name} must be in the request directory")
        paths[name] = target
    if len(set(paths.values()) | {path}) != 4:
        raise ValueError("SDK request and result paths must be distinct")
    return request["operation"], paths


def _capabilities(sdk, schema):
    supported = [name for name in _TYPES if callable(getattr(schema, name, None))]
    if supported != list(_TYPES):
        raise ImportError("SDK does not expose the required tactile metadata types")
    members = schema.HandJoint.__members__
    joints = {
        name: int(value) for name, value in members.items() if name != "NUM_JOINTS"
    }
    if not joints:
        raise ImportError("SDK does not expose HandJoint names and values")
    maximum = schema.TRACKER_METADATA_MAX_SIZE
    if type(maximum) is not int or maximum <= 0:
        raise ImportError("SDK does not expose a positive metadata size limit")
    return {
        "sdk_version": str(getattr(sdk, "__version__", "unknown")),
        "supported_types": supported,
        "joint_names_values": joints,
        "metadata_max_size": maximum,
    }


def _encode(values, schema, np, maximum):
    values = canonical_values(values)
    names = values["segment_source"]["joint_ids"]
    geometry = values["geometry_mapping"]
    arrays = {}
    for name, width, dtype in (
        ("segment_bind_poses", 7, np.float32),
        ("taxel_bind_poses", 7, np.float32),
        ("taxel_segment_indices", 4, np.uint32),
        ("taxel_segment_weights", 4, np.float32),
    ):
        rows = geometry[name]
        shaped = np.asarray(rows, dtype=dtype).reshape((len(rows), width))
        arrays[name] = np.array(shaped, dtype=dtype, copy=True, order="C")
    hand = schema.HandSkeletonRef([getattr(schema.HandJoint, name) for name in names])
    lbs = schema.LBS(
        arrays["segment_bind_poses"], arrays["taxel_bind_poses"],
        arrays["taxel_segment_indices"], arrays["taxel_segment_weights"],
    )
    metadata = schema.TactileMetadata(geometry_mapping=lbs, segment_source=hand)
    read_mapping, read_source = metadata.geometry_mapping, metadata.segment_source
    if read_mapping is None or read_source is None:
        raise ValueError("SDK metadata construction lost its mapping or source")
    read_joints = [joint.name for joint in read_source.joint_ids]
    if read_joints != names:
        raise ValueError("SDK joint identifier readback differs from the input order")
    for name, expected in arrays.items():
        actual = np.array(getattr(read_mapping, name), copy=True, order="C")
        if (
            actual.shape != expected.shape or actual.dtype != expected.dtype
            or not actual.flags.owndata or not actual.flags.c_contiguous
            or actual.tobytes(order="C") != expected.tobytes(order="C")
        ):
            raise ValueError(f"SDK {name} readback differs from the original slots")
    encoded = metadata.to_bytes()
    if not isinstance(encoded, bytes) or not encoded:
        raise ValueError("SDK to_bytes() did not return a nonempty byte buffer")
    if len(encoded) > maximum:
        raise ValueError(
            f"Encoded metadata has {len(encoded)} bytes; SDK limit is {maximum}"
        )
    return encoded


def run(request_path):
    operation, paths = _request(request_path)
    report = {
        "version": 1,
        "operation": operation,
        "status": "unavailable",
        "sdk_version": None,
        "supported_types": [],
        "joint_names_values": {},
        "metadata_max_size": None,
        "encoded_size": None,
        "issues": [],
    }
    try:
        import isaaccapture
        from isaaccapture import schema
        import numpy as np

        report.update(_capabilities(isaaccapture, schema))
    except Exception as error:
        report["issues"] = [{
            "severity": "error", "code": "sdk_unavailable",
            "message": f"{type(error).__name__}: {error}", "field": "sdk",
        }]
    else:
        try:
            if operation == "encode":
                encoded = _encode(
                    _read_json(paths["values_file"]), schema, np,
                    report["metadata_max_size"],
                )
                _write(paths["binary_file"], encoded, binary=True)
                report["encoded_size"] = len(encoded)
            report["status"] = "passed"
        except ValuePreparationError as error:
            report["status"] = "failed"
            report["issues"] = [asdict(issue) for issue in error.issues]
        except Exception as error:
            report["status"] = "failed"
            report["issues"] = [{
                "severity": "error", "code": "sdk_encode",
                "message": f"{type(error).__name__}: {error}", "field": "sdk",
            }]
    _write(paths["report_file"], report)
    return _EXIT_CODES[report["status"]]


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        return 2
    try:
        return run(Path(args[0]).resolve())
    except Exception as error:
        # Startup failures have no trusted report path; stderr remains diagnostic.
        sys.stderr.write(f"SDK worker failed: {type(error).__name__}: {error}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

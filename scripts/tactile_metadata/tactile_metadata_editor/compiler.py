# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only protocol preparation and a shared canonical numeric representation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

from .document import FORMAT, VERSION, import_values
from .domain import FrozenMap, Issue, Snapshot, thaw
from .mappings import mapping_for
from .profiles import profile_for
from .state import _index

UINT32_MAX = (1 << 32) - 1
FLOAT32_MAX = struct.unpack("<f", bytes.fromhex("ffff7f7f"))[0]


@dataclass(frozen=True)
class CompileFailure:
    revision: int
    device_id: str
    issues: tuple[Issue, ...]


class ValuePreparationError(ValueError):
    def __init__(self, issues):
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message for issue in self.issues))


def float32(value):
    """IEEE binary32 rounding, shared by JSON canonicalization and local checks."""
    # struct.pack reports huge Python integers as struct.error before rounding.
    if isinstance(value, int) and abs(value) > sys.float_info.max:
        return -math.inf if value < 0 else math.inf
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return -math.inf if value < 0 else math.inf


def quaternion_norm_error(row):
    squares = tuple(float32(component * component) for component in row[3:])
    total = squares[0]
    for square in squares[1:]:
        total = float32(total + square)
    return abs(float32(total - 1.0))


@dataclass(frozen=True)
class _NumericRows:
    source: FrozenMap
    segment_poses: tuple
    taxel_poses: tuple
    indices: tuple
    weights: tuple
    issues: tuple[Issue, ...]

    def values(self):
        return {
            "format": FORMAT,
            "version": VERSION,
            "segment_source": thaw(self.source),
            "geometry_mapping": {
                "type": "LBS",
                "segment_bind_poses": thaw(self.segment_poses),
                "taxel_bind_poses": thaw(self.taxel_poses),
                "taxel_segment_indices": thaw(self.indices),
                "taxel_segment_weights": thaw(self.weights),
            },
        }


def _numeric_rows(device, *, allow_empty):
    issues = []

    def error(code, message, field_name, entity_id=None):
        issues.append(Issue(
            "error", code, message, device.id, entity_id, field_name
        ))

    try:
        profile = profile_for(device)
        source = FrozenMap(profile.source_payload(device.segments))
        mapping = mapping_for(device)
    except (ValueError, TypeError) as failure:
        error("unsupported_metadata", str(failure), "segment_source")
        return tuple(issues)
    if device.mapping.get("type") != "LBS":
        error("unsupported_mapping", "Only LBS arrays are supported", "mapping")
        return tuple(issues)
    count, segment_count = len(device.taxels), len(device.segments)
    if not allow_empty and count == 0:
        error("empty_device", "Add taxels before formal export", "taxels")

    ordered = [None] * count
    for taxel in device.taxels:
        index = _index(taxel.taxel_index)
        if index is None or index > UINT32_MAX:
            error("output_index", "Output index must be a uint32 integer",
                  "taxel_index", taxel.id)
        elif index >= count:
            error("output_range", "Output indices must cover 0..N-1",
                  "taxel_index", taxel.id)
        elif ordered[index] is not None:
            error("output_duplicate", "Output index is already used",
                  "taxel_index", taxel.id)
        else:
            ordered[index] = taxel
    if any(taxel is None for taxel in ordered):
        error("output_gap", "Output indices have gaps", "taxel_index")
    if segment_count > UINT32_MAX + 1:
        error("segment_count", "Segment rows exceed uint32 addressing", "segments")

    def rows(
        value, expected, width, field_name, *, integer=False, nonnegative=False
    ):
        if not isinstance(value, tuple):
            error("array_shape", "Expected an array of rows", field_name)
            return ()
        if len(value) != expected:
            error(
                "row_count", f"Expected {expected} rows, got {len(value)}",
                field_name,
            )
        converted = []
        for row_index, row in enumerate(value):
            location = f"{field_name}[{row_index}]"
            entity_id = None
            if field_name == "segment_bind_poses" and row_index < segment_count:
                entity_id = device.segments[row_index].id
            elif field_name.startswith("taxel_") and row_index < count:
                taxel = ordered[row_index]
                entity_id = taxel.id if taxel is not None else None
            if not isinstance(row, tuple) or len(row) != width:
                error(
                    "row_shape", f"Expected a row of width {width}",
                    location, entity_id,
                )
                continue
            result, valid = [], True
            for slot, value in enumerate(row):
                target = f"{location}[{slot}]"
                is_number = not isinstance(value, bool) and (
                    isinstance(value, int)
                    or isinstance(value, float) and math.isfinite(value)
                )
                if not is_number:
                    error("number", "Expected a finite number", target, entity_id)
                    valid = False
                    continue
                if integer:
                    if value != int(value) or not 0 <= value <= UINT32_MAX:
                        error(
                            "index_dtype", "Index must be a uint32 integer",
                            target, entity_id,
                        )
                        valid = False
                    elif value >= segment_count:
                        error(
                            "segment_reference",
                            "Segment row is missing or out of range",
                            target, entity_id,
                        )
                        valid = False
                    else:
                        result.append(int(value))
                    continue
                if abs(value) > sys.float_info.max or abs(value) > FLOAT32_MAX:
                    error(
                        "float_dtype", "Value exceeds finite float32 range",
                        target, entity_id,
                    )
                    valid = False
                    continue
                if nonnegative and value < 0:
                    error(
                        "weight_negative", "Weight must be nonnegative",
                        target, entity_id,
                    )
                    valid = False
                    continue
                rounded = float32(float(value))
                if not math.isfinite(rounded):
                    error(
                        "float_dtype", "Conversion produced a nonfinite value",
                        target, entity_id,
                    )
                    valid = False
                    continue
                if value != 0 and rounded == 0:
                    issues.append(Issue(
                        "warning", "float32_underflow",
                        "A nonzero value rounds to zero in float32",
                        device.id, entity_id, target,
                    ))
                result.append(rounded)
            if valid:
                converted.append(tuple(result))
        return tuple(converted)

    segment_poses = rows(
        device.segment_bind_poses, segment_count, 7, "segment_bind_poses"
    )
    taxel_rows = tuple(
        taxel.bind_pose if taxel is not None else None for taxel in ordered
    )
    taxel_poses = rows(taxel_rows, count, 7, "taxel_bind_poses")
    indices = rows(
        device.skinning.indices, count, 4, "taxel_segment_indices", integer=True
    )
    weights = rows(
        device.skinning.weights, count, 4, "taxel_segment_weights", nonnegative=True
    )
    if any(issue.severity == "error" for issue in issues):
        return tuple(issues)
    for field_name, poses in (
        ("segment_bind_poses", segment_poses), ("taxel_bind_poses", taxel_poses)
    ):
        for row_index, pose in enumerate(poses):
            if quaternion_norm_error(pose) > float32(1e-4):
                entity = (
                    device.segments[row_index]
                    if field_name.startswith("segment") else ordered[row_index]
                )
                error(
                    "quaternion_norm",
                    "Quaternion is not unit length after float32 conversion",
                    f"{field_name}[{row_index}]", entity.id,
                )
    for row_index in range(count):
        for issue in mapping.validate_slots(
            indices[row_index], weights[row_index], segment_count
        ):
            field_name = issue.field or "weights"
            prefix = (
                "taxel_segment_indices" if field_name.startswith("indices")
                else "taxel_segment_weights"
            )
            suffix = field_name[field_name.find("["):] if "[" in field_name else ""
            issues.append(replace(
                issue, device_id=device.id, entity_id=ordered[row_index].id,
                field=f"{prefix}[{row_index}]{suffix}",
            ))
        active = {
            indices[row_index][slot] for slot in range(4)
            if weights[row_index][slot] > 0
        }
        if len(active) > 1:
            issues.append(Issue(
                "warning", "multi_segment_orientation",
                "Multiple segments define position without a unique rigid orientation",
                device.id, ordered[row_index].id, "skinning",
            ))
    if any(issue.severity == "error" for issue in issues):
        return tuple(issues)
    return _NumericRows(
        source, segment_poses, taxel_poses, indices, weights, tuple(issues)
    )


def canonical_values(values: Mapping) -> dict:
    """Target-dtype numeric JSON; empty schema vectors are allowed here."""
    document = import_values(values)
    prepared = _numeric_rows(document.devices[0], allow_empty=True)
    if isinstance(prepared, tuple):
        raise ValuePreparationError(prepared)
    return prepared.values()


def _json_values(values):
    return json.dumps(
        values, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    )


def metadata_fingerprint(values: Mapping) -> str:
    """Hash exactly the shared numeric representation, excluding editor data."""
    canonical = canonical_values(values)
    return hashlib.sha256(_json_values(canonical).encode("utf-8")).hexdigest()


@dataclass(frozen=True, init=False, eq=False)
class CompiledDevice:
    revision: int
    device_id: str
    side: str | None
    segment_rows: FrozenMap
    taxel_ids: tuple[str, ...]
    joint_names: tuple[str, ...]
    issues: tuple[Issue, ...]
    fingerprint: str
    _values: FrozenMap = field(repr=False)
    _segment_bind_poses: object = field(repr=False)
    _taxel_bind_poses: object = field(repr=False)
    _taxel_segment_indices: object = field(repr=False)
    _taxel_segment_weights: object = field(repr=False)

    def __init__(self, snapshot, device, prepared, np):
        data = prepared.values()
        members = {
            "revision": snapshot.revision,
            "device_id": device.id,
            "side": device.side,
            "segment_rows": FrozenMap({
                segment.id: row for row, segment in enumerate(device.segments)
            }),
            "taxel_ids": tuple(
                item.id for item in sorted(
                    device.taxels, key=lambda item: item.taxel_index
                )
            ),
            "joint_names": tuple(prepared.source["joint_ids"]),
            "issues": prepared.issues,
            "fingerprint": hashlib.sha256(
                _json_values(data).encode("utf-8")
            ).hexdigest(),
            "_values": FrozenMap(data),
        }
        for name, rows, width, dtype in (
            ("segment_bind_poses", prepared.segment_poses, 7, np.float32),
            ("taxel_bind_poses", prepared.taxel_poses, 7, np.float32),
            ("taxel_segment_indices", prepared.indices, 4, np.uint32),
            ("taxel_segment_weights", prepared.weights, 4, np.float32),
        ):
            wide = np.array(rows, dtype=np.float64, order="C", copy=True)
            wide = wide.reshape((len(rows), width))
            array = np.array(wide, dtype=dtype, order="C", copy=True)
            if (
                not array.flags.owndata or not array.flags.c_contiguous
                or array.dtype != np.dtype(dtype)
                or not bool(np.isfinite(array).all())
                or not bool((array.astype(np.float64) == wide).all())
            ):
                raise ValueError(f"{name}: target array conversion differs")
            array.flags.writeable = False
            members[f"_{name}"] = array
        for name, value in members.items():
            object.__setattr__(self, name, value)

    def _copy_array(self, name):
        # A read-only view still exposes its owning base through .base.
        result = getattr(self, name).copy(order="C")
        result.flags.writeable = False
        return result

    @property
    def segment_bind_poses(self):
        return self._copy_array("_segment_bind_poses")

    @property
    def taxel_bind_poses(self):
        return self._copy_array("_taxel_bind_poses")

    @property
    def taxel_segment_indices(self):
        return self._copy_array("_taxel_segment_indices")

    @property
    def taxel_segment_weights(self):
        return self._copy_array("_taxel_segment_weights")

    def values(self) -> dict:
        """Owned target-dtype JSON for SDK requests; saving uses Document rows."""
        return thaw(self._values)


def _model_issues(snapshot, device, model):
    issues = []
    config = snapshot.document.model

    def error(code, message, field_name, entity_id=None):
        issues.append(Issue(
            "error", code, message, device.id, entity_id, field_name
        ))

    if device.side is not None:
        if sum(
            other.side == device.side for other in snapshot.document.devices
        ) > 1:
            error("side_duplicate", "Device side is already used", "side")
        if (
            config is not None and config.sides in ("left", "right")
            and device.side != config.sides
        ):
            error(
                "model_side", "Device side differs from the model declaration",
                "side",
            )

    referenced = any(
        segment.frame_ref != "root" for segment in device.segments
    ) or any(taxel.surface_attachment is not None for taxel in device.taxels)
    referenced = referenced or device.wrist_mount is not None
    if config is None:
        if referenced:
            error(
                "model_missing", "Model attachments have no model configuration",
                "model",
            )
        return tuple(issues)
    if model is None:
        try:
            from .model import load_urdf

            model = load_urdf(config)
        except (ImportError, OSError, ValueError, TypeError, KeyError) as failure:
            error("model_unavailable", str(failure), "model")
            return tuple(issues)
    try:
        if Path(model.config.path).resolve() != Path(config.path).resolve():
            error(
                "model_mismatch", "Loaded model does not match this document",
                "model",
            )
            return tuple(issues)
        model = model.with_config(config)
        # Hash every model resource once, before checking individual attachments.
        issues.extend(
            replace(issue, device_id=device.id)
            for issue in model.resource_issues()
        )
        issues.extend(
            replace(issue, device_id=device.id)
            for issue in model.configuration_issues(
                snapshot.document.bind_configuration
            )
        )
        for segment in device.segments:
            reference = segment.frame_ref
            if reference == "root":
                continue
            if (
                not isinstance(reference, str)
                or not reference.startswith("link:")
                or reference[5:] not in model.links
            ):
                error(
                    "segment_link", "Segment references an unavailable link",
                    "frame_ref", segment.id,
                )
        if referenced:
            profile = profile_for(device)
            profile.root_pose(
                device, model, snapshot.document.bind_configuration
            )
        for taxel in device.taxels:
            if taxel.surface_attachment is not None and (
                not isinstance(taxel.surface_attachment, Mapping)
                or not model.attachment_valid(taxel.surface_attachment)
            ):
                error(
                    "surface_attachment", "Surface attachment is invalid or stale",
                    "surface_attachment", taxel.id,
                )
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as failure:
        error("model_reference", str(failure), "model")
    return tuple(issues)


def compile_device(
    snapshot: Snapshot, device_id: str, model=None
) -> CompiledDevice | CompileFailure:
    """Prepare protocol arrays without rewriting any editable rows or slots."""
    try:
        device = snapshot.document.device(device_id)
    except KeyError:
        issue = Issue(
            "error", "device_missing", "Device does not exist", device_id,
            field="devices",
        )
        return CompileFailure(snapshot.revision, device_id, (issue,))
    issues = list(_model_issues(snapshot, device, model))
    for taxel in device.taxels:
        if taxel.provenance == "surface":
            issues.append(Issue(
                "warning", "geometry_estimate",
                "Surface projection is a geometric estimate; "
                "physical calibration is unverified",
                device.id, taxel.id, "bind_pose",
            ))
        if taxel.generation_issue is not None:
            issues.append(Issue(
                "error", "generation", str(taxel.generation_issue), device.id,
                taxel.id, "bind_pose",
            ))
    try:
        issues.extend(profile_for(device).validate(device))
    except (ValueError, TypeError, OverflowError) as failure:
        issues.append(Issue(
            "error", "profile", str(failure), device.id, field="profile"
        ))
    prepared = _numeric_rows(device, allow_empty=False)
    if isinstance(prepared, tuple):
        issues.extend(prepared)
    else:
        issues.extend(prepared.issues)
    if any(issue.severity == "error" for issue in issues):
        return CompileFailure(snapshot.revision, device_id, tuple(issues))
    try:
        import numpy as np

        result = CompiledDevice(
            snapshot, device, replace(prepared, issues=tuple(issues)), np
        )
    except (ImportError, TypeError, ValueError, OverflowError) as failure:
        issues.append(Issue(
            "error", "array_preparation", str(failure), device.id,
            field="geometry_mapping",
        ))
        return CompileFailure(snapshot.revision, device_id, tuple(issues))
    return result

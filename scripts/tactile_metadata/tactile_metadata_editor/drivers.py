# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only named-joint, wrist-space and tactile-source view calculations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from uuid import UUID

from .checks import application_binding_issues, declared_source_indices
from .compiler import metadata_fingerprint
from .document import output_taxels
from .domain import FrozenMap, Issue
from .profiles import profile_for
from .profiles.hand import JOINT_NAMES, JOINT_VALUES


def _finite(value):
    try:
        return not isinstance(value, bool) and isinstance(
            value, (int, float)
        ) and math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class JointMappingResult:
    q: FrozenMap
    valid: bool
    rows: tuple[FrozenMap, ...]
    issues: tuple[Issue, ...]


def map_joints(binding, sample, model, q0) -> JointMappingResult:
    """JointState positions follow schema rad/m; physical units are not measured."""
    if binding is None:
        binding = {}
    if not isinstance(binding, Mapping):
        return JointMappingResult(
            FrozenMap(q0), False, (), application_binding_issues(binding, None),
        )
    hand = binding.get("hand_source")
    side = hand.get("side") if isinstance(hand, Mapping) else None
    issues = list(application_binding_issues(binding, side))
    free = set(model.actuated_joint_names)
    q = {name: q0.get(name, 0.0) for name in free}
    metadata = {row["name"]: row for row in model.joints}
    rows = []

    def error(code, message, field):
        issues.append(Issue("error", code, message, field=field))

    for name, value in q.items():
        if not _finite(value):
            error("bind_joint_value", f"Invalid q0 value for {name}",
                  "bind_configuration")
    source = binding.get("joint_source")
    if source is None:
        return JointMappingResult(FrozenMap(q), False, (), tuple(issues))
    if any(issue.severity == "error" for issue in issues):
        return JointMappingResult(FrozenMap(q), False, (), tuple(issues))
    if not isinstance(sample, Mapping) or sample.get("valid") is not True:
        error("joint_unavailable", "Configured JointState source is unavailable",
              "joint_source")
        return JointMappingResult(FrozenMap(q), False, (), tuple(issues))
    if sample.get("device_id") != source["device_id"]:
        error("joint_device", "JointState device ID differs from its configuration",
              "joint_source.device_id")
        return JointMappingResult(FrozenMap(q), False, (), tuple(issues))
    values = sample.get("joints")
    if not isinstance(values, (tuple, list)):
        error("joint_shape", "JointState sample has no named joint rows",
              "joint_source")
        values = ()
    by_name = {}
    for index, row in enumerate(values):
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("name"), str) or not row.get("name")
        ):
            error("joint_name", "JointState name is missing",
                  f"joint_source.joints[{index}]")
            continue
        name = row["name"]
        if name in by_name:
            error("joint_duplicate", f"JointState name is repeated: {name}",
                  "joint_source.joints")
        by_name[name] = row
    if any(issue.severity == "error" for issue in issues):
        return JointMappingResult(FrozenMap(q), False, (), tuple(issues))
    for index, row in enumerate(binding.get("joint_mapping", ())):
        target, source_name = row["urdf_joint"], row["source_joint"]
        field = f"application_binding.joint_mapping[{index}]"
        joint = metadata.get(target)
        raw, mapped, valid = None, None, False
        if target not in free:
            error("joint_target", f"Not a free URDF joint: {target}", field)
        else:
            sample_row = by_name.get(source_name)
            if (
                sample_row is None or sample_row.get("valid") is not True
                or not _finite(sample_row.get("position"))
            ):
                error("joint_value",
                      f"Missing or invalid source joint: {source_name}", field)
            else:
                raw = sample_row["position"]
                try:
                    mapped = float(row["scale"])*float(raw) + float(row["offset"])
                except (ValueError, OverflowError):
                    mapped = None
                if not _finite(mapped):
                    error("joint_conversion",
                          f"Nonfinite mapped value for {target}", field)
                else:
                    q[target], valid = mapped, True
                    lower, upper = joint.get("lower"), joint.get("upper")
                    if ((lower is not None and mapped < lower)
                            or (upper is not None and mapped > upper)):
                        issues.append(Issue(
                            "warning", "joint_limit",
                            f"Mapped value exceeds the URDF limit: {target}",
                            field=field,
                        ))
        rows.append(FrozenMap({
            "urdf_joint": target, "source_joint": source_name, "raw": raw,
            "mapped": mapped, "valid": valid,
            "unit": "m" if joint and joint["type"] == "prismatic" else "rad",
            "lower": joint.get("lower") if joint else None,
            "upper": joint.get("upper") if joint else None,
        }))
    issues.append(Issue(
        "info", "joint_units",
        "Positions follow schema rad/m conventions; physical source units "
        "are not independently verified",
        field="joint_source",
    ))
    return JointMappingResult(
        FrozenMap(q), not any(issue.severity == "error" for issue in issues),
        tuple(rows), tuple(issues),
    )


@dataclass(frozen=True)
class SegmentFrame:
    poses: tuple
    valid: tuple[bool, ...]
    issues: tuple[Issue, ...]


def current_segments(joint_names, hand_sample) -> SegmentFrame:
    import numpy as np

    from .geometry import matrix_pose, pose_matrix

    names = tuple(joint_names)
    poses, valid, issues = [None]*len(names), [False]*len(names), []
    try:
        if not isinstance(hand_sample, Mapping) or hand_sample.get("valid") is not True:
            raise ValueError("HandPose source is unavailable")
        positions = hand_sample.get("positions")
        orientations = hand_sample.get("orientations_xyzw")
        flags = hand_sample.get("is_valid")
        if any(not isinstance(value, (tuple, list)) or len(value) != len(JOINT_NAMES)
               for value in (positions, orientations, flags)):
            raise ValueError("HandPose joint arrays do not match the hand profile")
        wrist = JOINT_VALUES["WRIST"]
        if flags[wrist] is not True:
            raise ValueError("HandPose WRIST is invalid")
        if positions[wrist] is None or orientations[wrist] is None:
            raise ValueError("HandPose WRIST geometry is unavailable")
        root = np.linalg.inv(pose_matrix((*positions[wrist], *orientations[wrist])))
        for row, name in enumerate(names):
            try:
                joint = JOINT_VALUES.get(name)
                if joint is None or flags[joint] is not True:
                    raise ValueError(f"Required HandPose joint is invalid: {name}")
                if positions[joint] is None or orientations[joint] is None:
                    raise ValueError(f"HandPose joint geometry is unavailable: {name}")
                poses[row] = matrix_pose(root @ pose_matrix((
                    *positions[joint], *orientations[joint],
                )))
                valid[row] = True
            except (ValueError, TypeError, IndexError) as error:
                issues.append(Issue(
                    "warning", "hand_joint", str(error),
                    field=f"hand_source.{name}",
                ))
    except (ValueError, TypeError, IndexError, np.linalg.LinAlgError) as error:
        issues.append(Issue("warning", "hand_wrist", str(error), field="hand_source"))
    return SegmentFrame(tuple(poses), tuple(valid), tuple(issues))


@dataclass(frozen=True)
class ViewRoots:
    metadata_to_view: object
    model_to_view: object | None


def view_root(device, model, q, *, live=False, side_offset=(0, 0, 0)) -> ViewRoots:
    import numpy as np

    from .geometry import immutable_array

    if len(side_offset) != 3 or not all(_finite(value) for value in side_offset):
        raise ValueError("Display offset must contain three finite meters")
    offset = np.eye(4)
    offset[:3, 3] = side_offset
    if model is None:
        return ViewRoots(immutable_array(offset), None)
    if live:
        binding = device.application_binding
        if not binding or binding.get("joint_source") is None:
            return ViewRoots(immutable_array(offset), None)
        root = profile_for(device).root_pose(device, model, q)
        return ViewRoots(
            immutable_array(offset), immutable_array(offset @ np.linalg.inv(root)),
        )
    root = profile_for(device).root_pose(device, model, q)
    return ViewRoots(immutable_array(offset @ root), immutable_array(offset))


@dataclass(frozen=True)
class CorrespondenceResult:
    indices: FrozenMap
    forces: FrozenMap
    issues: tuple[Issue, ...]
    automatic: bool = False


def source_correspondence(
    device, live_snapshot, *, expected_epoch, local_values=None,
) -> CorrespondenceResult:
    indices, declared_issues = declared_source_indices(device)
    issues = list(declared_issues)
    empty_forces = {taxel.id: None for taxel in device.taxels}

    def result(automatic=False):
        return CorrespondenceResult(
            FrozenMap(indices), FrozenMap(empty_forces), tuple(issues), automatic,
        )

    def error(
        code, message, field="application_binding.source_taxel_map", entity_id=None,
    ):
        issues.append(Issue("error", code, message, device.id, entity_id, field))

    if any(issue.severity == "error" for issue in issues):
        return result()
    if live_snapshot is None or live_snapshot.epoch != expected_epoch:
        error("source_epoch", "No complete frame from the active receiving epoch")
        return result()
    binding = device.application_binding or {}
    collection = binding.get("tactile_collection_id")
    baseline = (
        live_snapshot.metadata.get(collection)
        if isinstance(collection, str) else None
    )
    if baseline is None:
        error("source_metadata", "The configured producer metadata is unavailable")
        return result()
    fingerprint = baseline["fingerprint"].lower()
    count = len(baseline["values"]["geometry_mapping"]["taxel_bind_poses"])
    taxels = {UUID(taxel.id): taxel for taxel in device.taxels}
    descriptors = {
        taxels[UUID(row["taxel_id"])].id: row
        for row in binding.get("source_taxel_map", ())
    }
    descriptors.update({
        taxel.id: taxel.input_ref for taxel in device.taxels
        if taxel.input_ref is not None
    })
    if any(row["metadata_fingerprint"].lower() != fingerprint
           for row in descriptors.values()):
        indices = {}
        error("source_baseline", "Producer metadata changed; the entire "
              "source correspondence is stale")
        return result()
    for taxel_id, index in indices.items():
        if index >= count:
            error("source_range", "Input taxel row exceeds the producer metadata",
                  entity_id=taxel_id)
    if any(issue.severity == "error" for issue in issues):
        return result()
    local_fingerprint = None
    if local_values is not None:
        try:
            local_fingerprint = metadata_fingerprint(local_values)
        except (ValueError, TypeError, KeyError, OverflowError):
            pass
    automatic = False
    if not indices and not descriptors and local_fingerprint == fingerprint:
        try:
            indices = {
                taxel.id: index
                for index, taxel in enumerate(output_taxels(device))
            }
            automatic = True
        except ValueError as failure:
            error("source_order", str(failure))
    # Incomplete drafts cannot prove that edited axes match the source calibration.
    if (
        indices and local_values is not None
        and local_fingerprint != fingerprint
    ):
        issues.append(Issue(
            "warning", "layout_only_force_view",
            "Received calibration forces shown on edited geometry/skinning "
            "are layout inspection, not recalibration",
            device.id, field="application_binding.source_taxel_map",
        ))
    if not indices:
        issues.append(Issue(
            "info", "source_unmapped",
            "Establish an explicit source table; equal taxel counts do not bind rows",
            device.id, field="application_binding.source_taxel_map",
        ))
    source = live_snapshot.sources.get("tactile", {}).get(collection)
    if not isinstance(source, Mapping) or source.get("valid") is not True:
        issues.append(Issue(
            "warning", "tactile_unavailable", "Tactile force source is unavailable",
            device.id, field="live",
        ))
        return result(automatic)
    forces = source.get("forces")
    if not isinstance(forces, (tuple, list)) or len(forces) != count or any(
        not isinstance(row, (tuple, list)) or len(row) != 3
        or not all(_finite(value) for value in row) for row in forces
    ):
        error("force_count", "Finite force rows must match producer metadata count")
        return result(automatic)
    for taxel_id, index in indices.items():
        empty_forces[taxel_id] = forces[index]
    return result(automatic)

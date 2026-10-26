# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared local checks and independent official SDK status."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING
from uuid import UUID

from .compiler import UINT32_MAX, CompiledDevice, CompileFailure, compile_device
from .domain import FrozenMap, Issue, Snapshot
from .state import _index

if TYPE_CHECKING:
    from .sdk import SdkResult


@dataclass(frozen=True)
class CheckResult:
    revision: int
    device_id: str
    compiled: CompiledDevice | None
    issues: tuple[Issue, ...]
    local_status: str
    sdk_status: str = "not_run"
    sdk_result: SdkResult | None = None


def application_binding_issues(binding, side, device_id=None):
    """Persisted source configuration, independent of runtime or model loading."""
    issues = []

    def error(code, message, field):
        issues.append(Issue("error", code, message, device_id, field=field))

    if binding is None:
        return ()
    if not isinstance(binding, Mapping):
        return (Issue(
            "error", "binding_shape", "Expected an application binding object",
            device_id, field="application_binding",
        ),)
    hand = binding.get("hand_source")
    if hand is not None:
        if (
            not isinstance(hand, Mapping) or hand.get("side") != side
            or side not in ("left", "right")
        ):
            error("hand_side", "Hand source side must match the device",
                  "application_binding.hand_source")
        else:
            vendor = hand.get("vendor_id")
            params = hand.get("vendor_params", {})
            if vendor is not None and (not isinstance(vendor, str) or not vendor):
                error("hand_vendor", "Vendor ID must be a nonempty string or null",
                      "application_binding.hand_source.vendor_id")
            if not isinstance(params, Mapping) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in params.items()
            ):
                error("hand_vendor_params", "Vendor parameters must be strings",
                      "application_binding.hand_source.vendor_params")
            elif params and not vendor:
                error("hand_vendor", "Vendor parameters require an explicit vendor ID",
                      "application_binding.hand_source.vendor_id")
    source = binding.get("joint_source")
    if source is not None:
        if not isinstance(source, Mapping) or any(
            not isinstance(source.get(name), str) or not source.get(name)
            for name in ("collection_id", "device_id")
        ):
            error("joint_source", "Joint source requires collection ID and device ID",
                  "application_binding.joint_source")
    rows = binding.get("joint_mapping", ())
    if not isinstance(rows, (tuple, list)):
        error("joint_mapping", "Joint mapping must be an array",
              "application_binding.joint_mapping")
        return tuple(issues)
    used = set()
    for index, row in enumerate(rows):
        field = f"application_binding.joint_mapping[{index}]"
        if not isinstance(row, Mapping):
            error("joint_mapping", "Joint mapping row must be an object", field)
            continue
        for name in ("urdf_joint", "source_joint"):
            if not isinstance(row.get(name), str) or not row.get(name):
                error("joint_name", "Joint names must be nonempty strings",
                      f"{field}.{name}")
        target = row.get("urdf_joint")
        if isinstance(target, str):
            if target in used:
                error("joint_duplicate", "URDF joint has multiple mappings", field)
            used.add(target)
        for name in ("scale", "offset"):
            value = row.get(name)
            try:
                valid = not isinstance(value, bool) and isinstance(
                    value, (int, float)
                ) and math.isfinite(value)
            except OverflowError:
                valid = False
            if not valid:
                error("joint_number", "Scale and offset must be finite numbers",
                      f"{field}.{name}")
    return tuple(issues)


def declared_source_indices(device):
    """Check explicit offline declarations; no live baseline is inferred."""
    issues = list(application_binding_issues(
        device.application_binding, device.side, device.id
    ))
    declared = {}
    entities = {UUID(taxel.id): taxel for taxel in device.taxels}
    binding = device.application_binding

    def error(code, message, field, entity_id=None):
        issues.append(Issue(
            "error", code, message, device.id, entity_id, field
        ))

    if binding is not None and not isinstance(binding, Mapping):
        binding = None
    collection = binding.get("tactile_collection_id") if binding else None
    if collection is not None and not isinstance(collection, str):
        error("collection_id", "Collection ID must be a string or null",
              "application_binding.tactile_collection_id")

    def descriptor(raw, field, taxel):
        if not isinstance(raw, Mapping):
            error("source_shape", "Expected an explicit source reference",
                  field, taxel.id)
            return None
        start = len(issues)
        index = _index(raw.get("input_taxel_index"))
        if index is None or index > UINT32_MAX:
            error("source_index", "Input index must be a uint32 integer",
                  field + ".input_taxel_index", taxel.id)
        source_collection = raw.get("collection_id")
        if (
            not isinstance(source_collection, str) or not source_collection
            or not collection or source_collection != collection
        ):
            error("source_collection", "Source collection must match the binding",
                  field + ".collection_id", taxel.id)
        side = raw.get("side")
        if side not in ("left", "right") or side != device.side:
            error("source_side", "Source side must match the device",
                  field + ".side", taxel.id)
        fingerprint = raw.get("metadata_fingerprint")
        if (
            not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in fingerprint)
        ):
            error("source_fingerprint", "Source fingerprint must be SHA256 hex",
                  field + ".metadata_fingerprint", taxel.id)
        else:
            fingerprint = fingerprint.lower()
        if "taxel_id" in raw:
            try:
                identity = UUID(raw["taxel_id"])
            except (ValueError, TypeError, AttributeError):
                identity = None
            if identity != UUID(taxel.id):
                error("source_taxel", "Source reference has a different taxel UUID",
                      field + ".taxel_id", taxel.id)
        if len(issues) != start:
            return None
        return (index, source_collection, side, fingerprint)

    rows = binding.get("source_taxel_map", ()) if binding else ()
    if not isinstance(rows, (tuple, list)):
        error("source_map_shape", "Source map must be an array",
              "application_binding.source_taxel_map")
        rows = ()
    seen = set()
    for row_index, row in enumerate(rows):
        field = f"application_binding.source_taxel_map[{row_index}]"
        try:
            identity = UUID(row.get("taxel_id")) if isinstance(row, Mapping) else None
        except (ValueError, TypeError, AttributeError):
            identity = None
        taxel = entities.get(identity)
        if taxel is None:
            error("source_taxel", "Source map references a missing taxel UUID", field)
            continue
        if identity in seen:
            error("source_duplicate_taxel", "Taxel has multiple source declarations",
                  field, taxel.id)
            continue
        seen.add(identity)
        value = descriptor(row, field, taxel)
        if value is not None:
            declared[taxel.id] = value
    for taxel in device.taxels:
        if taxel.input_ref is None:
            continue
        field = "input_ref"
        value = descriptor(taxel.input_ref, field, taxel)
        if value is None:
            continue
        if taxel.id in declared and declared[taxel.id] != value:
            error("source_conflict", "Taxel and source map declarations disagree",
                  field, taxel.id)
        else:
            declared[taxel.id] = value
    used, fingerprints = {}, set()
    for taxel_id, value in declared.items():
        index, _, _, fingerprint = value
        if index in used:
            error("source_duplicate_index", "Input row is used by multiple taxels",
                  "input_ref.input_taxel_index", taxel_id)
        used[index] = taxel_id
        fingerprints.add(fingerprint)
    if len(fingerprints) > 1:
        error("source_baseline", "Source declarations use different baselines",
              "application_binding.source_taxel_map")
    return FrozenMap({key: value[0] for key, value in declared.items()}), tuple(issues)


def issue_summary(issue):
    return {
        "severity": issue.severity,
        "code": issue.code,
        "message": issue.message,
        "device_id": issue.device_id,
        "entity_id": issue.entity_id,
        "field": issue.field,
    }


def check_summary(result):
    """Finite JSON details only; arrays and encoded bytes stay outside reports."""
    sdk = result.sdk_result
    return {
        "revision": result.revision,
        "device_id": result.device_id,
        "local_status": result.local_status,
        "sdk_status": result.sdk_status,
        "sdk_version": sdk.report.get("sdk_version") if sdk else None,
        "metadata_max_size": sdk.report.get("metadata_max_size") if sdk else None,
        "encoded_size": len(sdk.encoded) if sdk and sdk.encoded is not None else None,
        "issues": [issue_summary(issue) for issue in result.issues],
    }


def checks_exit_code(results, *, use_sdk=False):
    results = tuple(results)
    if any(result.local_status != "passed" for result in results):
        return 1
    if use_sdk:
        if any(result.sdk_status == "failed" for result in results):
            return 1
        if any(result.sdk_status == "unavailable" for result in results):
            return 3
        if any(result.sdk_status != "passed" for result in results):
            return 1
    return 0


def check_device(
    snapshot: Snapshot, device_id: str, model=None, use_sdk=False
) -> CheckResult:
    prepared = compile_device(snapshot, device_id, model)
    compiled = None if isinstance(prepared, CompileFailure) else prepared
    issues = list(prepared.issues)
    try:
        _, binding_issues = declared_source_indices(
            snapshot.document.device(device_id)
        )
        issues.extend(binding_issues)
    except KeyError:
        pass
    if any(issue.severity == "error" for issue in issues):
        compiled = None
    sdk_result, sdk_status = None, "not_run"
    if use_sdk and compiled is not None:
        try:
            from .sdk import encode

            sdk_result = encode(compiled.values())
            sdk_status = sdk_result.status
            for row in sdk_result.report.get("issues", ()):
                issues.append(Issue(
                    row["severity"], row["code"], row["message"], device_id,
                    field=row.get("field") if isinstance(row.get("field"), str)
                    else "sdk",
                ))
        except (OSError, ValueError, TypeError) as error:
            sdk_status = "unavailable"
            issues.append(Issue(
                "error", "sdk_invocation", str(error), device_id, field="sdk"
            ))
    return CheckResult(
        snapshot.revision, device_id, compiled, tuple(issues),
        "passed" if compiled is not None else "failed",
        sdk_status, sdk_result,
    )

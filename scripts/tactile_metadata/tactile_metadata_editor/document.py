# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata JSON, lossless draft records and atomic local persistence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import UUID

from .domain import (
    Device,
    Document,
    FrozenMap,
    ModelConfig,
    Segment,
    Skinning,
    Snapshot,
    Taxel,
    freeze,
    new_id,
    thaw,
)

FORMAT = "isaaccapture.tactile-values"
VERSION = 1


class MetadataFormatError(ValueError):
    """The container, version or discriminator cannot be interpreted."""


def _object(value: Any, field: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise MetadataFormatError(f"{field}: expected an object")
    return value


def _array(value: Any, field: str) -> tuple:
    if not isinstance(value, (list, tuple)):
        raise MetadataFormatError(f"{field}: expected an array")
    return freeze(value)


def _version(value: Mapping, field: str):
    version = value.get("version")
    if type(version) is not int or version != VERSION:
        raise MetadataFormatError(f"{field}.version: expected {VERSION}")


def _side(value: Any) -> str | None:
    if value not in (None, "left", "right"):
        raise MetadataFormatError("editor.side: expected left or right")
    return value


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise MetadataFormatError(f"{field}: expected a UUID")
    try:
        UUID(value)
    except ValueError as error:
        raise MetadataFormatError(f"{field}: expected a UUID") from error
    return value


def _source_arrays(values: Mapping) -> tuple:
    if values.get("format") != FORMAT:
        raise MetadataFormatError(f"format: expected {FORMAT}")
    _version(values, "metadata")
    source = _object(values.get("segment_source"), "segment_source")
    mapping = _object(values.get("geometry_mapping"), "geometry_mapping")
    if source.get("type") != "HandSkeletonRef":
        raise MetadataFormatError("segment_source.type: unsupported source")
    if mapping.get("type") != "LBS":
        raise MetadataFormatError("geometry_mapping.type: unsupported mapping")
    return (
        _array(source.get("joint_ids"), "segment_source.joint_ids"),
        _array(mapping.get("segment_bind_poses"), "segment_bind_poses"),
        _array(mapping.get("taxel_bind_poses"), "taxel_bind_poses"),
        _array(mapping.get("taxel_segment_indices"), "taxel_segment_indices"),
        _array(mapping.get("taxel_segment_weights"), "taxel_segment_weights"),
    )


def import_values(values: Mapping, side: str | None = None) -> Document:
    """Accept semantic/shape errors as raw records, without repairing arrays."""
    values = _object(values, "metadata")
    if "editor" in values and "draft" in _object(values["editor"], "editor"):
        raise MetadataFormatError(
            "editor.draft and numeric fields are mutually exclusive"
        )
    joints, segment_poses, taxel_poses, indices, weights = _source_arrays(values)
    device = Device(
        side=_side(side),
        segments=tuple(Segment(semantic=joint) for joint in joints),
        segment_bind_poses=segment_poses,
        taxels=tuple(
            Taxel(taxel_index=index, bind_pose=pose, provenance="imported")
            for index, pose in enumerate(taxel_poses)
        ),
        skinning=Skinning(indices, weights),
    )
    return Document(devices=(device,))


def output_taxels(device: Device) -> tuple[Taxel, ...]:
    indices = []
    for taxel in device.taxels:
        index = taxel.taxel_index
        if (
            isinstance(index, bool)
            or not isinstance(index, (int, float))
            or (isinstance(index, float) and not math.isfinite(index))
            or index != int(index)
        ):
            raise ValueError(
                "A complete container requires integral taxel output indices"
            )
        indices.append(int(index))
    if sorted(indices) != list(range(len(indices))):
        raise ValueError(
            "A complete container requires contiguous unique taxel output indices"
        )
    return tuple(sorted(device.taxels, key=lambda taxel: taxel.taxel_index))


def raw_values(device: Device, *, output_order: bool = False) -> dict:
    """Return row-preserving values; this is not a validity or dtype check."""
    taxels = output_taxels(device) if output_order else device.taxels
    return {
        "format": FORMAT,
        "version": VERSION,
        "segment_source": {
            "type": device.profile.get("type"),
            "joint_ids": [thaw(segment.semantic) for segment in device.segments],
        },
        "geometry_mapping": {
            "type": device.mapping.get("type"),
            "segment_bind_poses": thaw(device.segment_bind_poses),
            "taxel_bind_poses": [thaw(taxel.bind_pose) for taxel in taxels],
            "taxel_segment_indices": thaw(device.skinning.indices),
            "taxel_segment_weights": thaw(device.skinning.weights),
        },
    }


def _resolve_path(value: Any, base: Path, field: str) -> str:
    if not isinstance(value, str):
        raise MetadataFormatError(f"{field}: expected a local path")
    if "://" in value:
        raise MetadataFormatError(f"{field}: expected a local filesystem path")
    path = Path(value).expanduser()
    return str((base / path).resolve() if not path.is_absolute() else path.resolve())


def _load_model(value: Any, base: Path) -> ModelConfig | None:
    if value is None:
        return None
    model = _object(value, "editor.model")
    sides = model.get("sides")
    if sides not in ("left", "right", "both"):
        raise MetadataFormatError("editor.model.sides: expected left, right or both")
    packages = _object(model.get("packages", {}), "editor.model.packages")
    return ModelConfig(
        path=_resolve_path(model.get("path"), base, "editor.model.path"),
        sides=sides,
        packages=FrozenMap(
            {
                key: _resolve_path(path, base, f"editor.model.packages.{key}")
                for key, path in packages.items()
            }
        ),
        asset_hashes=FrozenMap(
            _object(model.get("asset_hashes", {}), "editor.model.asset_hashes")
        ),
        link_adjustments=FrozenMap(
            _object(model.get("link_adjustments", {}), "editor.model.link_adjustments")
        ),
    )


def _ids(value: Any, count: int, field: str) -> tuple[str, ...]:
    if value is None:
        return tuple(new_id() for _ in range(count))
    result = _array(value, field)
    if len(result) != count:
        raise MetadataFormatError(f"{field}: expected {count} identifiers")
    for item in result:
        _identifier(item, field)
    if len({UUID(item) for item in result}) != len(result):
        raise MetadataFormatError(f"{field}: duplicate identifiers")
    return result


def _full_device(values: Mapping, editor: Mapping, side: str | None) -> Device:
    device = import_values(values, side).devices[0]
    segment_ids = _ids(editor.get("segment_ids"), len(device.segments), "segment_ids")
    taxel_ids = _ids(editor.get("taxel_ids"), len(device.taxels), "taxel_ids")
    if {UUID(item) for item in segment_ids}.intersection(
        UUID(item) for item in taxel_ids
    ):
        raise MetadataFormatError("editor: duplicate entity identifiers")
    links = editor.get("segment_links")
    if links is not None:
        links = _array(links, "segment_links")
        if len(links) != len(device.segments):
            raise MetadataFormatError("segment_links: expected one entry per segment")
        if any(link is not None and not isinstance(link, str) for link in links):
            raise MetadataFormatError("segment_links: expected names or null")
    hints = editor.get("taxel_hints")
    if hints is not None:
        hints = _array(hints, "taxel_hints")
        if len(hints) != len(device.taxels):
            raise MetadataFormatError("taxel_hints: expected one entry per taxel")
    segments = []
    for index, segment in enumerate(device.segments):
        link = links[index] if links is not None else None
        # B remains authoritative; the model layer later restores local mounts.
        segments.append(
            Segment(
                id=segment_ids[index],
                semantic=segment.semantic,
                frame_ref=f"link:{link}" if link else "root",
            )
        )
    taxels = []
    for index, taxel in enumerate(device.taxels):
        hint = _object(hints[index], f"taxel_hints[{index}]") if hints else {}
        taxels.append(
            Taxel(
                id=taxel_ids[index],
                taxel_index=taxel.taxel_index,
                bind_pose=taxel.bind_pose,
                surface_attachment=hint.get("surface_attachment"),
                generation_issue=hint.get("generation_issue"),
                provenance=hint.get("provenance", "imported"),
                input_ref=hint.get("input_ref"),
            )
        )
    binding = editor.get("application_binding")
    if binding is not None:
        _object(binding, "application_binding")
    return Device(
        id=_identifier(editor.get("device_id", device.id), "editor.device_id"),
        side=side,
        segments=tuple(segments),
        segment_bind_poses=device.segment_bind_poses,
        taxels=tuple(taxels),
        skinning=device.skinning,
        wrist_mount=editor.get("wrist_mount"),
        application_binding=binding,
    )


def _draft_device(draft: Any, editor: Mapping, side: str | None) -> Device:
    draft = _object(draft, "editor.draft")
    profile = _object(draft.get("profile"), "draft.profile")
    mapping = _object(draft.get("mapping"), "draft.mapping")
    if profile.get("type") != "HandSkeletonRef" or mapping.get("type") != "LBS":
        raise MetadataFormatError("draft: unsupported profile or mapping")
    segments = []
    for index, value in enumerate(_array(draft.get("segments"), "draft.segments")):
        row = _object(value, f"draft.segments[{index}]")
        frame_ref = row.get("frame_ref", "root")
        if not isinstance(frame_ref, str):
            raise MetadataFormatError("segment.frame_ref: expected a string")
        segments.append(
            Segment(
                id=_identifier(row.get("id"), "segment.id"),
                semantic=row.get("semantic"),
                frame_ref=frame_ref,
                mount_pose=row.get("mount_pose"),
            )
        )
    taxels = []
    for index, value in enumerate(_array(draft.get("taxels"), "draft.taxels")):
        row = _object(value, f"draft.taxels[{index}]")
        taxels.append(
            Taxel(
                id=_identifier(row.get("id"), "taxel.id"),
                taxel_index=row.get("taxel_index"),
                bind_pose=row.get("bind_pose"),
                surface_attachment=row.get("surface_attachment"),
                generation_issue=row.get("generation_issue"),
                provenance=row.get("provenance"),
                input_ref=row.get("input_ref"),
            )
        )
    identifiers = [item.id for item in (*segments, *taxels)]
    if len({UUID(item) for item in identifiers}) != len(identifiers):
        raise MetadataFormatError("draft: duplicate entity identifiers")
    skinning = _object(draft.get("skinning"), "draft.skinning")
    binding = editor.get("application_binding")
    if binding is not None:
        _object(binding, "application_binding")
    return Device(
        id=_identifier(editor.get("device_id", new_id()), "editor.device_id"),
        side=side,
        profile=FrozenMap(profile),
        mapping=FrozenMap(mapping),
        segments=tuple(segments),
        segment_bind_poses=_array(
            draft.get("segment_bind_poses"), "draft.segment_bind_poses"
        ),
        taxels=tuple(taxels),
        skinning=Skinning(
            _array(skinning.get("indices"), "draft.skinning.indices"),
            _array(skinning.get("weights"), "draft.skinning.weights"),
            FrozenMap(_object(
                skinning.get("deleted_rows", {}), "draft.skinning.deleted_rows"
            )),
        ),
        wrist_mount=editor.get("wrist_mount"),
        application_binding=binding,
    )


def load_metadata(path: str | Path, side: str | None = None) -> Document:
    try:
        return _load_metadata(path, side)
    except MetadataFormatError:
        raise
    except (TypeError, ValueError) as error:
        raise MetadataFormatError(str(error)) from error


def _load_metadata(path: str | Path, side: str | None) -> Document:
    path = Path(path).expanduser().resolve()
    try:
        with path.open(encoding="utf-8") as stream:
            values = json.load(
                stream,
                parse_constant=_reject_constant,
                parse_float=_finite_float,
            )
    except (json.JSONDecodeError, UnicodeError) as error:
        raise MetadataFormatError(str(error)) from error
    values = _object(values, "metadata")
    if values.get("format") != FORMAT:
        raise MetadataFormatError(f"format: expected {FORMAT}")
    _version(values, "metadata")
    editor = values.get("editor")
    if editor is None:
        return import_values(values, side)
    editor = _object(editor, "editor")
    _version(editor, "editor")
    stored_side = _side(editor.get("side"))
    if side is not None and stored_side is not None and side != stored_side:
        raise MetadataFormatError("Requested side conflicts with editor.side")
    assigned_side = _side(side) if side is not None else stored_side
    if "draft" in editor:
        if "segment_source" in values or "geometry_mapping" in values:
            raise MetadataFormatError(
                "editor.draft and numeric fields are mutually exclusive"
            )
        device = _draft_device(editor["draft"], editor, assigned_side)
    else:
        device = _full_device(values, editor, assigned_side)
    return Document(
        model=_load_model(editor.get("model"), path.parent),
        devices=(device,),
        bind_configuration=FrozenMap(
            _object(editor.get("bind_configuration", {}), "bind_configuration")
        ),
    )


def _reject_constant(value: str):
    raise MetadataFormatError(f"Non-finite JSON number is not permitted: {value}")


def _finite_float(value: str) -> float:
    # parse_constant does not see exponent overflow such as the JSON number 1e999.
    number = float(value)
    if not math.isfinite(number):
        raise MetadataFormatError(f"Non-finite JSON number is not permitted: {value}")
    return number


def device_draft(device: Device) -> dict:
    return {
        "profile": thaw(device.profile),
        "mapping": thaw(device.mapping),
        "segments": [
            {
                "id": segment.id,
                "semantic": thaw(segment.semantic),
                "frame_ref": segment.frame_ref,
                "mount_pose": thaw(segment.mount_pose),
            }
            for segment in device.segments
        ],
        "segment_bind_poses": thaw(device.segment_bind_poses),
        "taxels": [
            {
                "id": taxel.id,
                "taxel_index": thaw(taxel.taxel_index),
                "bind_pose": thaw(taxel.bind_pose),
                "surface_attachment": thaw(taxel.surface_attachment),
                "generation_issue": thaw(taxel.generation_issue),
                "provenance": thaw(taxel.provenance),
                "input_ref": thaw(taxel.input_ref),
            }
            for taxel in device.taxels
        ],
        "skinning": {
            "indices": thaw(device.skinning.indices),
            "weights": thaw(device.skinning.weights),
            "deleted_rows": thaw(device.skinning.deleted_rows),
        },
    }


def _model_payload(model: ModelConfig | None, base: Path | None) -> dict | None:
    if model is None:
        return None

    def relative(path):
        return os.path.relpath(path, base) if base is not None else path

    return {
        "path": relative(model.path),
        "sides": model.sides,
        "packages": {key: relative(path) for key, path in model.packages.items()},
        "asset_hashes": thaw(model.asset_hashes),
        "link_adjustments": thaw(model.link_adjustments),
    }


def persistent_device(
    document: Document, device_id: str, base: Path | None = None
) -> dict:
    device = document.device(device_id)
    return {
        "version": VERSION,
        "device_id": device.id,
        "side": device.side,
        "model": _model_payload(document.model, base),
        "bind_configuration": thaw(document.bind_configuration),
        "wrist_mount": thaw(device.wrist_mount),
        "application_binding": thaw(device.application_binding),
        "draft": device_draft(device),
    }


def content_hash(document: Document, device_id: str | None = None) -> str:
    ids = (
        (device_id,)
        if device_id is not None
        else tuple(device.id for device in document.devices)
    )
    payload = [persistent_device(document, item) for item in ids]
    if not ids:
        payload = [
            _model_payload(document.model, None),
            thaw(document.bind_configuration),
        ]
    # Nonfinite in-memory edits must remain dirty even though saving rejects them.
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def metadata_payload(
    snapshot: Snapshot,
    device_id: str | None = None,
    *,
    path: str | Path | None = None,
    values: Mapping | None = None,
) -> dict:
    """Checked values enable full output; original numbers remain authoritative."""
    document = snapshot.document
    if device_id is None:
        if len(document.devices) != 1:
            raise ValueError("Select a device when saving a multi-device document")
        device_id = document.devices[0].id
    base = Path(path).expanduser().resolve().parent if path is not None else None
    editor = persistent_device(document, device_id, base)
    if values is None:
        return {"format": FORMAT, "version": VERSION, "editor": editor}
    _source_arrays(values)
    device = document.device(device_id)
    joints, _, poses, _, _ = _source_arrays(values)
    if len(joints) != len(device.segments) or len(poses) != len(device.taxels):
        raise ValueError("Compiled values do not match the snapshot's entity rows")
    # Compiler dtypes are evidence only; persistence never substitutes their rounding.
    taxels = output_taxels(device)
    payload = raw_values(device, output_order=True)
    del editor["draft"]
    editor["segment_ids"] = [segment.id for segment in device.segments]
    editor["segment_links"] = [
        segment.frame_ref[5:] if segment.frame_ref.startswith("link:") else None
        for segment in device.segments
    ]
    editor["taxel_ids"] = [taxel.id for taxel in taxels]
    editor["taxel_hints"] = [
        {
            "surface_attachment": thaw(taxel.surface_attachment),
            "generation_issue": thaw(taxel.generation_issue),
            "provenance": thaw(taxel.provenance),
            "input_ref": thaw(taxel.input_ref),
        }
        for taxel in taxels
    ]
    payload["editor"] = editor
    return payload


def save_metadata(
    snapshot: Snapshot,
    path: str | Path,
    device_id: str | None = None,
    *,
    values: Mapping | None = None,
) -> Path:
    path = Path(path).expanduser().resolve()
    payload = metadata_payload(snapshot, device_id, path=path, values=values)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def merge_documents(documents: tuple[Document, ...]) -> Document:
    """Combine separately saved sides only when their shared model agrees."""
    if not documents:
        return Document()
    first = documents[0]
    devices = []
    sides = set()
    identifiers = set()
    for document in documents:
        if (
            document.model != first.model
            or document.bind_configuration != first.bind_configuration
        ):
            raise MetadataFormatError(
                "Metadata files have conflicting shared model configuration"
            )
        for device in document.devices:
            if device.side is None or device.side in sides:
                raise MetadataFormatError(
                    "Metadata files have duplicate or unassigned sides"
                )
            sides.add(device.side)
            identity = UUID(device.id)
            if identity in identifiers:
                raise MetadataFormatError("Metadata files have duplicate device IDs")
            identifiers.add(identity)
            devices.append(device)
    return Document(first.model, tuple(devices), first.bind_configuration)

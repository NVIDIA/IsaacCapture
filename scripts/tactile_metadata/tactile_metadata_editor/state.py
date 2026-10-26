# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-owner editing transactions and bounded document history."""

from __future__ import annotations

import logging
import math
import shutil
import tempfile
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from .document import content_hash, import_values, save_metadata
from .domain import (
    Candidate,
    Command,
    Device,
    Document,
    FrozenMap,
    Issue,
    MetadataSource,
    Snapshot,
    Skinning,
    ViewState,
    thaw,
)

LOGGER = logging.getLogger("isaaccapture.tools.tactile_metadata.state")
METADATA_GROUPS = (
    "Skeleton binding poses",
    "Taxel binding poses",
    "Skinning",
)


class StaleCandidateError(ValueError):
    pass


@dataclass(frozen=True)
class _HistoryEntry:
    document: Document
    issues: tuple[Issue, ...]
    label: str


@dataclass(frozen=True)
class RecordResult:
    snapshot: Snapshot
    backup_paths: tuple[Path, ...]


def replace_device(document: Document, device: Device) -> Document:
    document.device(device.id)
    return replace(
        document,
        devices=tuple(
            device if item.id == device.id else item for item in document.devices
        ),
    )


def _index(value, limit=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    valid = value == int(value) and value >= 0
    if limit is not None:
        valid = valid and value < limit
    return int(value) if valid else None


def _permutation(items, ordered_ids):
    order = tuple(ordered_ids)
    old = {item.id: index for index, item in enumerate(items)}
    if (
        len(order) != len(items) or len(set(order)) != len(order)
        or set(order) != set(old)
    ):
        raise ValueError("Explicit reordering requires every entity exactly once")
    return tuple(old[identifier] for identifier in order)


def _move_rows(rows, order, old_count):
    occupied_end = max(
        (new + 1 for new, old in enumerate(order) if old < len(rows)), default=0
    )
    extent = max(len(rows), occupied_end)
    result = [None] * extent
    for new, old in enumerate(order):
        if old < len(rows):
            result[new] = rows[old]
    if len(rows) > old_count:
        result[old_count:] = rows[old_count:]
    return tuple(result)


class State:
    """Owned by the UI state loop; worker callbacks only enqueue commands."""

    def __init__(
        self, document: Document, *, saved: bool = False, history_limit: int = 10
    ):
        if history_limit < 10:
            raise ValueError("History must retain at least ten transactions")
        self._document = document
        self._revision = 0
        self._undo = deque(maxlen=history_limit)
        self._redo = deque(maxlen=history_limit)
        self._issues = ()
        self._saved_hashes = {}
        self.view = ViewState(
            active_device_id=document.devices[0].id if document.devices else None
        )
        if saved:
            for device in document.devices:
                self._saved_hashes[device.id] = content_hash(document, device.id)

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def issues(self) -> tuple[Issue, ...]:
        return self._issues

    @property
    def undo_count(self) -> int:
        return len(self._undo)

    @property
    def redo_count(self) -> int:
        return len(self._redo)

    def snapshot(self) -> Snapshot:
        return Snapshot(self._document, self._revision)

    def dirty(self, device_id: str | None = None) -> bool:
        if device_id is not None:
            return self._saved_hashes.get(device_id) != content_hash(
                self._document, device_id
            )
        return any(self.dirty(device.id) for device in self._document.devices)

    def mark_saved(self, snapshot: Snapshot, device_id: str):
        """Mark only the successfully written side and the exact saved content."""
        snapshot.document.device(device_id)
        self._saved_hashes[device_id] = content_hash(snapshot.document, device_id)

    def is_current(self, snapshot: Snapshot) -> bool:
        return snapshot.revision == self._revision

    def _fixed_associations(self, updated: Document):
        model = self._document.model
        if model is not None:
            if updated.model is None or updated.model.sides != model.sides:
                raise ValueError(
                    "Replacing the model or changing its sides requires a new document"
                )
            if updated.model.path != model.path:
                raise ValueError("Replacing the URDF requires a new document")
        existing = {device.id: device for device in self._document.devices}
        for device in updated.devices:
            previous = existing.get(device.id)
            if (
                previous is not None
                and previous.side is not None
                and device.side != previous.side
            ):
                raise ValueError(
                    "An assigned device side cannot change in this document"
                )

    def dispatch(self, command: Command | Candidate) -> Snapshot:
        if isinstance(command, Candidate):
            if command.base_revision != self._revision:
                raise StaleCandidateError(
                    f"Candidate revision {command.base_revision} is stale; "
                    f"current revision is {self._revision}"
                )
            updated, issues, label = command.document, command.issues, command.label
        elif isinstance(command, Command):
            updated, issues, label = command.apply(self._document), (), command.label
        else:
            raise TypeError("Expected a Command or Candidate")
        if not isinstance(updated, Document):
            raise TypeError("An editing command must return a Document")
        self._fixed_associations(updated)
        record_unchanged = (
            isinstance(command, Candidate) and command.record_unchanged
        )
        if (
            content_hash(updated) == content_hash(self._document)
            and not record_unchanged
        ):
            return self.snapshot()
        self._undo.append(_HistoryEntry(self._document, self._issues, label))
        self._redo.clear()
        self._document, self._issues = updated, tuple(issues)
        self._revision += 1
        return self.snapshot()

    def edit_device(
        self, device_id: str, update: Callable[[Device], Device], label: str = "Edit"
    ) -> Snapshot:
        return self.dispatch(
            Command(
                label,
                lambda document: replace_device(
                    document, update(document.device(device_id))
                ),
            )
        )

    def undo(self) -> Snapshot:
        if not self._undo:
            return self.snapshot()
        entry = self._undo.pop()
        self._redo.append(_HistoryEntry(self._document, self._issues, entry.label))
        self._document, self._issues = entry.document, entry.issues
        self._revision += 1
        return self.snapshot()

    def redo(self) -> Snapshot:
        if not self._redo:
            return self.snapshot()
        entry = self._redo.pop()
        self._undo.append(_HistoryEntry(self._document, self._issues, entry.label))
        self._document, self._issues = entry.document, entry.issues
        self._revision += 1
        return self.snapshot()

    def reorder_segments(self, device_id, ordered_ids):
        def update(device):
            order = _permutation(device.segments, ordered_ids)
            inverse = {old: new for new, old in enumerate(order)}
            rows = []
            for row in device.skinning.indices:
                if isinstance(row, tuple):
                    row = tuple(
                        inverse[index]
                        if (index := _index(value, len(order))) is not None else value
                        for value in row
                    )
                rows.append(row)
            return replace(
                device, segments=tuple(device.segments[index] for index in order),
                segment_bind_poses=_move_rows(
                    device.segment_bind_poses, order, len(order)
                ),
                skinning=Skinning(
                    rows, device.skinning.weights, device.skinning.deleted_rows
                ),
            )
        return self.edit_device(device_id, update, "Reorder segments")

    def delete_segments(self, device_id, entity_ids):
        deleted = set(entity_ids)

        def update(device):
            known = {item.id for item in device.segments}
            if not deleted <= known:
                raise KeyError("A selected segment no longer exists")
            retained = tuple(
                index for index, item in enumerate(device.segments)
                if item.id not in deleted
            )
            moved = {old: new for new, old in enumerate(retained)}

            def remap(value):
                index = _index(value, len(device.segments))
                if index is None:
                    return value
                if index in moved:
                    return moved[index]
                return {
                    "missing_segment_id": device.segments[index].id,
                    "original_index": value,
                }

            rows = tuple(
                tuple(remap(value) for value in row)
                if isinstance(row, tuple) else row
                for row in device.skinning.indices
            )
            poses = tuple(
                value for index, value in enumerate(device.segment_bind_poses)
                if index >= len(device.segments) or index in moved
            )
            return replace(
                device,
                segments=tuple(device.segments[index] for index in retained),
                segment_bind_poses=poses,
                skinning=Skinning(
                    rows, device.skinning.weights, device.skinning.deleted_rows
                ),
            )
        return self.edit_device(device_id, update, "Delete segments")

    def reorder_taxels(self, device_id, ordered_ids):
        def update(device):
            order = _permutation(device.taxels, ordered_ids)
            row_count = max(
                len(device.skinning.indices), len(device.skinning.weights)
            )
            used = {_index(item.taxel_index, row_count) for item in device.taxels}

            def rearrange(rows, field):
                result = []
                for old in order:
                    index = _index(device.taxels[old].taxel_index, len(rows))
                    result.append(rows[index] if index is not None else None)
                deleted = device.skinning.deleted_rows.get(field, FrozenMap())
                # Only explicit deletion provenance permits cleaning an orphan row.
                result.extend(
                    row for index, row in enumerate(rows)
                    if index not in used and str(index) not in deleted
                )
                return tuple(result)

            taxels = tuple(
                replace(device.taxels[old], taxel_index=new)
                for new, old in enumerate(order)
            )
            return replace(
                device, taxels=taxels,
                skinning=Skinning(
                    rearrange(device.skinning.indices, "indices"),
                    rearrange(device.skinning.weights, "weights"),
                ),
            )
        return self.edit_device(device_id, update, "Reorder taxel output")

    def delete_taxels(self, device_id, entity_ids):
        deleted = set(entity_ids)

        def update(device):
            if not deleted <= {item.id for item in device.taxels}:
                raise KeyError("A selected taxel no longer exists")
            indices = list(device.skinning.indices)
            weights = list(device.skinning.weights)
            provenance = thaw(device.skinning.deleted_rows)
            retained = tuple(item for item in device.taxels if item.id not in deleted)
            for item in device.taxels:
                if item.id not in deleted or any(
                    other.taxel_index == item.taxel_index for other in retained
                ):
                    continue
                for field, rows in (("indices", indices), ("weights", weights)):
                    index = _index(item.taxel_index, len(rows))
                    if index is not None:
                        provenance.setdefault(field, {})[str(index)] = item.id
            binding = device.application_binding
            if isinstance(binding, FrozenMap) and isinstance(
                binding.get("source_taxel_map"), tuple
            ):
                values = thaw(binding)
                values["source_taxel_map"] = [
                    row for row in values["source_taxel_map"]
                    if not isinstance(row, dict) or row.get("taxel_id") not in deleted
                ]
                binding = FrozenMap(values)
            return replace(
                device, taxels=retained,
                skinning=Skinning(indices, weights, provenance),
                application_binding=binding,
            )
        return self.edit_device(device_id, update, "Delete taxels")

    def _geometry_document(self, updated, model):
        from .profiles import profile_for

        configured = model.with_config(updated.model)
        devices = tuple(
            profile_for(device).restore_mounts(
                device, configured, updated.bind_configuration
            )
            for device in updated.devices
        )
        return replace(updated, devices=devices)

    def set_bind_configuration(self, configuration, model):
        before = self.snapshot()
        updated = replace(before.document, bind_configuration=FrozenMap(configuration))
        updated = self._geometry_document(updated, model)
        return self.dispatch(Candidate(
            before.revision, updated, "Set binding configuration"
        ))

    def set_wrist_mount(self, device_id, mount, model):
        before = self.snapshot()
        updated = replace_device(
            before.document,
            replace(before.document.device(device_id), wrist_mount=mount),
        )
        updated = self._geometry_document(updated, model)
        return self.dispatch(Candidate(before.revision, updated, "Set wrist reference"))

    def set_link_adjustment(self, link, local_adjustment, model):
        before = self.snapshot()
        if before.document.model is None or link not in model.links:
            raise ValueError("Select a model link")
        values = thaw(before.document.model.link_adjustments)
        values[link] = local_adjustment
        updated = replace(
            before.document,
            model=replace(before.document.model, link_adjustments=FrozenMap(values)),
        )
        updated = self._geometry_document(updated, model)
        return self.dispatch(Candidate(before.revision, updated, "Move URDF link"))

    def associate_segment(self, device_id, segment_id, link, model):
        from .profiles import profile_for

        before = self.snapshot()
        configured = model.with_config(before.document.model)
        device = before.document.device(device_id)
        device = profile_for(device).associate(
            device, segment_id, link, configured, before.document.bind_configuration
        )
        return self.dispatch(Candidate(
            before.revision, replace_device(before.document, device),
            "Associate segment link",
        ))

    def set_segment_pose(self, device_id, segment_id, pose, model=None):
        before = self.snapshot()
        device = before.document.device(device_id)
        row = next((
            index for index, item in enumerate(device.segments)
            if item.id == segment_id
        ), None)
        if row is None:
            raise KeyError(segment_id)
        poses = list(device.segment_bind_poses)
        while len(poses) <= row:
            poses.append(None)
        poses[row] = pose
        segments = tuple(
            replace(item, mount_pose=None) if item.id == segment_id else item
            for item in device.segments
        )
        updated = replace_device(
            before.document,
            replace(device, segments=segments, segment_bind_poses=poses),
        )
        if model is not None:
            updated = self._geometry_document(updated, model)
        return self.dispatch(Candidate(
            before.revision, updated, "Set segment bind pose"
        ))

    def set_taxel_pose(self, device_id, taxel_id, pose):
        def update(device):
            if not any(item.id == taxel_id for item in device.taxels):
                raise KeyError(taxel_id)
            taxels = tuple(replace(item, bind_pose=pose, surface_attachment=None,
                                   generation_issue=None, provenance="manual")
                           if item.id == taxel_id else item for item in device.taxels)
            return replace(device, taxels=taxels)
        return self.edit_device(device_id, update, "Set taxel bind pose")

    def record_metadata(
        self,
        device_id: str,
        values,
        *,
        source: MetadataSource,
        groups: tuple[str, ...] = METADATA_GROUPS,
        backup_directory: str | Path | None = None,
    ) -> RecordResult:
        groups = tuple(groups)
        if not groups or len(set(groups)) != len(groups) or any(
            group not in METADATA_GROUPS for group in groups
        ):
            raise ValueError("Select distinct metadata groups")
        before = self.snapshot()
        current = before.document.device(device_id)
        if current.side != source.side:
            raise ValueError(
                "The received source side differs from the editable device"
            )
        if not source.collection_id or not source.fingerprint:
            raise ValueError(
                "Recording requires the received collection and baseline fingerprint"
            )
        received = import_values(values, current.side).devices[0]
        fields = {}
        if "Skeleton binding poses" in groups:
            fields["segments"] = received.segments
            fields["segment_bind_poses"] = received.segment_bind_poses
        if "Skinning" in groups:
            fields["skinning"] = received.skinning
        if "Taxel binding poses" in groups:
            taxels = tuple(
                replace(
                    taxel,
                    input_ref={
                        "input_taxel_index": index,
                        "collection_id": source.collection_id,
                        "side": source.side,
                        "metadata_fingerprint": source.fingerprint,
                    },
                )
                for index, taxel in enumerate(received.taxels)
            )
            fields["taxels"] = taxels
            binding = (
                thaw(current.application_binding)
                if current.application_binding
                else {}
            )
            binding["tactile_collection_id"] = source.collection_id
            binding["source_taxel_map"] = [
                {
                    "taxel_id": taxel.id,
                    "input_taxel_index": index,
                    "collection_id": source.collection_id,
                    "side": source.side,
                    "metadata_fingerprint": source.fingerprint,
                }
                for index, taxel in enumerate(taxels)
            ]
            fields["application_binding"] = FrozenMap(binding)
        updated = replace_device(before.document, replace(current, **fields))
        candidate = Candidate(
            before.revision,
            updated,
            "Record received metadata",
            record_unchanged=True,
        )
        directory = Path(
            tempfile.mkdtemp(prefix="tactile-metadata-backup-", dir=backup_directory)
        )
        backups = []
        try:
            for index, device in enumerate(before.document.devices):
                path = directory / f"draft-{index}.metadata.json"
                save_metadata(before, path, device.id)
                backups.append(path)
            for path in backups:
                LOGGER.info("Draft backup saved to %s", path)
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        result = self.dispatch(candidate)
        return RecordResult(result, tuple(backups))

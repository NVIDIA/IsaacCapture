# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable editor records; protocol semantics belong to profiles and checks."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import UUID, uuid4


@dataclass(frozen=True, init=False, eq=False)
class FrozenMap(Mapping):
    """A recursively immutable JSON object with no mutable backing dictionary."""

    _items: tuple

    def __init__(self, values: Mapping | None = None):
        pairs = []
        for key, value in (values or {}).items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            pairs.append((key, freeze(value)))
        object.__setattr__(self, "_items", tuple(sorted(pairs)))

    def __hash__(self) -> int:
        return hash(self._items)

    def __getitem__(self, key: str) -> Any:
        for name, value in self._items:
            if name == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (name for name, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)


def freeze(value: Any) -> Any:
    if isinstance(value, FrozenMap):
        return value
    if isinstance(value, Mapping):
        return FrozenMap(value)
    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def thaw(value: Any) -> Any:
    """Return independent JSON containers, never editor-owned mutable data."""
    if isinstance(value, FrozenMap):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    return value


def new_id() -> str:
    return str(uuid4())


def _uuid(value: str) -> UUID:
    if not isinstance(value, str):
        raise TypeError("Entity identifiers must be UUID strings")
    try:
        return UUID(value)
    except ValueError as error:
        raise ValueError("Entity identifiers must be UUID strings") from error


def _unique_ids(values):
    identities = [_uuid(value) for value in values]
    if len(set(identities)) != len(identities):
        raise ValueError("Device and entity UUIDs must be unique")


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    message: str
    device_id: str | None = None
    entity_id: str | None = None
    field: str | None = None


@dataclass(frozen=True)
class Segment:
    id: str = field(default_factory=new_id)
    semantic: Any = None
    frame_ref: str = "root"
    mount_pose: Any = None

    def __post_init__(self):
        _uuid(self.id)
        if not isinstance(self.frame_ref, str):
            raise TypeError("Segment frame references must be strings")
        object.__setattr__(self, "semantic", freeze(self.semantic))
        # Only link-local A is derived here; Device.segment_bind_poses owns B.
        object.__setattr__(self, "mount_pose", freeze(self.mount_pose))


@dataclass(frozen=True)
class Taxel:
    id: str = field(default_factory=new_id)
    taxel_index: Any = None
    bind_pose: Any = None
    surface_attachment: Any = None
    generation_issue: Any = None
    provenance: Any = "manual"
    input_ref: Any = None

    def __post_init__(self):
        _uuid(self.id)
        for name in (
            "taxel_index",
            "bind_pose",
            "surface_attachment",
            "generation_issue",
            "provenance",
            "input_ref",
        ):
            object.__setattr__(self, name, freeze(getattr(self, name)))

    @property
    def position_m(self):
        pose = self.bind_pose
        return pose[:3] if isinstance(pose, tuple) and len(pose) == 7 else None

    @property
    def orientation_xyzw(self):
        pose = self.bind_pose
        return pose[3:] if isinstance(pose, tuple) and len(pose) == 7 else None


@dataclass(frozen=True)
class Skinning:
    indices: tuple = ()
    weights: tuple = ()
    deleted_rows: FrozenMap = field(default_factory=FrozenMap)

    def __post_init__(self):
        object.__setattr__(self, "indices", freeze(self.indices))
        object.__setattr__(self, "weights", freeze(self.weights))
        object.__setattr__(self, "deleted_rows", FrozenMap(self.deleted_rows))
        for field_name, rows in self.deleted_rows.items():
            if field_name not in ("indices", "weights") or not isinstance(
                rows, FrozenMap
            ):
                raise ValueError("Unknown deletion provenance field")
            for index, identifier in rows.items():
                if not index.isdecimal() or str(int(index)) != index:
                    raise ValueError(
                        "Deletion provenance requires canonical row indices"
                    )
                if int(index) >= len(getattr(self, field_name)):
                    raise ValueError("Deletion provenance references a missing raw row")
                _uuid(identifier)


@dataclass(frozen=True)
class ModelConfig:
    path: str
    sides: str
    packages: FrozenMap = field(default_factory=FrozenMap)
    asset_hashes: FrozenMap = field(default_factory=FrozenMap)
    link_adjustments: FrozenMap = field(default_factory=FrozenMap)

    def __post_init__(self):
        if not isinstance(self.path, str):
            raise TypeError("Model path must be a string")
        if self.sides not in ("left", "right", "both"):
            raise ValueError("Model sides must be left, right or both")
        for name in ("packages", "asset_hashes", "link_adjustments"):
            object.__setattr__(self, name, FrozenMap(getattr(self, name)))


@dataclass(frozen=True)
class Device:
    id: str = field(default_factory=new_id)
    side: str | None = None
    profile: FrozenMap = field(
        default_factory=lambda: FrozenMap({"type": "HandSkeletonRef"})
    )
    mapping: FrozenMap = field(default_factory=lambda: FrozenMap({"type": "LBS"}))
    segments: tuple[Segment, ...] = ()
    # Pose rows remain independent of semantic rows, including invalid counts.
    segment_bind_poses: tuple = ()
    taxels: tuple[Taxel, ...] = ()
    skinning: Skinning = field(default_factory=Skinning)
    wrist_mount: Any = None
    application_binding: Any = None

    def __post_init__(self):
        _uuid(self.id)
        if self.side is not None and not isinstance(self.side, str):
            raise TypeError("Application side identifiers must be strings")
        object.__setattr__(self, "profile", FrozenMap(self.profile))
        object.__setattr__(self, "mapping", FrozenMap(self.mapping))
        object.__setattr__(self, "segments", tuple(self.segments))
        object.__setattr__(self, "segment_bind_poses", freeze(self.segment_bind_poses))
        object.__setattr__(self, "taxels", tuple(self.taxels))
        if any(not isinstance(item, Segment) for item in self.segments) or any(
            not isinstance(item, Taxel) for item in self.taxels
        ):
            raise TypeError("Device entities must be Segment and Taxel records")
        _unique_ids((self.id, *(item.id for item in (*self.segments, *self.taxels))))
        object.__setattr__(self, "wrist_mount", freeze(self.wrist_mount))
        object.__setattr__(
            self, "application_binding", freeze(self.application_binding)
        )
        if not isinstance(self.skinning, Skinning):
            raise TypeError("Device.skinning must be Skinning")


@dataclass(frozen=True)
class Document:
    model: ModelConfig | None = None
    devices: tuple[Device, ...] = ()
    bind_configuration: FrozenMap = field(default_factory=FrozenMap)

    def __post_init__(self):
        object.__setattr__(self, "devices", tuple(self.devices))
        if any(not isinstance(item, Device) for item in self.devices):
            raise TypeError("Document devices must be Device records")
        if self.model is not None and not isinstance(self.model, ModelConfig):
            raise TypeError("Document model must be ModelConfig")
        object.__setattr__(
            self, "bind_configuration", FrozenMap(self.bind_configuration)
        )
        _unique_ids(
            identifier
            for device in self.devices
            for identifier in (
                device.id,
                *(item.id for item in (*device.segments, *device.taxels)),
            )
        )

    def device(self, device_id: str) -> Device:
        for device in self.devices:
            if device.id == device_id:
                return device
        raise KeyError(device_id)


@dataclass(frozen=True)
class Snapshot:
    document: Document
    revision: int

    def __post_init__(self):
        if not isinstance(self.document, Document) or type(self.revision) is not int:
            raise TypeError("Snapshot requires a Document and an integer revision")


@dataclass(frozen=True)
class Candidate:
    base_revision: int
    document: Document
    label: str = "Edit"
    issues: tuple[Issue, ...] = ()
    record_unchanged: bool = False

    def __post_init__(self):
        object.__setattr__(self, "issues", tuple(self.issues))


@dataclass(frozen=True)
class Command:
    label: str
    apply: Callable[[Document], Document]


@dataclass(frozen=True)
class MetadataSource:
    collection_id: str
    side: str
    fingerprint: str

    def __post_init__(self):
        if any(
            not isinstance(value, str)
            for value in (self.collection_id, self.side, self.fingerprint)
        ):
            raise TypeError("Metadata source identifiers must be strings")


@dataclass(frozen=True)
class LiveSnapshot:
    epoch: str
    seq: int
    read_monotonic_ns: int
    metadata: FrozenMap = field(default_factory=FrozenMap)
    sources: FrozenMap = field(default_factory=FrozenMap)

    def __post_init__(self):
        object.__setattr__(self, "metadata", FrozenMap(self.metadata))
        object.__setattr__(self, "sources", FrozenMap(self.sources))


@dataclass
class ViewState:
    mode: str = "EDIT"
    active_device_id: str | None = None
    selected_ids: tuple[str, ...] = ()
    q_preview: FrozenMap = field(default_factory=FrozenMap)
    metadata_source: str = "received"
    frozen_live: LiveSnapshot | None = None

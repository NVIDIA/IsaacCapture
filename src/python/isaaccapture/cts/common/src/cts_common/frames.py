# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocols for the decoded frames the shared checks read, and for frame sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, Sequence, runtime_checkable


class JointPose(Protocol):
    @property
    def position(self) -> tuple[float, float, float]: ...

    @property
    def orientation(self) -> tuple[float, float, float, float]: ...

    @property
    def is_valid(self) -> bool: ...


class Frame(Protocol):
    @property
    def sequence(self) -> int: ...

    @property
    def log_time_ns(self) -> int: ...

    @property
    def has_payload(self) -> bool: ...

    @property
    def available_time_ns(self) -> int | None: ...

    @property
    def sample_time_ns(self) -> int | None: ...

    @property
    def device_time_ns(self) -> int | None: ...

    @property
    def joints(self) -> Sequence[JointPose] | None: ...

    @property
    def has_joints(self) -> bool: ...

    def valid_joints(self) -> Iterator[tuple[int, JointPose]]: ...


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    """Container-level facts about the matched channel."""

    schema_name: str | None
    schema_encoding: str | None
    schema_data: bytes | None
    message_encoding: str | None
    topic: str | None
    profile: str | None
    channel_found: bool


@runtime_checkable
class FrameSource(Protocol):
    """Iterating must yield frames in the order they appear in the source."""

    @property
    def metadata(self) -> SourceMetadata: ...

    def __iter__(self) -> Iterator[Frame]: ...

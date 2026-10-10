# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Walks an MCAP file for the channels that declare one schema, matched by schema name.

Read in file order; make_reader() re-sorts by log time and hides out-of-order timestamps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from mcap.records import Channel, Header, Message, Schema
from mcap.stream_reader import StreamReader

from .frames import SourceMetadata


def scan(path: str | Path, schema_name: str) -> SourceMetadata:
    """Finds the first matching channel without decoding any payload."""
    profile = None
    schemas: dict[int, Schema] = {}
    with Path(path).open("rb") as handle:
        for record in StreamReader(handle).records:
            if isinstance(record, Header):
                profile = record.profile
            elif isinstance(record, Schema):
                schemas[record.id] = record
            elif isinstance(record, Channel):
                schema = schemas.get(record.schema_id)
                if schema is not None and schema.name == schema_name:
                    return SourceMetadata(
                        schema_name=schema.name,
                        schema_encoding=schema.encoding,
                        schema_data=schema.data,
                        message_encoding=record.message_encoding,
                        topic=record.topic,
                        profile=profile,
                        channel_found=True,
                    )
    return SourceMetadata(
        schema_name=None,
        schema_encoding=None,
        schema_data=None,
        message_encoding=None,
        topic=None,
        profile=profile,
        channel_found=False,
    )


def topics(path: str | Path, schema_name: str) -> tuple[str, ...]:
    """Every channel topic declaring ``schema_name``, in registration order."""
    found: list[str] = []
    schemas: dict[int, Schema] = {}
    with Path(path).open("rb") as handle:
        for record in StreamReader(handle).records:
            if isinstance(record, Schema):
                schemas[record.id] = record
            elif isinstance(record, Channel):
                schema = schemas.get(record.schema_id)
                if schema is not None and schema.name == schema_name:
                    found.append(record.topic)
    return tuple(found)


def messages(path: str | Path, schema_name: str) -> Iterator[tuple[str, Message]]:
    """``(topic, message)`` for every message on a matching channel, in file order."""
    schemas: dict[int, Schema] = {}
    channels: dict[int, str] = {}
    with Path(path).open("rb") as handle:
        for record in StreamReader(handle).records:
            if isinstance(record, Schema):
                schemas[record.id] = record
            elif isinstance(record, Channel):
                schema = schemas.get(record.schema_id)
                if schema is not None and schema.name == schema_name:
                    channels[record.id] = record.topic
            elif isinstance(record, Message) and record.channel_id in channels:
                yield channels[record.channel_id], record

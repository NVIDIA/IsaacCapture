# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Walks an MCAP file for the channels that declare one schema.

Two things here are load-bearing.

``mcap.reader.make_reader()`` re-sorts messages by log time, which silently repairs a
recording with non-monotonic timestamps. ``StreamReader`` walks raw records in file
order, the way the C++ ``LinearMessageView`` does, so a reordered file stays reordered.

Channels are located by declared schema name. The topic is ``<source name>/<sub-channel>``
where the prefix is whatever ``name=`` the recording script passed, so it is not intrinsic
to the format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from mcap.records import Channel, Header, Message, Schema
from mcap.stream_reader import StreamReader

from .frames import SourceMetadata


def scan(path: str | Path, schema_name: str) -> SourceMetadata:
    """Finds the first matching channel without decoding any payload.

    A channel that is registered but carries no messages is a real case: the Pico
    tracker registers its channels at construction, then returns early from
    ``update()`` in limp mode and never publishes. That must read as insufficient
    data, not as a pass.
    """
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

# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The LiveKit streams a station offers, read from the ``--livekit-streams`` YAML file.

Only locates the video (server URL, room, publisher identity and track names). The server never joins the room:
each consumer connects to LiveKit itself and learns the live details there.

.. code-block:: yaml

    livekit:
      url: wss://LIVEKIT_HOST
      room: sensing-preview
      publisher: orin-sensing
      timestamp_basis: processing_unix_us   # what the publisher puts in the frame timestamp (see protocol.py)
      streams:
        - {name: Head left, track: sensing-left, width: 2560, height: 1984, fps: 30}
"""

from __future__ import annotations

from pathlib import Path

import yaml

from . import protocol as p

_ROOM_KEYS = {"url", "room", "publisher", "timestamp_basis", "streams"}
_BASES = {
    "monotonic_us": p.LiveKitTimestampBasis.PUBLISHER_MONOTONIC_US,
    "processing_unix_us": p.LiveKitTimestampBasis.PUBLISHER_PROCESSING_UNIX_US,
    "capture_unix_us": p.LiveKitTimestampBasis.SENSOR_CAPTURE_UNIX_US,
    "capture_monotonic_us": p.LiveKitTimestampBasis.SENSOR_CAPTURE_MONOTONIC_US,
}
_STREAM_KEYS = {"name", "track", "width", "height", "fps"}


def _mapping(value: object, where: str, allowed: set[str]) -> dict:
    """A YAML mapping whose keys are all in ``allowed``; a typo is an error, not a silently ignored setting."""
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{where} has unknown key(s): {sorted(unknown)}")
    return value


def load_livekit_streams(path: Path) -> tuple[p.LiveKitCameraStream, ...]:
    """The streams named by the YAML file at ``path``; raises ``ValueError`` for anything unusable."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read LiveKit streams file {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"LiveKit streams file {path} is not valid YAML: {exc}") from exc
    room = _mapping(_mapping(data, "the file", {"livekit"}).get("livekit"), "livekit", _ROOM_KEYS)
    raw_streams = room.get("streams")
    if not isinstance(raw_streams, list):
        raise ValueError("livekit.streams must be a list")
    basis_name = room.get("timestamp_basis", "monotonic_us")
    # A YAML list or mapping is unhashable, so check the type before the lookup.
    if not isinstance(basis_name, str) or basis_name not in _BASES:
        raise ValueError(f"livekit.timestamp_basis must be one of {sorted(_BASES)}")
    streams: list[p.LiveKitCameraStream] = []
    try:
        for index, raw in enumerate(raw_streams):
            entry = _mapping(raw, f"livekit.streams[{index}]", _STREAM_KEYS)
            # The wire parser is the one validator: a file that loads is a list every client will accept.
            streams.append(
                p.LiveKitCameraStream.from_json(
                    {
                        "stream_id": p.LIVEKIT_STREAM_ID_BASE + index,
                        "kind": int(p.StreamKind.LIVEKIT_IMAGE),
                        "name": entry.get("name"),
                        "url": room.get("url"),
                        "room": room.get("room"),
                        "publisher": room.get("publisher"),
                        "track": entry.get("track"),
                        "timestamp_basis": int(_BASES[basis_name]),
                        "width": entry.get("width"),
                        "height": entry.get("height"),
                        "fps": entry.get("fps"),
                    }
                )
            )
        p.check_livekit_streams(streams)
    except p.ProtocolError as exc:
        raise ValueError(f"LiveKit streams file {path}: {exc}") from exc
    return tuple(streams)

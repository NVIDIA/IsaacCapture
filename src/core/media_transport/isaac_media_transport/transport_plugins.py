# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Startup-only plugin discovery; listing never imports transport SDKs."""

from dataclasses import dataclass
from importlib import metadata
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .transport import TransportAdapter

ENTRY_POINT_GROUP = "isaac_media_bridge.transports.v1"
API_VERSION = 1


@dataclass(frozen=True)
class TransportPlugin:
    """An installed entry point exports this descriptor, not an SDK session."""

    name: str
    api_version: int
    create: Callable[[], "TransportAdapter"]


def _entry_points():
    entries = {}
    for entry in metadata.entry_points(group=ENTRY_POINT_GROUP):
        if entry.name in entries:
            raise ValueError(f"Duplicate transport plugin registration: {entry.name!r}")
        entries[entry.name] = entry
    return entries


def available_transports() -> tuple[str, ...]:
    """Registered names, not a claim that native dependencies are operational."""
    return tuple(sorted(set(_entry_points())))


def load_transport(name: str) -> "TransportAdapter":
    entries = _entry_points()
    if not isinstance(name, str) or name not in entries:
        raise ValueError(
            f"Transport {name!r} is not implemented or installed; available: "
            + ", ".join(sorted(set(entries)))
        )
    try:
        plugin = entries[name].load()
        if not isinstance(plugin, TransportPlugin):
            raise ValueError(
                f"Transport {name!r} must export a TransportPlugin descriptor"
            )
        if plugin.name != name or plugin.api_version != API_VERSION:
            raise ValueError(f"Transport {name!r} has incompatible name or API version")
        if not callable(plugin.create):
            raise ValueError(f"Transport {name!r} has no callable factory")
        adapter = plugin.create()
    except (ImportError, OSError) as exc:
        raise ValueError(
            f"Cannot load transport {name!r}: {exc}. Install its SDK and "
            "architecture-matched native libraries in this Python environment."
        ) from exc
    if getattr(getattr(adapter, "capabilities", None), "name", None) != name:
        raise ValueError(
            f"Transport {name!r} returned an adapter with mismatched capabilities"
        )
    for method in ("messages", "video_sender", "video_receiver", "provisioner"):
        if not callable(getattr(adapter, method, None)):
            raise ValueError(f"Transport {name!r} is missing adapter method {method}")
    return adapter


if __name__ == "__main__":
    print("Registered transports (runtime dependencies not checked):")
    for name in available_transports():
        print(f"  {name}")

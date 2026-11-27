# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in, low-rate numeric WebRTC statistics; no credentials or payloads."""

import asyncio
import time

from google.protobuf.json_format import MessageToDict


def numeric_fields(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(numeric_fields(item, name))
        elif type(item) in (int, float, bool):
            result[name] = item
        elif isinstance(item, str) and item.replace("-", "", 1).isdigit():
            result[name] = int(item)
    return result


async def sample_stats(track, name, recorder):
    """Keep slow RPCs and serialization off the per-frame capture path."""
    while True:
        started = time.monotonic_ns()
        try:
            async with asyncio.timeout(2):
                stats = await track.get_stats()
            for index, stat in enumerate(stats):
                fields = numeric_fields(
                    MessageToDict(stat, preserving_proto_field_name=True)
                )
                entries = list(fields.items())
                for offset in range(0, len(entries), 12):
                    recorder.record(
                        "sdk_stats",
                        name,
                        0,
                        details={
                            "report": index,
                            **dict(entries[offset : offset + 12]),
                        },
                    )
            recorder.record(
                "sdk_stats_poll",
                name,
                0,
                details={
                    "elapsed_ms": (time.monotonic_ns() - started) / 1e6,
                    "reports": len(stats),
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Diagnostics must report unavailable statistics without stopping media.
            recorder.record(
                "sdk_stats_error", name, 0, details={"error": str(exc)[:240]}
            )
        await asyncio.sleep(2)

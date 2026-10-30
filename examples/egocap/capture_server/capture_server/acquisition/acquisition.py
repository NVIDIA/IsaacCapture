# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""``Acquisition``: the one concrete object ``_Session`` holds for its
acquisition source -- composing one optional :class:`IsaacCaptureAdapter`
(``None`` for a camera-only request, which must never construct an IC
session) with zero or more :class:`DepthAIAdapter`, one per configured
camera stream.

This is not a generic protocol/interface: it is one
fixed, concretely named composition that always holds exactly these two
concrete adapter types, mode-agnostic by construction since both adapters
already are. ``_Session`` calls it directly --
``enter``/``next_tracking_sample``/``next_camera_jpeg``/``status``/``close``.
``next_tracking_sample``/``next_camera_jpeg`` take no
``t``/``tick`` argument: each adapter paces and times itself -- see their
own docstrings.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from capture_server import protocol as p

from .depthai import DepthAIAdapter
from .errors import CameraSourceError
from .isaac_capture import IsaacCaptureAdapter

__all__ = ["Acquisition"]


@dataclass
class Acquisition:
    tracking: IsaacCaptureAdapter | None
    cameras: tuple[DepthAIAdapter, ...]

    async def enter(self) -> None:
        """A REQUIRED source's startup failure is fatal and propagates
        (IsaacCapture is always implicitly required when present -- a
        camera-only session never constructs it in the first place).
        An OPTIONAL camera's startup failure is an honest degraded/error
        observation, not fatal: other sources keep starting normally, and
        the camera's own adapter already recorded itself as degraded
        (DepthAIAdapter.camera_status) rather than falsely claiming
        CONNECTED -- a required camera's terminal failure instead ends
        the whole capture with StopReason.CAMERA_ERROR, per the
        station-chosen, server-selected `required` acquisition policy
        for that camera."""
        if self.tracking is not None:
            await self.tracking.enter()
        for camera in self.cameras:
            try:
                await camera.enter()
            except CameraSourceError:
                if camera.required:
                    raise

    async def next_tracking_sample(self) -> tuple[int, dict]:
        assert self.tracking is not None
        return await self.tracking.next_tracking_sample()

    async def next_camera_jpeg(self, stream_id: int) -> tuple[int, bytes] | None:
        for camera in self.cameras:
            if camera.stream_id == stream_id:
                return await camera.next_camera_jpeg()
        raise KeyError(f"no configured camera for stream_id {stream_id}")

    async def status(self, updated_at_ns: int) -> p.StatusSnapshot:
        camera_statuses = tuple(camera.camera_status() for camera in self.cameras)
        if self.tracking is None:
            # A camera-only attempt never entered IC/OpenXR -- native
            # providers/devices are empty outright, not a filtered
            # subset.
            return p.StatusSnapshot(updated_at_ns=updated_at_ns, cameras=camera_statuses)
        snapshot = await self.tracking.status(updated_at_ns)
        return p.StatusSnapshot(
            updated_at_ns=updated_at_ns,
            providers=snapshot.providers,
            devices=snapshot.devices,
            cameras=camera_statuses,
        )

    async def close(self) -> bool:
        """Closes every adapter CONCURRENTLY, each on its own dedicated
        worker thread (native thread affinity is per-adapter already --
        see ThreadRunner -- so running them concurrently never puts two
        threads on the same native resource). Up to four independent
        adapters (one IC + up to three cameras) each bound to their own
        release timeout would, awaited sequentially, sum to up to four
        times that bound -- comfortably exceeding the outer release-wait
        budget server.py's own finalization assumes for ONE source
        release. Running them concurrently keeps the total wall-clock
        cost to whichever single adapter is slowest, not their sum.
        Every adapter's close is still attempted regardless of whether
        another raised or timed out -- one release failure must never
        skip releasing the others, and a failure to confirm release
        within bound is reported (``ok=False``), never silently assumed
        successful."""
        closers = ([self.tracking.close()] if self.tracking is not None else []) + [
            camera.close() for camera in self.cameras
        ]
        if not closers:
            return True
        results = await asyncio.gather(*closers, return_exceptions=True)
        first_exc = next((r for r in results if isinstance(r, BaseException)), None)
        if first_exc is not None:
            raise first_exc
        return all(results)

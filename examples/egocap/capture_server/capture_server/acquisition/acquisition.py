# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""``Acquisition``: the one concrete object ``_Session`` holds for its acquisition source.

It composes one optional :class:`IsaacCaptureAdapter` (``None`` for a camera-only request, which must never
construct an IC session), zero or more camera adapters (:class:`DepthAIAdapter` for OAK-D,
:class:`SensingAdapter` for SENSING), and the optional :class:`SensingRig` -- the one native producer process
the SENSING adapters read from. The rig is owned here and only borrowed by its adapters, so it is started
before them and reaped after they have settled.

This is NOT a generic source interface: it is one fixed, concretely named composition. ``_Session`` calls it
directly -- ``enter``/``next_tracking_sample``/``next_camera_jpeg``/``status``/``close`` -- and each adapter
paces and times itself (see their own docstrings).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from capture_server import protocol as p

from .depthai import DepthAIAdapter
from .errors import CameraSourceError
from .isaac_capture import IsaacCaptureAdapter
from .sensing import SensingAdapter, SensingRig

__all__ = ["Acquisition"]


def _observe_close(task: "asyncio.Future") -> "asyncio.Future":
    """Register an observer on a close task as soon as it exists, so a late failure is never left unretrieved.

    Observation only: it does not own, cancel or await the task.
    """
    task.add_done_callback(lambda f: f.cancelled() or f.exception())
    return task


# The server's whole source-release budget for one session; adapters and the SENSING producer share it.
_CLOSE_BUDGET_S = 5.0


@dataclass
class Acquisition:
    tracking: IsaacCaptureAdapter | None
    cameras: tuple[DepthAIAdapter | SensingAdapter, ...]
    # The one native SENSING producer shared by the SENSING adapters in ``cameras``; owned here, borrowed by them.
    sensing_rig: SensingRig | None = None

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
        if self.sensing_rig is not None:
            # Never raises: a start failure is recorded on the rig and each SENSING adapter reports it under
            # its own required/optional policy below, so optional streams degrade while OAK/tracking go on.
            await self.sensing_rig.start()
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
        """Releases every adapter and the SENSING producer, concurrently, each on its own bound.

        Every adapter is closed on its own dedicated worker thread (native thread affinity is per-adapter -- see
        ThreadRunner). The tracking and OAK adapters each keep their own bound. The SENSING stream adapters and
        the producer they borrow form ONE chain (:meth:`_close_rig_chain`): the adapters settle first, then the
        producer is terminated and reaped, so a worker still reading never races its teardown. The chain runs
        concurrently with the other closes and is sized to fit the same ``_CLOSE_BUDGET_S``, so a slow OAK or
        tracking release is not cut short by the producer's shutdown, nor the reverse.
        Waiting is bounded by ``_CLOSE_BUDGET_S``; a timeout does not cancel any close job -- it keeps running and
        is only reported as unconfirmed (False). Every close is attempted; one failure never skips another. A
        False means a worker MAY still be running: the wait bounds how long we waited, it does not prove native
        settlement. A cancelled caller does not prove release either: every job keeps running, observed, so the
        producer is still reaped.
        """
        borrowing = [c for c in self.cameras if getattr(c, "borrows_rig", False)]
        independent = ([self.tracking] if self.tracking is not None else []) + [
            c for c in self.cameras if not getattr(c, "borrows_rig", False)
        ]
        tasks = [_observe_close(asyncio.ensure_future(a.close())) for a in independent]
        if borrowing or self.sensing_rig is not None:
            tasks.append(_observe_close(asyncio.ensure_future(self._close_rig_chain(borrowing))))
        pending: set[asyncio.Future] = set()
        if tasks:
            _done, pending = await asyncio.wait(tasks, timeout=_CLOSE_BUDGET_S)
        results: list[bool | BaseException] = []
        for task in tasks:
            if task in pending or task.cancelled():
                results.append(False)  # unconfirmed: may still be running
            elif (exc := task.exception()) is not None:
                results.append(exc)
            else:
                results.append(task.result())
        if (first_exc := next((r for r in results if isinstance(r, BaseException)), None)) is not None:
            raise first_exc
        return all(results)

    async def _close_rig_chain(self, adapters: list) -> bool:
        """Close the adapters that borrow the producer, then terminate and reap it.

        The producer is reaped whether or not an adapter failed to close. True only if every adapter and the
        producer confirmed release; the first failure is raised after both have been attempted.
        """
        results = await asyncio.gather(*(a.close() for a in adapters), return_exceptions=True)
        rig_result: bool | BaseException = True
        if self.sensing_rig is not None:
            try:
                rig_result = await self.sensing_rig.close()
            except Exception as exc:  # noqa: BLE001 -- reported below, after the adapters' own results
                rig_result = exc
        exceptions = [r for r in (*results, rig_result) if isinstance(r, BaseException)]
        if exceptions:
            # A cancellation takes priority over any other exception collected here: dropping it in favor of
            # an unrelated close failure would silently defeat the caller's own cancellation.
            cancelled = next((e for e in exceptions if isinstance(e, asyncio.CancelledError)), None)
            raise cancelled if cancelled is not None else exceptions[0]
        return all(results) and rig_result is True

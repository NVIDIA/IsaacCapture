# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The production :class:`~.feed.SlotMapper`: opens the publisher's CUDA IPC handles with CuPy.

Only memory mapping and device-to-host copies are used -- no CuPy kernels -- so this works on a GPU
the installed CuPy wheel carries no kernel image for (as long as the driver accepts the handles),
which is also exactly what the publishing receiver does on its side. CuPy is imported lazily so the
rest of the package, and its tests, never need a GPU runtime.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

from .feed import FeedError, StreamInfo
from .slot_guard import read_slot

__all__ = ["CudaSlotMapper", "CudaSlotReader"]


class CudaSlotReader:
    """Mapped CUDA IPC slots of one stream; ``read`` copies a slot to host memory under the slot's read lock."""

    def __init__(self, stream: StreamInfo, cupy: Any) -> None:
        self._cp = cupy
        self._gpu_id = stream.gpu_id
        self._ptrs: list[int] = []
        self._arrays: list[Any] = []
        # One guard file per slot with a version 3 publisher; None for a version 2 one, which has no locks.
        self._guards: list[str | None] = []
        with cupy.cuda.Device(stream.gpu_id):
            for slot in stream.slots:
                guard = slot.get("guard_path")
                self._guards.append(str(guard) if guard else None)
                handle = base64.b64decode(slot["handle"])
                # Lazy peer access matches the receiver's own consumer example; the handle is owned by the
                # publisher process and must be closed with ipcCloseMemHandle, never freed.
                ptr = int(
                    cupy.cuda.runtime.ipcOpenMemHandle(handle, cupy.cuda.runtime.cudaIpcMemLazyEnablePeerAccess)
                )
                self._ptrs.append(ptr)
                memory = cupy.cuda.UnownedMemory(ptr, int(slot["nbytes"]), self)
                self._arrays.append(
                    cupy.ndarray(
                        tuple(int(v) for v in slot["shape"]),
                        dtype=cupy.uint8,
                        memptr=cupy.cuda.MemoryPointer(memory, 0),
                        strides=tuple(int(v) for v in slot["strides"]),
                    )
                )

    def read(self, slot: int, generation: int) -> np.ndarray | None:
        """Device-to-host copy of ``slot`` as a uint8 array with the slot's shape, or None if it is stale.

        With a version 3 publisher the copy runs while holding the slot's shared lock, so the publisher cannot
        overwrite it mid-copy, and only if the slot still holds ``generation``. ``asnumpy`` blocks until the
        copy has finished, so the lock is never released while the GPU is still reading.
        """
        guard = self._guards[slot]
        with self._cp.cuda.Device(self._gpu_id):
            if guard is None:
                # Version 2 publisher: no lock exists, the caller guards against torn frames itself.
                return self._cp.asnumpy(self._arrays[slot])
            with read_slot(guard, generation) as current:
                if not current:
                    return None
                return self._cp.asnumpy(self._arrays[slot])

    def close(self) -> None:
        """Close every opened IPC handle; idempotent."""
        arrays, ptrs = self._arrays, self._ptrs
        self._arrays, self._ptrs = [], []
        # Drop the array views first so no CuPy object still references memory we are about to close.
        del arrays
        with self._cp.cuda.Device(self._gpu_id):
            for ptr in ptrs:
                try:
                    self._cp.cuda.runtime.ipcCloseMemHandle(ptr)
                except Exception:  # noqa: BLE001 -- closing a handle twice or after the publisher died is harmless
                    pass


class CudaSlotMapper:
    """Maps subscribed streams with CuPy; the import happens on first use with a clear install hint."""

    def open(self, stream: StreamInfo) -> CudaSlotReader:
        """Map ``stream``'s slots; a missing CuPy or an unopenable handle is reported as a feed failure."""
        try:
            import cupy  # noqa: PLC0415 -- optional dependency, loaded only on the GMSL path
        except ImportError as exc:
            raise FeedError("reading the GMSL frame feed needs CuPy: install the 'gmsl' extra") from exc
        try:
            return CudaSlotReader(stream, cupy)
        except Exception as exc:  # noqa: BLE001 -- a CUDA failure is a camera failure, mapped uniformly
            raise FeedError(f"cannot map stream {stream.name!r} from the frame feed: {exc}") from exc

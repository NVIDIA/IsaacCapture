# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal ctypes binding for the CUDA virtual-memory-management API.

Only what importing a shared device allocation needs. ``libcuda.so.1`` ships
with the driver, so this keeps the ``cuda_ipc`` source free of any new Python
dependency — CuPy alone is not enough, as it exposes no ``cuMemImport*``.

Legacy CUDA IPC (``cudaIpcOpenMemHandle``) is not an option here: on Tegra it
fails with ``cudaErrorInvalidValue`` even though the producer's
``cudaIpcGetMemHandle`` succeeded. The VMM path is the one that works.
"""

from __future__ import annotations

import ctypes

# CUmemAllocationHandleType
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
# CUmemLocationType
CU_MEM_LOCATION_TYPE_DEVICE = 1
# CUmemAccess_flags
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _CUmemLocation), ("flags", ctypes.c_int)]


class CudaDriverError(RuntimeError):
    pass


class CudaDriver:
    """Lazily-loaded handle onto the few driver entry points we need."""

    def __init__(self) -> None:
        try:
            self._lib = ctypes.CDLL("libcuda.so.1")
        except OSError as e:
            raise CudaDriverError(
                "libcuda.so.1 not found — the NVIDIA driver is not installed "
                "or not visible in this container."
            ) from e

        # cuMemImportFromShareableHandle takes the fd by value in a void*, not
        # a pointer to it; passing &fd yields CUDA_ERROR_INVALID_VALUE.
        self._lib.cuMemImportFromShareableHandle.argtypes = [
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self._lib.cuMemAddressReserve.argtypes = [
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_ulonglong,
            ctypes.c_ulonglong,
        ]
        self._lib.cuMemMap.argtypes = [
            ctypes.c_ulonglong,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_ulonglong,
            ctypes.c_ulonglong,
        ]
        self._lib.cuMemSetAccess.argtypes = [
            ctypes.c_ulonglong,
            ctypes.c_size_t,
            ctypes.POINTER(CUmemAccessDesc),
            ctypes.c_size_t,
        ]
        self._lib.cuMemUnmap.argtypes = [ctypes.c_ulonglong, ctypes.c_size_t]
        self._lib.cuMemAddressFree.argtypes = [ctypes.c_ulonglong, ctypes.c_size_t]
        self._lib.cuMemRelease.argtypes = [ctypes.c_ulonglong]
        self._lib.cuDevicePrimaryCtxRetain.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
        ]
        self._lib.cuDevicePrimaryCtxRelease.argtypes = [ctypes.c_int]
        self._lib.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        self._lib.cuInit.argtypes = [ctypes.c_uint]
        self._lib.cuGetErrorName.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]

        self._check(self._lib.cuInit(0), "cuInit")

    def _check(self, result: int, what: str) -> None:
        if result == 0:
            return
        name = ctypes.c_char_p()
        self._lib.cuGetErrorName(result, ctypes.byref(name))
        detail = name.value.decode() if name.value else f"code {result}"
        raise CudaDriverError(f"{what} failed: {detail}")

    # ── context ───────────────────────────────────────────────────────

    def primary_ctx_retain(self, device_id: int) -> int:
        ctx = ctypes.c_void_p()
        self._check(
            self._lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), device_id),
            "cuDevicePrimaryCtxRetain",
        )
        return ctx.value or 0

    def primary_ctx_release(self, device_id: int) -> None:
        self._lib.cuDevicePrimaryCtxRelease(device_id)

    def ctx_set_current(self, ctx: int) -> None:
        self._check(self._lib.cuCtxSetCurrent(ctypes.c_void_p(ctx)), "cuCtxSetCurrent")

    # ── shared allocation ─────────────────────────────────────────────

    def import_fd(self, fd: int) -> int:
        """Import a POSIX fd exported by ``cuMemExportToShareableHandle``."""
        handle = ctypes.c_ulonglong()
        self._check(
            self._lib.cuMemImportFromShareableHandle(
                ctypes.byref(handle),
                ctypes.c_void_p(fd),
                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            ),
            "cuMemImportFromShareableHandle",
        )
        return handle.value

    def map_readwrite(self, handle: int, size: int, device_id: int) -> int:
        """Reserve VA, map the handle into it, and grant this device access.

        Returns the device pointer. On failure everything already acquired is
        rolled back, so the caller never has to unwind a partial mapping.
        """
        ptr = ctypes.c_ulonglong()
        self._check(
            self._lib.cuMemAddressReserve(ctypes.byref(ptr), size, 0, 0, 0),
            "cuMemAddressReserve",
        )
        try:
            self._check(self._lib.cuMemMap(ptr, size, 0, handle, 0), "cuMemMap")
        except CudaDriverError:
            self._lib.cuMemAddressFree(ptr, size)
            raise

        desc = CUmemAccessDesc()
        desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        desc.location.id = device_id
        desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        try:
            self._check(
                self._lib.cuMemSetAccess(ptr, size, ctypes.byref(desc), 1),
                "cuMemSetAccess",
            )
        except CudaDriverError:
            self._lib.cuMemUnmap(ptr, size)
            self._lib.cuMemAddressFree(ptr, size)
            raise

        return ptr.value

    def unmap(self, ptr: int, size: int, handle: int) -> list:
        """Tear down a ``map_readwrite`` mapping. Idempotent.

        Returns the name of each step that failed rather than raising, so a
        caller unwinding after an error still completes the rest. Silence here
        would turn a failed release into a slow leak, so callers should log
        whatever comes back.
        """
        failures = []
        if ptr:
            for name, fn in (
                ("cuMemUnmap", self._lib.cuMemUnmap),
                ("cuMemAddressFree", self._lib.cuMemAddressFree),
            ):
                if fn(ptr, size) != 0:
                    failures.append(name)
        if handle and self._lib.cuMemRelease(handle) != 0:
            failures.append("cuMemRelease")
        return failures


_driver: CudaDriver | None = None


def driver() -> CudaDriver:
    """Process-wide driver handle; ``cuInit`` runs once."""
    global _driver
    if _driver is None:
        _driver = CudaDriver()
    return _driver

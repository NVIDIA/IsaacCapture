# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public OVRTX mappings and bounded application-owned CUDA snapshots."""

from __future__ import annotations

import ctypes
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def _api():
    from ovrtx._src import bindings as b

    root = b.ovrtx_loaded_bin_dir()
    if root is None:
        raise RuntimeError("OVRTX loaded provider is unavailable")
    lib = ctypes.CDLL(str(Path(root) / b.OVRTX_LOADER_LIB_NAME))
    lib.ovrtx_map_render_var_output.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.POINTER(b.ovrtx_map_output_description_t),
        b.ovrtx_timeout_t,
        ctypes.POINTER(b.ovrtx_render_var_output_t),
    ]
    lib.ovrtx_map_render_var_output.restype = b.ovrtx_result_t
    lib.ovrtx_unmap_render_var_output.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        b.ovrtx_cuda_sync_t,
    ]
    lib.ovrtx_unmap_render_var_output.restype = b.ovrtx_result_t
    lib.ovrtx_fetch_results.argtypes = [
        ctypes.c_void_p,
        b.ovrtx_step_result_handle_t,
        b.ovrtx_timeout_t,
        ctypes.POINTER(b.ovrtx_render_product_set_outputs_t),
    ]
    lib.ovrtx_fetch_results.restype = b.ovrtx_result_t
    return b, lib


def native_renderer_handle(renderer) -> int:
    """Normalize the borrowed renderer object before calling the OVRTX C API."""
    # Accept Renderer objects and raw native pointers at this boundary.
    handle = getattr(renderer, "handle", getattr(renderer, "_handle", renderer))
    value = ctypes.cast(handle, ctypes.c_void_p).value
    if not value:
        raise ValueError("borrowed OVRTX renderer has no native handle")
    return value


def completed_outputs(renderer: int, step_result: int):
    """Borrow completed output metadata until OVXR consumes the bound plan."""
    b, lib = _api()
    outputs = b.ovrtx_render_product_set_outputs_t()
    result = lib.ovrtx_fetch_results(
        renderer, step_result, b.OVRTX_TIMEOUT_INFINITE, ctypes.byref(outputs)
    )
    if result.status != b.OVRTX_API_SUCCESS:
        raise RuntimeError("OVRTX completed output fetch failed")
    return ctypes.pointer(outputs)


def _check_cuda(result):
    if int(result[0]):
        raise RuntimeError(f"GPU output synchronization failed: CUDA {result[0]}")
    return result[1] if len(result) > 1 else None


def _wait_ready(cuda, sync) -> None:
    # Both fields are independently valid in the public OVRTX contract.
    if sync.wait_event:
        _check_cuda(cuda.cudaEventSynchronize(sync.wait_event))
    if sync.stream:
        stream = cuda.cudaStreamLegacy if sync.stream == 1 else sync.stream
        _check_cuda(cuda.cudaStreamSynchronize(stream))


class OutputMapCleanupError(RuntimeError):
    """The caller must retain this map and stop admission until owner cleanup."""

    def __init__(self, image, cause, cleanup):
        super().__init__(f"Output map failed: {cause}; cleanup failed: {cleanup}")
        self.image = image


@dataclass(kw_only=True)
class MappedOutput:
    """One native output map retained until its owner proves unmap succeeded."""

    renderer: int
    mapped: Any
    owner: int
    closed: bool = False
    busy: bool = False

    def close(self) -> None:
        """Owner only, after every consumer has completed its last source read."""
        if threading.get_ident() != self.owner:
            raise RuntimeError("Output must be unmapped on its renderer owner")
        if self.busy:
            raise RuntimeError("Output consumer completion is unproved")
        if not self.closed:
            b, lib = _api()
            result = lib.ovrtx_unmap_render_var_output(
                self.renderer, self.mapped.map_handle, b.ovrtx_cuda_sync_t()
            )
            if result.status != b.OVRTX_API_SUCCESS:
                raise RuntimeError("OVRTX output unmap failed")
            self.closed = True


@dataclass
class MappedGpuImage(MappedOutput):
    width: int
    height: int
    pointer: int
    pitch: int
    device: int
    is_array: bool = False


@dataclass
class OwnedGpuImage:
    """One reusable CUDA allocation, leased until the UI's source read completes."""

    width: int
    height: int
    pointer: int
    pitch: int
    device: int
    owner: int
    closed: bool = True

    def close(self) -> None:
        if threading.get_ident() != self.owner:
            raise RuntimeError("GPU image must be retired on its allocation owner")
        self.closed = True


class OwnedCudaImagePool:
    """At most two mirror copies; no borrowed OVRTX map escapes a frame."""

    def __init__(self) -> None:
        self._images: list[OwnedGpuImage] = []
        self._source: MappedGpuImage | None = None
        self._device: int | None = None

    def configure_opengl_destination(self) -> None:
        """UI owner, after creating its GL context and before any GPU upload."""
        from cuda.bindings import runtime as cuda

        result = cuda.cudaGLGetDevices(16, cuda.cudaGLDeviceList.cudaGLDeviceListAll)
        try:
            _check_cuda(result)
        except RuntimeError as error:
            raise RuntimeError(
                "Mirror needs an OpenGL-associated CUDA device; start the process with "
                "OMNIUI_BACKEND=opengl before any OVUI initialization"
            ) from error
        if result[1] != 1:
            raise RuntimeError("mirror needs one unambiguous OpenGL CUDA device")
        self.set_destination_device(int(result[2][0]))

    def set_destination_device(self, device: int) -> None:
        if device < 0 or self._device not in (None, device):
            raise RuntimeError("mirror CUDA destination cannot change while live")
        self._device = device

    def __call__(
        self, renderer: int, handle: int, max_bytes: int
    ) -> OwnedGpuImage | None:
        destination = self._device
        if destination is None:
            return None  # GUI startup is an expected optional-output omission.
        self.finish_frame()
        from cuda.bindings import runtime as cuda

        try:
            source = map_cuda_array_rgba(renderer, handle, max_bytes)
        except OutputMapCleanupError as error:
            self._source = error.image
            raise
        self._source = source
        try:
            image = next((item for item in self._images if item.closed), None)
            if image is not None and (image.width, image.height, image.device) != (
                source.width,
                source.height,
                destination,
            ):
                _check_cuda(cuda.cudaSetDevice(image.device))
                _check_cuda(cuda.cudaFree(image.pointer))
                self._images.remove(image)
                image = None
            _check_cuda(cuda.cudaSetDevice(destination))
            if image is None:
                if len(self._images) >= 2:
                    raise RuntimeError("both owned mirror images are in use")
                result = cuda.cudaMallocPitch(source.width * 4, source.height)
                _check_cuda(result)
                image = OwnedGpuImage(
                    source.width,
                    source.height,
                    int(result[1]),
                    int(result[2]),
                    destination,
                    threading.get_ident(),
                )
                self._images.append(image)
            # Check reused allocations too: CUDA may pad a new allocation past
            # the limit, and a rejected allocation remains owned until cleanup.
            if image.pitch * image.height > max_bytes:
                raise RuntimeError("owned mirror allocation exceeds the byte bound")
            # The source map and the destination have separate lifetimes. Prove
            # the read complete before unmapping even when enqueue reports error.
            source.busy = True
            if source.device == destination:
                _check_cuda(
                    cuda.cudaMemcpy2DFromArrayAsync(
                        image.pointer,
                        image.pitch,
                        source.pointer,
                        0,
                        0,
                        source.width * 4,
                        source.height,
                        cuda.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                        0,
                    )
                )
            else:
                # Keep OVUI's GL registration on its original CUDA device even
                # when OVRTX assigns a selected eye/camera to another GPU.
                copy = cuda.cudaMemcpy3DPeerParms()
                copy.srcArray, copy.srcDevice = source.pointer, source.device
                copy.dstPtr.ptr, copy.dstPtr.pitch = image.pointer, image.pitch
                copy.dstPtr.xsize, copy.dstPtr.ysize = source.width * 4, source.height
                copy.dstDevice = destination
                # An array endpoint makes width RGBA elements, not bytes.
                copy.extent.width, copy.extent.height, copy.extent.depth = (
                    source.width,
                    source.height,
                    1,
                )
                _check_cuda(cuda.cudaMemcpy3DPeerAsync(copy, 0))
        finally:
            _check_cuda(cuda.cudaSetDevice(destination))
            if source.busy:
                _check_cuda(cuda.cudaStreamSynchronize(0))
                source.busy = False
            source.close()
            self._source = None
        image.closed = False
        return image

    def finish_frame(self) -> None:
        """Fail closed before submit/cancel if a source read/unmap was not proved."""
        if self._source is not None:
            raise RuntimeError(
                "mirror still holds a borrowed frame output; retain its owner"
            )

    def close(self) -> None:
        self.finish_frame()
        if not self._images:
            return
        from cuda.bindings import runtime as cuda

        for image in list(self._images):
            if not image.closed:
                raise RuntimeError("owned mirror image still has a consumer")
            _check_cuda(cuda.cudaSetDevice(image.device))
            _check_cuda(cuda.cudaFree(image.pointer))
            self._images.remove(image)


def map_cuda_rgba(renderer: int, handle: int, max_bytes: int) -> MappedGpuImage:
    """Retain a ready linear CUDA map for stock OVUI's device-pointer upload.

    OVRTX may copy an array to linear memory here. This removes the CPU round
    trip, not all GPU copies. Map lifetime survives destruction of the step
    result; only the application owner unmaps, after UI copy completion.
    """
    return _map_rgba(renderer, handle, max_bytes, array=False)


def map_cuda_array_rgba(renderer: int, handle: int, max_bytes: int) -> MappedGpuImage:
    """Retain the native image array, avoiding OVRTX array-to-linear conversion."""
    return _map_rgba(renderer, handle, max_bytes, array=True)


def _map_rgba(renderer, handle, max_bytes, *, array):
    from cuda.bindings import runtime as cuda

    b, lib = _api()
    mapped = b.ovrtx_render_var_output_t()
    desc = b.ovrtx_map_output_description_t(
        device_type=b.Device.CUDA_ARRAY if array else b.Device.CUDA, sync_stream=None
    )
    result = lib.ovrtx_map_render_var_output(
        renderer,
        handle,
        ctypes.byref(desc),
        b.OVRTX_TIMEOUT_INFINITE,
        ctypes.byref(mapped),
    )
    if result.status != b.OVRTX_API_SUCCESS:
        raise RuntimeError("OVRTX GPU output map failed")
    image = MappedGpuImage(
        0, 0, 0, 0, 0, renderer=renderer, mapped=mapped, owner=threading.get_ident()
    )
    try:
        if mapped.status != b.EventStatus.COMPLETED or mapped.num_tensors != 1:
            raise RuntimeError(
                "OVRTX GPU output is incomplete or has unexpected tensors"
            )
        dl = mapped.tensors[0].dl.contents
        shape = tuple(int(dl.shape[i]) for i in range(dl.ndim))
        if dl.device.device_type.value != 2 or not dl.data:
            raise ValueError("mirror GPU output must be CUDA memory")
        image.device = int(dl.device.device_id)
        _check_cuda(cuda.cudaSetDevice(image.device))
        if array:
            if (
                dl.dtype.code.value != 3
                or dl.byte_offset
                or len(shape) != 3
                or shape[2] != 4
            ):
                raise ValueError("camera GPU output must be an opaque RGBA CUDA array")
            info = cuda.cudaArrayGetInfo(int(dl.data))
            _check_cuda(info)
            channel, extent = info[1:3]
            height, width, _ = shape
            if (channel.x, channel.y, channel.z, channel.w, int(channel.f)) != (
                8,
                8,
                8,
                8,
                1,
            ) or (extent.width, extent.height) != (width, height):
                raise ValueError(
                    "camera CUDA array is not unsigned RGBA8 at the declared size"
                )
            pitch = width * 4
            image.is_array = True
        elif dl.dtype.code.value != 1 or dl.dtype.bits != 8:
            raise ValueError("mirror GPU output must be CUDA uint8 RGBA")
        elif len(shape) == 3 and shape[2] == 4 and dl.dtype.lanes == 1:
            height, width, _ = shape
            pitch = int(dl.strides[0]) if dl.strides else width * 4
            if dl.strides and tuple(dl.strides[i] for i in (1, 2)) != (4, 1):
                raise ValueError("mirror GPU output channels are not packed RGBA")
        elif len(shape) == 2 and dl.dtype.lanes == 4:
            height, width = shape
            pitch = int(dl.strides[0]) * 4 if dl.strides else width * 4
            if dl.strides and dl.strides[1] != 1:
                raise ValueError("mirror GPU output pixels are not packed RGBA")
        else:
            raise ValueError(f"unexpected GPU output shape {shape}")
        if width <= 0 or height <= 0 or pitch < width * 4 or pitch * height > max_bytes:
            raise ValueError("mirror GPU output exceeds bounded image size")
        image.width, image.height, image.pitch = width, height, pitch
        image.pointer = int(dl.data) + int(dl.byte_offset)
        # Wait for producer completion before reading the mapped image.
        _wait_ready(cuda, mapped.cuda_sync)
        return image
    except BaseException as cause:
        try:
            image.close()
        except Exception as cleanup:
            raise OutputMapCleanupError(image, cause, cleanup) from cause
        raise

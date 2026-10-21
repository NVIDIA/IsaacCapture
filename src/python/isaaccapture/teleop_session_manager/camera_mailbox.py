# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded completed camera outputs for later application presentation.

CPU snapshots own their bytes. GPU snapshots retain a public OVRTX map until
next production or teardown; they may be consumed only on the renderer lane,
with synchronous copy completion. Cadence gaps retain the last completed image.
No borrowed result/output handle survives the callback.
"""

from __future__ import annotations

import ctypes
import threading
from dataclasses import dataclass
from typing import Any, Callable

from .gpu_output import MappedGpuImage, MappedOutput, OutputMapCleanupError, _api


@dataclass(frozen=True)
class CameraImage:
    width: int
    height: int
    rgba: bytes | None
    sensor_time: float
    revision: int
    gpu: MappedGpuImage | None = None


def _text(value: Any) -> str:
    if not value.ptr:
        return ""
    return ctypes.string_at(value.ptr, value.length).decode("utf-8")


def _latest_color_handle(outputs: Any, product_path: str) -> tuple[int, float] | None:
    """Select the latest *complete* LdrColor frame, not the first array entry."""
    latest: tuple[int, float] | None = None
    produced = False
    for product_index in range(outputs.output_count):
        product = outputs.outputs[product_index]
        if _text(product.render_product_path) != product_path:
            continue
        for frame_index in range(product.output_frame_count):
            produced = True
            frame = product.output_frames[frame_index]
            for var_index in range(frame.render_var_count):
                var = frame.output_render_vars[var_index]
                if _text(var.source_name) != "LdrColor" or not var.output_handle:
                    continue
                candidate = (int(var.output_handle), float(frame.frame_end_time))
                if latest is None or candidate[1] >= latest[1]:
                    latest = candidate
    if produced and latest is None:
        raise RuntimeError(f"{product_path} produced a frame without LdrColor")
    return latest


def _copy_host_rgba(
    renderer: int, handle: int, max_bytes: int
) -> tuple[int, int, bytes]:
    """Map one completed OVRTX output to CPU and copy before unmap."""
    b, lib = _api()
    description = b.ovrtx_map_output_description_t(
        device_type=b.Device.CPU, sync_stream=None
    )
    mapped = b.ovrtx_render_var_output_t()
    result = lib.ovrtx_map_render_var_output(
        renderer,
        handle,
        ctypes.byref(description),
        b.OVRTX_TIMEOUT_INFINITE,
        ctypes.byref(mapped),
    )
    if result.status != b.OVRTX_API_SUCCESS:
        raise RuntimeError("OVRTX camera output map failed")
    lease = MappedOutput(renderer=renderer, mapped=mapped, owner=threading.get_ident())
    failure = None
    try:
        if mapped.status != b.EventStatus.COMPLETED or mapped.num_tensors != 1:
            raise RuntimeError(
                "OVRTX camera output is incomplete or has unexpected tensors"
            )
        tensor = mapped.tensors[0].dl.contents
        shape = tuple(int(tensor.shape[i]) for i in range(tensor.ndim))
        dtype = tensor.dtype
        if dtype.code.value != 1 or dtype.bits != 8 or not tensor.data:
            raise RuntimeError("OVRTX camera output is not CPU uint8")
        if len(shape) == 3 and shape[2] == 4 and dtype.lanes == 1:
            height, width, _ = shape
            expected_strides = (width * 4, 4, 1)
        elif len(shape) == 2 and dtype.lanes == 4:
            height, width = shape
            expected_strides = (width, 1)
        else:
            raise RuntimeError(f"unexpected OVRTX camera tensor shape {shape}")
        length = width * height * 4
        if width <= 0 or height <= 0 or length > max_bytes:
            raise RuntimeError("OVRTX camera output exceeds bounded mailbox size")
        if (
            tensor.strides
            and tuple(tensor.strides[i] for i in range(tensor.ndim)) != expected_strides
        ):
            raise RuntimeError("OVRTX camera output is not contiguous RGBA8")
        return width, height, ctypes.string_at(tensor.data + tensor.byte_offset, length)
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            lease.close()
        except Exception as cleanup:
            # Keep the native handle even when validation or the host copy failed.
            raise OutputMapCleanupError(lease, failure, cleanup) from failure


class CompletedCameraMailbox:
    """Latest camera image and explicit error/omission state for one product."""

    def __init__(
        self,
        product_path: str,
        max_image_bytes: int,
        *,
        copier: Callable[[int, int, int], tuple[int, int, bytes]] = _copy_host_rgba,
        gpu_mapper=None,
    ) -> None:
        if not product_path.startswith("/") or max_image_bytes <= 0:
            raise ValueError(
                "camera mailbox requires an absolute product path and positive byte bound"
            )
        self.product_path = product_path
        self.max_image_bytes = max_image_bytes
        self._copier = copier
        self._gpu_mapper = gpu_mapper
        self._mapped = None
        self._gpu_error = None
        self._lock = threading.Lock()
        self._image: CameraImage | None = None
        self._revision = 0
        self._error: str | None = None

    def snapshot(self) -> CameraImage | None:
        with self._lock:
            return self._image

    def error(self) -> str | None:
        with self._lock:
            return self._error

    def on_outputs(
        self, _user_data: object, renderer: int, outputs_pointer: Any
    ) -> None:
        """C callback: contain all exceptions and never retain borrowed handles."""
        try:
            if self._gpu_error is not None:
                raise RuntimeError(self._gpu_error)
            selected = _latest_color_handle(outputs_pointer.contents, self.product_path)
            if selected is None:
                return
            handle, sensor_time = selected
            gpu = None
            if self._gpu_mapper is not None:
                self.close()
                gpu = self._gpu_mapper(renderer, handle, self.max_image_bytes)
                self._mapped = gpu
                width, height, rgba = gpu.width, gpu.height, None
            else:
                width, height, rgba = self._copier(
                    renderer, handle, self.max_image_bytes
                )
                if len(rgba) != width * height * 4 or len(rgba) > self.max_image_bytes:
                    raise RuntimeError("camera copier returned invalid RGBA8 size")
            with self._lock:
                self._revision += 1
                self._image = CameraImage(
                    width,
                    height,
                    bytes(rgba) if rgba is not None else None,
                    sensor_time,
                    self._revision,
                    gpu,
                )
                self._error = None
        except Exception as exc:
            if isinstance(exc, OutputMapCleanupError):
                self._mapped = exc.image
                self._gpu_error = str(exc)
            with self._lock:
                self._image = None
                self._error = str(exc)

    def close(self) -> None:
        """Frame owner only, after synchronous quad consumption, before renderer destruction."""
        if self._mapped is not None:
            try:
                self._mapped.close()
            except Exception as error:
                self._gpu_error = str(error)
                raise
            self._mapped = None

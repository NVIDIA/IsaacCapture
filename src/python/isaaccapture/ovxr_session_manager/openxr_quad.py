# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application-owned camera panels on borrowed OpenXR/Vulkan resources."""

from __future__ import annotations

import ctypes
import logging
from enum import IntEnum
from pathlib import Path

from .camera_mailbox import CameraImage

_LOGGER = logging.getLogger(__name__)


class _UploadResult(IntEnum):
    """Integer C ABI shared with UploadResult in pip_layer.cpp."""

    SUCCESS = 0
    INVALID_ARGUMENT = -1
    NOT_ATTACHED = -2
    CAMERA_UNAVAILABLE = -3
    SWAPCHAIN_FAILED = -4
    ALLOCATION_FAILED = -5
    UPLOAD_FAILED = -6
    COMPLETION_UNPROVED = -7
    CUDA_UNAVAILABLE = -8
    UPLOAD_MODE_MISMATCH = -9


class OpenXRCameraPanels:
    """One native owner for panel spaces, swapchains and upload completion."""

    def __init__(self, panels: tuple, *, library_path: str | None = None) -> None:
        library = (
            Path(library_path)
            if library_path
            else Path(__file__).parent / "native/libisaaccapture_camera_panels.so"
        )
        self._native = ctypes.CDLL(str(library.resolve(strict=True)))
        pointer = ctypes.c_void_p
        uint = ctypes.c_uint32
        signatures = {
            "create": ([], pointer),
            "max_cameras": ([], uint),
            "max_image_dimension": ([], uint),
            "supports_cuda": ([], ctypes.c_int),
            "attach": ([pointer] * 7 + [uint], ctypes.c_int),
            "destroy": ([pointer], ctypes.c_int),
            "wait_cuda": ([pointer, uint], ctypes.c_int),
            "layer": ([pointer, uint], pointer),
            "configure": (
                [
                    pointer,
                    uint,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                ],
                ctypes.c_int,
            ),
            "publish_rgba": (
                [pointer, uint, pointer, uint, uint, ctypes.c_size_t],
                ctypes.c_int,
            ),
            "publish_cuda_array": (
                [pointer, uint, pointer, uint, uint, ctypes.c_int],
                ctypes.c_int,
            ),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self._native, f"isaacteleop_pip_{name}")
            function.argtypes, function.restype = args, result
        self._panels = tuple(panels)
        limit = self._native.isaacteleop_pip_max_cameras()
        if len(panels) > limit:
            raise ValueError(f"native camera helper supports at most {limit} cameras")
        self._max_dimension = self._native.isaacteleop_pip_max_image_dimension()
        self.supports_gpu = bool(self._native.isaacteleop_pip_supports_cuda())
        self._visible = [False] * len(panels)
        self._published_revision = [0] * len(panels)
        self.last_errors = [None] * len(panels)
        self._attached = False
        self._pending_uploads = {}
        self._retired_panels = set()
        self._handle = self._native.isaacteleop_pip_create()
        if not self._handle:
            raise MemoryError("could not create camera panel owner")

    def attach(self, openxr, vulkan) -> None:
        if self._attached:
            return
        if self.supports_gpu and not (vulkan.external_memory_handle_types & 1):
            raise RuntimeError(
                "camera GPU upload requires enabled opaque-FD external memory"
            )
        result = self._native.isaacteleop_pip_attach(
            self._handle,
            openxr.xr_instance,
            openxr.xr_session,
            openxr.xr_get_instance_proc_addr,
            vulkan.vk_physical_device,
            vulkan.vk_device,
            vulkan.vk_queue,
            vulkan.queue_family_index,
        )
        if result != 0:
            # The owner also retains partial attachment until close succeeds.
            raise RuntimeError(f"camera session binding failed: {result}")
        self._attached = True

    def layer(self, index: int):
        return self._native.isaacteleop_pip_layer(self._handle, index)

    def _set_panel(self, index: int, enabled: bool, width: int, height: int) -> None:
        panel = self._panels[index]
        pose = (ctypes.c_float * 7)(*panel.position, *panel.orientation_xyzw)
        size = (ctypes.c_float * 2)(
            panel.panel_width_m, panel.panel_width_m * height / width
        )
        result = self._native.isaacteleop_pip_configure(
            self._handle, index, int(enabled), 1, pose, size
        )
        if result != 0:
            raise RuntimeError(f"OpenXR camera quad configuration failed: {result}")
        self._visible[index] = enabled

    def publish(
        self, index: int, image: CameraImage | None, *, invalid: bool = False
    ) -> bool:
        if index in self._retired_panels:
            return False
        if invalid:
            self.hide(index)
            self.last_errors[index] = "completed camera output was invalid"
            return False
        if image is None or image.revision <= self._published_revision[index]:
            return False
        try:
            if (
                not 0 < image.width <= self._max_dimension
                or not 0 < image.height <= self._max_dimension
            ):
                raise ValueError("camera image exceeds native helper dimensions")
            self._set_panel(index, True, image.width, image.height)
            if image.gpu is not None:
                gpu = image.gpu
                if gpu.closed or gpu.busy or not gpu.is_array or not self.supports_gpu:
                    raise RuntimeError("camera GPU array is unavailable")
                gpu.busy = True
                result = self._native.isaacteleop_pip_publish_cuda_array(
                    self._handle,
                    index,
                    gpu.pointer,
                    image.width,
                    image.height,
                    gpu.device,
                )
                # Keep the borrowed map until the native copy has completed.
                if result != _UploadResult.COMPLETION_UNPROVED:
                    gpu.busy = False
                else:
                    self._pending_uploads[index] = gpu
                    self._retired_panels.add(index)
                    _LOGGER.warning(
                        "Camera panel %s disabled after incomplete GPU copy", index
                    )
            else:
                pixels = ctypes.create_string_buffer(image.rgba)
                result = self._native.isaacteleop_pip_publish_rgba(
                    self._handle,
                    index,
                    pixels,
                    image.width,
                    image.height,
                    len(image.rgba),
                )
        except Exception:
            self.hide(index)
            raise
        if result != _UploadResult.SUCCESS:
            self.hide(index)
            self.last_errors[index] = (
                f"OpenXR camera quad upload failed: {_UploadResult(result).name} ({result})"
            )
            return False
        self._published_revision[index] = image.revision
        self.last_errors[index] = None
        return True

    def hide(self, index: int) -> None:
        if self._visible[index]:
            self._set_panel(index, False, 1, 1)

    def finish_frame(self) -> None:
        first_error = None
        for index, gpu in tuple(self._pending_uploads.items()):
            result = self._native.isaacteleop_pip_wait_cuda(self._handle, index)
            if result != _UploadResult.SUCCESS:
                error = RuntimeError(
                    f"camera {index} GPU copy completion is unproved: "
                    f"{_UploadResult(result).name} ({result})"
                )
                if first_error is None:
                    first_error = error
                else:
                    first_error.add_note(str(error))
                continue
            gpu.busy = False
            del self._pending_uploads[index]
        if first_error is not None:
            raise first_error

    def close(self) -> None:
        if self._handle:
            result = self._native.isaacteleop_pip_destroy(self._handle)
            if result != 0:
                raise RuntimeError(
                    "camera GPU completion/cleanup is unproved; retain OVXR context"
                )
            self._handle = None
            self._attached = False

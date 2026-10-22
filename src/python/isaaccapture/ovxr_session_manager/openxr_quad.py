# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application-owned camera quads using borrowed OVXR OpenXR/Vulkan handles.

Only the frame owner calls this helper. Uploads finish before layer descriptors
are passed to OVXR submit; all owned resources close before the borrowed context.
The package manifest locates the native helper but is never registered with the
OpenXR loader. Explicit v3 binding rejects the former interception-only binary.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .camera_mailbox import CameraImage


_LAYER_NAME = "XR_APILAYER_ISAACTELEOP_camera_quad"
_MANIFEST_NAME = "XrApiLayer_isaacteleop_pip.json"


class OpenXRCameraQuad:
    def __init__(self, manifest: str | None = None, *, camera_index: int = 0) -> None:
        if not 0 <= camera_index < 3:
            raise ValueError("camera quad index must be in [0, 3)")
        self._camera_index = camera_index
        override = manifest or os.environ.get("ISAAC_TELEOP_PIP_LAYER_MANIFEST")
        root = (
            Path(override)
            if override
            else Path(__file__).parent / "openxr_layer" / _MANIFEST_NAME
        )
        root = root.resolve(strict=True)
        data = json.loads(root.read_text())
        layer = data["api_layer"]
        if layer["name"] != _LAYER_NAME:
            raise RuntimeError(f"unexpected OpenXR camera layer: {layer['name']}")
        library = (root.parent / layer["library_path"]).resolve(strict=True)
        self._native = ctypes.CDLL(str(library))
        self._native.isaacteleop_pip_configure_v2.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self._native.isaacteleop_pip_configure_v2.restype = ctypes.c_int
        self._native.isaacteleop_pip_publish_rgba_v2.argtypes = [
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_size_t,
        ]
        self._native.isaacteleop_pip_publish_rgba_v2.restype = ctypes.c_int
        self._native.isaacteleop_pip_supports_cuda_v2.restype = ctypes.c_int
        self.supports_gpu = bool(self._native.isaacteleop_pip_supports_cuda_v2())
        self._native.isaacteleop_pip_publish_cuda_array_v2.argtypes = [
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_int,
        ]
        self._native.isaacteleop_pip_publish_cuda_array_v2.restype = ctypes.c_int
        self._native.isaacteleop_pip_attach_v3.argtypes = [ctypes.c_void_p] * 6 + [
            ctypes.c_uint32
        ]
        self._native.isaacteleop_pip_attach_v3.restype = ctypes.c_int
        self._native.isaacteleop_pip_detach_v3.argtypes = []
        self._native.isaacteleop_pip_detach_v3.restype = ctypes.c_int
        self._native.isaacteleop_pip_layer_v3.argtypes = [ctypes.c_uint32]
        self._native.isaacteleop_pip_layer_v3.restype = ctypes.c_void_p
        self._attached = False
        self._published_revision = 0
        self.last_error: str | None = None
        self._panel: object | None = None
        self._visible = False

    def attach(self, openxr, vulkan) -> None:
        """Borrow the existing session and queue; never create a second owner."""
        if self._camera_index != 0 or self._attached:
            raise RuntimeError("camera session binding must have exactly one owner")
        if self.supports_gpu and not (vulkan.external_memory_handle_types & 1):
            raise RuntimeError(
                "camera GPU upload requires enabled opaque-FD external memory"
            )
        result = self._native.isaacteleop_pip_attach_v3(
            openxr.xr_instance,
            openxr.xr_session,
            openxr.xr_get_instance_proc_addr,
            vulkan.vk_physical_device,
            vulkan.vk_device,
            vulkan.vk_queue,
            vulkan.queue_family_index,
        )
        # -3 allocated partial owned spaces; close must retire them before the
        # borrowed session may die. Other failures allocated nothing for us.
        self._attached = result in (0, -3)
        if result != 0:
            raise RuntimeError(f"camera session binding failed: {result}")

    def layer(self):
        """Borrow the latest completed quad descriptor through the next submit."""
        return self._native.isaacteleop_pip_layer_v3(self._camera_index)

    def configure(self, panel: object) -> None:
        if panel.space != "head_locked":
            raise ValueError(
                "the first camera-quad slice requires explicit head_locked placement"
            )
        self._panel = panel

    def _set_panel(self, enabled: bool, width: int, height: int) -> None:
        panel = self._panel
        if panel is None:
            raise RuntimeError("OpenXR camera quad has no application panel policy")
        pose = (ctypes.c_float * 7)(*panel.position, *panel.orientation_xyzw)
        size = (ctypes.c_float * 2)(
            panel.panel_width_m, panel.panel_width_m * height / width
        )
        result = self._native.isaacteleop_pip_configure_v2(
            self._camera_index, int(enabled), 1, pose, size
        )
        if result != 0:
            raise RuntimeError(f"OpenXR camera quad configuration failed: {result}")
        self._visible = enabled

    def publish(self, image: CameraImage | None, *, invalid: bool = False) -> bool:
        if invalid:
            self.hide()
            self.last_error = "completed camera output was invalid"
            return False
        if image is None or image.revision <= self._published_revision:
            return False
        try:
            self._set_panel(True, image.width, image.height)
            if image.gpu is not None:
                gpu = image.gpu
                if gpu.closed or gpu.busy or not gpu.is_array or not self.supports_gpu:
                    raise RuntimeError("camera GPU array is unavailable")
                gpu.busy = True
                result = self._native.isaacteleop_pip_publish_cuda_array_v2(
                    self._camera_index,
                    gpu.pointer,
                    image.width,
                    image.height,
                    gpu.device,
                )
                # A -7 return cannot prove input-copy completion. Keep the map
                # and reject further production/teardown rather than recycle it.
                if result != -7:
                    gpu.busy = False
            else:
                pixels = ctypes.create_string_buffer(image.rgba)
                result = self._native.isaacteleop_pip_publish_rgba_v2(
                    self._camera_index,
                    pixels,
                    image.width,
                    image.height,
                    len(image.rgba),
                )
        except Exception:
            self.hide()
            raise
        if result != 0:
            self.hide()
            self.last_error = f"OpenXR camera quad upload failed: {result}"
            return False
        self._published_revision = image.revision
        self.last_error = None
        return True

    def hide(self) -> None:
        """Withdraw the previous image when a produced frame cannot be published."""
        if self._visible:
            self._set_panel(False, 1, 1)

    def close(self) -> None:
        """Release owned spaces/swapchains/imports before OVXR destroys its device."""
        self.hide()
        if self._attached:
            result = self._native.isaacteleop_pip_detach_v3()
            if result != 0:
                raise RuntimeError(
                    "camera GPU completion/cleanup is unproved; retain OVXR context"
                )
            self._attached = False

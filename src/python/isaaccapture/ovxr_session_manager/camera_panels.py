# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera picture-in-picture as an application-neutral OVXR frame consumer."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

from .camera_mailbox import CompletedCameraMailbox, FrameOutputs
from .gpu_output import map_cuda_array_rgba
from .openxr_quad import OpenXRCameraPanels
from .ovxr_session import OvxrFrameConsumer

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraPanelConfig:
    """One host-produced camera and its head-locked presentation policy."""

    render_product_path: str
    max_image_bytes: int = 64 * 1024 * 1024
    panel_width_m: float = 0.48
    position: tuple[float, float, float] = (0.0, -0.15, -0.8)
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    space: Literal["head_locked"] = "head_locked"
    label: str | None = None

    def __post_init__(self) -> None:
        if not self.render_product_path.startswith("/"):
            raise ValueError("camera render product path must be absolute")
        if self.max_image_bytes <= 0:
            raise ValueError("camera panel max_image_bytes must be positive")
        if not math.isfinite(self.panel_width_m) or self.panel_width_m <= 0.0:
            raise ValueError("camera panel panel_width_m must be finite and positive")
        if len(self.position) != 3 or not all(
            math.isfinite(value) for value in self.position
        ):
            raise ValueError("camera panel position must contain three finite values")
        if len(self.orientation_xyzw) != 4 or not all(
            math.isfinite(value) for value in self.orientation_xyzw
        ):
            raise ValueError("camera panel orientation must contain four finite values")
        norm = math.sqrt(sum(value * value for value in self.orientation_xyzw))
        if norm <= 0.0:
            raise ValueError("camera panel orientation must have non-zero norm")
        object.__setattr__(
            self,
            "orientation_xyzw",
            tuple(value / norm for value in self.orientation_xyzw),
        )
        if self.space != "head_locked":
            raise ValueError("camera panel space must be 'head_locked'")


class CameraPanelConsumer(OvxrFrameConsumer):
    """Own camera uploads and composition layers until the XR owner retires them."""

    def __init__(self, panels: tuple[CameraPanelConfig, ...]) -> None:
        self.panels = tuple(panels)
        paths = self.render_product_paths
        if len(paths) != len(set(paths)):
            raise ValueError("camera panels require unique paths")
        self._native = None
        self._mailboxes = []
        self._layers = ()
        self._attach_attempted = False
        self._enabled = False

    @property
    def render_product_paths(self) -> tuple[str, ...]:
        return tuple(panel.render_product_path for panel in self.panels)

    def on_session_started(self, binding, config, request_shutdown) -> None:
        self._native = OpenXRCameraPanels(self.panels)
        for panel in self.panels:
            self._mailboxes.append(
                CompletedCameraMailbox(
                    panel.render_product_path,
                    panel.max_image_bytes,
                    gpu_mapper=map_cuda_array_rgba
                    if self._native.supports_gpu
                    else None,
                )
            )

    def on_openxr_ready(self, openxr, vulkan) -> bool:
        if not self._attach_attempted:
            self._attach_attempted = True
            try:
                if vulkan is None:
                    raise RuntimeError("Vulkan handles are unavailable")
                self._native.attach(openxr, vulkan)
            except Exception:
                # Partial native attachment stays owned until normal close.
                _LOGGER.exception(
                    "Camera panels disabled; teleoperation remains available"
                )
            else:
                self._enabled = True
        return True

    def on_outputs(self, frame: FrameOutputs) -> None:
        if not self._enabled:
            return
        layers = []
        self._layers = ()
        for index, mailbox in enumerate(self._mailboxes):
            try:
                mailbox.on_outputs(frame)
                self._native.publish(
                    index, mailbox.snapshot(), invalid=mailbox.error() is not None
                )
                if layer := self._native.layer(index):
                    layers.append(layer)
            except Exception:
                _LOGGER.exception("OpenXR camera quad publication failed")
                try:
                    self._native.hide(index)
                except Exception:
                    _LOGGER.exception("OpenXR camera quad hiding failed")
        self._layers = tuple(layers)

    def composition_layers(self) -> tuple[int, ...]:
        return self._layers

    def finish_frame(self) -> None:
        first_error = None
        operations = [self._native.finish_frame] if self._native is not None else []
        operations.extend(mailbox.close for mailbox in self._mailboxes)
        for finish in operations:
            try:
                finish()
            except Exception as error:
                # Retain unresolved maps without skipping independent mailboxes.
                if first_error is None:
                    first_error = error
                else:
                    first_error.add_note(str(error))
        if first_error is not None:
            raise first_error

    def close(self) -> None:
        self.finish_frame()
        if self._native is not None:
            self._native.close()
            self._native = None
        self._mailboxes.clear()
        self._layers = ()

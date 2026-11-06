# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application publication, resource ownership and frame-consumer contracts."""

from __future__ import annotations
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from .camera_mailbox import FrameOutputs


class OvxrScenePublication(Protocol):
    """Application-thread publication and scene reads."""

    @abstractmethod
    def publication_guard(self) -> AbstractContextManager[None]:
        raise NotImplementedError

    @abstractmethod
    def publish_complete_scene(self, publication_id: Hashable) -> int:
        raise NotImplementedError

    @abstractmethod
    def validate_first_published_scene(self, ordinal: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_target_frame_T_world(self, prim_path: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def close_publication(self) -> None:
        raise NotImplementedError


class OvxrResourceOwner(Protocol):
    """Transfer and retirement of application-owned rendering resources."""

    @abstractmethod
    def offer_render_owner_transfer(self) -> object:
        raise NotImplementedError

    @abstractmethod
    def attach_on_render_thread(self, capability: object) -> OvxrSessionBinding:
        raise NotImplementedError

    @abstractmethod
    def owner_lane(self) -> AbstractContextManager[None]:
        raise NotImplementedError

    @abstractmethod
    def close_on_render_thread(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close_environment_on_render_thread(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def abort(self) -> None:
        raise NotImplementedError


class OvxrApplicationInterface(OvxrScenePublication, OvxrResourceOwner, Protocol):
    """Combined application contract accepted by OvxrSession."""


class StageOrdinalLedger(Protocol):
    """Application-owned reservations for explicit OVXR Stage writes."""

    @abstractmethod
    def reserve(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def resolve_external(
        self, ordinal: int, *, expected_published: bool | None
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def abandon(self, ordinal: int) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class OvxrSessionBinding:
    """Exact application-owned objects borrowed by one OVXR context."""

    renderer: object
    stage: object
    ordinal_ledger: StageOrdinalLedger
    frame_consumers: tuple[OvxrFrameConsumer, ...] = ()
    prepare_frame_scene: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        if self.renderer is None or self.stage is None or self.ordinal_ledger is None:
            raise ValueError(
                "OVXR session requires renderer, Stage, and ordinal ledger"
            )


@dataclass
class OvxrSessionConfig:
    """Application-neutral OVXR and CloudXR configuration.

    The supported OVXR package supplies bound-frame outputs. Optional PiP and
    output consumers copy those outputs on the sole frame owner before retirement.
    """

    camera_paths: tuple[str, ...]
    application_name: str = "isaacteleop-ovxr-session"
    session_mode: Literal["ar", "vr"] = "vr"
    cloudxr_env_file: str | None = None
    auto_launch_cloudxr: bool = True
    accept_eula: bool = False
    render_products: tuple[str, ...] = ()
    create_render_products: bool = True
    up_axis: Literal["Y", "Z"] = "Z"
    meters_per_unit: float = 1.0
    depth_near_z: float = 0.0
    depth_far_z: float = 0.0
    tracking_anchor_path: str = ""
    floor_aligned_anchor: bool | None = None
    host_wait_render_output: bool = False
    extra_openxr_extensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.up_axis not in ("Y", "Z"):
            raise ValueError("up_axis must be 'Y' or 'Z'")
        if self.session_mode not in ("ar", "vr"):
            raise ValueError("session_mode must be 'ar' or 'vr'")
        if not math.isfinite(self.meters_per_unit) or self.meters_per_unit <= 0:
            raise ValueError("meters_per_unit must be finite and positive")
        if (self.depth_near_z, self.depth_far_z) != (0.0, 0.0) and (
            not math.isfinite(self.depth_near_z)
            or not math.isfinite(self.depth_far_z)
            or self.depth_near_z <= 0.0
            or self.depth_far_z <= self.depth_near_z
        ):
            raise ValueError(
                "depth_near_z and depth_far_z must both be zero or finite with 0 < near < far"
            )
        if len(self.camera_paths) != 2 or any(not path for path in self.camera_paths):
            raise ValueError("OVXR stereo requires exactly two non-empty camera paths")


class OvxrFrameConsumer(ABC):
    """Optional presentation on the OVXR owner thread.

    Outputs are borrowed until finish_frame, before submit/cancel. Composition
    layers remain owned by the consumer through submit. on_openxr_ready may be
    called repeatedly and must be idempotent. close must tolerate
    partial startup and retain resources when retirement cannot be proved.
    """

    @property
    def render_product_paths(self) -> tuple[str, ...]:
        return ()

    def on_session_started(self, binding, config, request_shutdown) -> None:
        return None

    def on_openxr_ready(self, openxr, vulkan) -> bool:
        """Bind borrowed XR resources; return False while required handles are absent."""
        return True

    @abstractmethod
    def on_outputs(self, frame: FrameOutputs) -> None:
        """Consume this completed frame without retaining its output descriptors."""
        raise NotImplementedError

    def composition_layers(self) -> tuple[int, ...]:
        """Borrow native layer pointers valid through the following submission."""
        return ()

    def on_frame_submitted(self) -> None:
        return None

    def finish_frame(self) -> None:
        return None

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

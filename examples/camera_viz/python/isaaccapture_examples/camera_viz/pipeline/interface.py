# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source contract for camera_viz.

Sources own their own producer thread (or rely on the vendor SDK's
callback thread) and expose ``latest()`` as a non-blocking mailbox.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class SourceSpec:
    """Static description of what a source produces.

    Layers are sized to ``(width, height)`` at construction; the source
    contract is that every ``Frame.image`` it later emits matches this
    shape + format exactly.
    """

    name: str
    width: int
    height: int
    pixel_format: str = "rgba8"  # Phase 1: only RGBA8


@dataclass
class Frame:
    """One produced frame, GPU-resident.

    ``image`` is anything that exposes ``__cuda_array_interface__`` —
    CuPy / PyTorch / Numba arrays all work. ``stream`` is the producer's
    CUDA stream so the consumer can synchronize when it's not 0/default.

    Stereo: ``image`` is the left eye; ``image_right`` carries the
    right eye when paired (both from the same capture instant).

    ``image`` may be *borrowed* rather than owned: a source is allowed to
    hand back a view onto memory it will reuse (``cuda_ipc`` maps producer
    memory directly). A borrowed frame stays valid only until the next
    ``latest()`` on the source that produced it, so a consumer must finish
    reading — or copy — before polling again, and must not retain a Frame
    across polls. ``ImageLayerBase::submit`` upholds this by synchronizing
    its copy before returning; keep that guarantee if you change it.
    """

    image: Any
    timestamp_ns: int
    source_id: str
    stream: int = 0
    image_right: Optional[Any] = None


class FrameSource(ABC):
    """Pull-based GPU-resident frame source."""

    @property
    @abstractmethod
    def spec(self) -> SourceSpec: ...

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def latest(self) -> Optional[Frame]:
        """Return the freshest frame, or None if no new frame since the
        last call. Must be non-blocking — the render loop polls this
        every frame and skips submission on None. Raise on a fatal
        producer failure so the runner can stop and report it."""

    @abstractmethod
    def stop(self) -> None: ...

    # Convenience context manager so app code can `with source:`.
    def __enter__(self) -> "FrameSource":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

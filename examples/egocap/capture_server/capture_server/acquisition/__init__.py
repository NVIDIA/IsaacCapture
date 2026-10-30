"""Ordinary, SDK-agnostic acquisition adapters shared by mock and real
modes -- see each module's own docstring. No mock-specific behavior
lives here; see :mod:`capture_server.mock` for that."""

from __future__ import annotations

from .acquisition import Acquisition
from .depthai import DepthAIAdapter
from .errors import CameraSourceError, SourceError, SourceExited
from .isaac_capture import IsaacCaptureAdapter, TrackingSelection

__all__ = [
    "Acquisition",
    "CameraSourceError",
    "DepthAIAdapter",
    "IsaacCaptureAdapter",
    "SourceError",
    "SourceExited",
    "TrackingSelection",
]

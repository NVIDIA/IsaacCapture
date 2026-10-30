"""Exceptions an acquisition adapter raises when its underlying SDK
(real or :mod:`capture_server.mock`) fails. These are plain, SDK-agnostic
exception types -- not tied to any particular adapter shape -- so
``_Session`` can map them onto :class:`capture_server.protocol.StopReason`
the same way regardless of which concrete adapter raised them."""

from __future__ import annotations

__all__ = ["CameraSourceError", "SourceError", "SourceExited"]


class SourceError(Exception):
    """A startup/operation failure from an acquisition adapter, mapped to
    ``StopReason.IC_ERROR`` (or ``CAMERA_ERROR`` for :class:`CameraSourceError`)."""


class SourceExited(SourceError):
    """The underlying acquisition process/runtime exited on its own mid
    capture, mapped to ``StopReason.IC_EXITED``."""


class CameraSourceError(SourceError):
    """A specific camera failed to start or deliver a frame, mapped to
    ``StopReason.CAMERA_ERROR``. ``camera_name`` identifies which
    configured camera failed."""

    def __init__(self, message: str, *, camera_name: str) -> None:
        super().__init__(message)
        self.camera_name = camera_name

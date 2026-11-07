# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GET /api/settings: the backend's effective configuration, read-only.

Everything reported here was set before the backend started, through the environment or the station YAML
(see backend/README.md). The page can inspect it so a misconfigured station is visible from the browser instead
of only from the launch shell, but it cannot change it: a restart is the only way to apply a new value, and the
browser is never a configuration source (config.py states the same rule for capture). Secrets are reported as
present or absent, never by value. The application lifespan resolves the snapshot once into
``app.state.settings``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from . import recordings
from .config import CaptureStatusConfig, VisualizationConfig
from .livekit_preview import API_KEY_ENV, API_SECRET_ENV
from .preview_protocol import CAMERA_SAMPLE_HZ, TRACKING_SAMPLE_HZ
from .preview_routes import _configured_allowed_origins, _trusted_hostnames
from .station_config import ResolvedStation

router = APIRouter()

_NO_STORE = {"Cache-Control": "no-store"}


def _enum_name(value: Any) -> Any:
    """An enum member as its lower-case name; anything else as it is (the wire form avoids bare numbers)."""
    name = getattr(value, "name", None)
    return name.lower() if isinstance(name, str) else value


@dataclass(frozen=True)
class EffectiveSettings:
    """What the backend is running with, resolved once at startup by the application lifespan."""

    capture: CaptureStatusConfig
    #: ``EGOCAP_STATION_CONFIG`` as given, or ``None`` when unset; ``station`` is ``None`` when unset or invalid.
    station_path: Path | None
    station: ResolvedStation | None
    #: The public (path-free) reason the station file was rejected, or ``None``.
    station_error: str | None
    recording_output_root: Path
    #: ``None`` when the 3D view settings could not be read; ``visualization_error`` then says why.
    visualization: VisualizationConfig | None
    visualization_error: str | None
    task_file_root: Path

    def to_json(self) -> dict[str, Any]:
        """The wire form: plain JSON types, enum names instead of numbers, secrets as booleans."""
        capture, station, viz = self.capture, self.station, self.visualization
        # Origin policy, LiveKit credentials and the recordings flag are read from the environment on every call,
        # the same way their own modules read them, so this view never disagrees with the behaviour it describes.
        trusted = _trusted_hostnames()
        return {
            "capture": {
                "server_url": capture.capture_server_url,
                "station_label": capture.station_label,
                "profile_label": capture.profile_label,
                "sources": [_enum_name(source) for source in capture.sources],
                "plugins": [_enum_name(plugin) for plugin in capture.plugins],
                "cameras": [{"name": camera.name, "type": _enum_name(camera.type)} for camera in capture.cameras],
            },
            "preview": {"camera_sample_hz": CAMERA_SAMPLE_HZ, "tracking_sample_hz": TRACKING_SAMPLE_HZ},
            "station": {
                "path": str(self.station_path) if self.station_path is not None else None,
                "error": self.station_error,
                "id": station.station_id if station is not None else None,
                "operator_name": station.operator_name if station is not None else None,
                "auto_fill_operator_name": station.auto_fill_operator_name if station is not None else False,
                "task_description": station.task_description if station is not None else None,
                "data_dir": str(station.data_dir)
                if station is not None and station.data_dir is not None
                else None,
            },
            "recording": {"output_root": str(self.recording_output_root)},
            "visualization": {
                "error": self.visualization_error,
                "enabled": viz.enabled if viz is not None else False,
                "host": viz.host if viz is not None else None,
                "port": viz.port if viz is not None else None,
                "public_url": viz.public_url if viz is not None else None,
                "device": viz.device if viz is not None else None,
                "head_model": str(viz.head_model) if viz is not None and viz.head_model is not None else None,
                "head_model_scale": viz.head_model_scale if viz is not None else None,
            },
            "livekit": {
                "credentials_configured": bool(os.environ.get(API_KEY_ENV))
                and bool(os.environ.get(API_SECRET_ENV)),
            },
            "recordings_browser": {"enabled": recordings._enabled()},
            "task_files": {"dir": str(self.task_file_root), "available": self.task_file_root.is_dir()},
            "origins": {
                "allowed": sorted(_configured_allowed_origins()),
                "trusted_hostnames": sorted(trusted) if trusted is not None else None,
            },
        }


@router.get("/api/settings")
async def read_settings(request: Request) -> JSONResponse:
    """The effective configuration as JSON; 503 until the lifespan has resolved it."""
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        return JSONResponse({"detail": "settings are not resolved yet"}, status_code=503, headers=_NO_STORE)
    return JSONResponse(settings.to_json(), headers=_NO_STORE)

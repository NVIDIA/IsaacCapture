"""Backend-launch configuration for the capture-status application.

Everything here is chosen by whoever starts the backend process (an
operator/deployment action), never by the browser -- the browser only ever
reads the resulting status, it cannot edit configuration or supply native
paths/addresses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from capture_server import protocol as p


@dataclass(frozen=True)
class CaptureStatusConfig:
    """One fixed CaptureServer attempt's configuration, with defaults that work against a local
    mock CaptureServer for development. station_label and cameras are overridden by the station
    YAML when configured (see app.py's lifespan); the rest is fixed except for the CaptureServer
    URL. No cameras are requested until a station YAML configures them."""

    capture_server_url: str = "ws://127.0.0.1:8082"
    station_label: str = "Station 01"
    profile_label: str = "Headset and hands"
    sources: tuple[p.SourceType, ...] = (
        p.SourceType.HEAD,
        p.SourceType.CONTROLLERS,
        p.SourceType.HANDS,
    )
    cameras: tuple[p.CameraSelection, ...] = field(default_factory=tuple)
    plugins: tuple[p.PluginType, ...] = field(default_factory=tuple)


class PluginConfigError(ValueError):
    """Raised for a malformed EGOCAP_CAPTURE_PLUGINS value. Raised while the
    backend is starting, so a typo in a deployment stops the process with a
    message naming the bad value instead of silently running without the
    plugin an operator asked for."""


#: Operator-facing plugin names -> wire types, derived from the protocol enum so
#: a plugin added there becomes configurable here without a second list to update.
_PLUGIN_NAMES: dict[str, p.PluginType] = {member.name.lower(): member for member in p.PluginType}


def _parse_plugins(raw: str | None) -> tuple[p.PluginType, ...]:
    """Parses EGOCAP_CAPTURE_PLUGINS: a comma-separated list of plugin names
    (case-insensitive, surrounding whitespace ignored), e.g. ``manus``.
    Unset or blank means no plugins -- the default session uses only the
    headset's own tracking.

    Only the plugin *type* is chosen here. CaptureClient turns each type into
    a ``PluginRequest`` with its defaults: ``required=False``, so a plugin
    whose device is absent or fails to load is reported in the status instead
    of aborting the whole headset session, and the plugin's default datasets.
    """
    if raw is None or raw.strip() == "":
        return ()
    plugins: list[p.PluginType] = []
    for item in raw.split(","):
        name = item.strip().lower()
        # A blank entry ("manus,") is a malformed value, not a request for nothing: say so rather than guess.
        if not name:
            raise PluginConfigError(f"EGOCAP_CAPTURE_PLUGINS has an empty entry: {raw!r}")
        plugin = _PLUGIN_NAMES.get(name)
        # Unknown names fail fast with the accepted spellings rather than silently running without the plugin.
        if plugin is None:
            raise PluginConfigError(
                f"EGOCAP_CAPTURE_PLUGINS entry {item.strip()!r} is not a known plugin; "
                f"valid names: {sorted(_PLUGIN_NAMES)}"
            )
        # The same plugin twice is almost certainly an editing mistake; the wire format forbids it too.
        if plugin in plugins:
            raise PluginConfigError(f"EGOCAP_CAPTURE_PLUGINS lists {name!r} more than once")
        plugins.append(plugin)
    return tuple(plugins)


class SourceConfigError(ValueError):
    """Raised for a malformed EGOCAP_CAPTURE_SOURCES value, at backend startup, so a typo stops the process
    with a message naming the bad value instead of silently requesting different tracking than intended."""


#: Operator-facing tracking source names -> wire types, derived from the protocol enum like ``_PLUGIN_NAMES``.
_SOURCE_NAMES: dict[str, p.SourceType] = {member.name.lower(): member for member in p.SourceType}


def _parse_sources(raw: str | None) -> tuple[p.SourceType, ...] | None:
    """Parses EGOCAP_CAPTURE_SOURCES: a comma-separated list of tracking source names (``head``,
    ``controllers``, ``hands``; case-insensitive), or ``none`` for a cameras-only session that needs no
    headset and no IsaacCapture. Returns ``None`` when unset or blank, meaning "keep the default sources"
    -- an empty selection must be asked for by name, never reached by leaving the variable blank."""
    if raw is None or raw.strip() == "":
        return None
    if raw.strip().lower() == "none":
        return ()
    sources: list[p.SourceType] = []
    for item in raw.split(","):
        name = item.strip().lower()
        if not name:
            raise SourceConfigError(f"EGOCAP_CAPTURE_SOURCES has an empty entry: {raw!r}")
        source = _SOURCE_NAMES.get(name)
        if source is None:
            raise SourceConfigError(
                f"EGOCAP_CAPTURE_SOURCES entry {item.strip()!r} is not a known source; "
                f"valid names: {sorted(_SOURCE_NAMES)} or 'none'"
            )
        if source in sources:
            raise SourceConfigError(f"EGOCAP_CAPTURE_SOURCES lists {name!r} more than once")
        sources.append(source)
    return tuple(sources)


def config_from_env() -> CaptureStatusConfig:
    """Reads where CaptureServer is and which sources and plugins to request; everything else keeps its fixed
    default above."""
    overrides = {}
    sources = _parse_sources(os.environ.get("EGOCAP_CAPTURE_SOURCES"))
    if sources is not None:
        overrides["sources"] = sources
    return CaptureStatusConfig(
        capture_server_url=os.environ.get("EGOCAP_CAPTURE_SERVER_URL", "ws://127.0.0.1:8082"),
        plugins=_parse_plugins(os.environ.get("EGOCAP_CAPTURE_PLUGINS")),
        **overrides,
    )


_ASSETS = Path(__file__).resolve().parents[2] / "assets" / "viser"


@dataclass(frozen=True)
class DeviceModels:
    """The model files that depict one VR device set in the 3D view: a headset and its two controllers.

    ``None`` for a controller means no model is known for it and the view falls back to a coordinate frame, so
    an operator is never shown a controller shaped like hardware they are not holding."""

    label: str
    head: Path
    left_controller: Path | None
    right_controller: Path | None


#: The VR device sets the view can depict, keyed by the operator-facing profile name. The headset and controller
#: models are authored in the tracked frames the scene places them in (head pose and GRIP pose, metres), so a
#: profile needs no per-model correction. The default is the Pico 4 Ultra because that is the headset the stations
#: run; the view still re-selects at runtime when the capture status names a different device (see
#: ``viser_scene``).
DEVICE_MODEL_PROFILES: dict[str, DeviceModels] = {
    "pico-4-ultra": DeviceModels(
        label="Pico 4 Ultra",
        head=_ASSETS / "pico-4-ultra" / "headset.glb",
        left_controller=_ASSETS / "pico-4-ultra" / "left.glb",
        right_controller=_ASSETS / "pico-4-ultra" / "right.glb",
    ),
    "pico-4": DeviceModels(
        label="Pico 4",
        # No redistributable Pico 4 headset model is shipped; the generic headset stands in for the head.
        head=_ASSETS / "headset.glb",
        left_controller=_ASSETS / "pico-4" / "left.glb",
        right_controller=_ASSETS / "pico-4" / "right.glb",
    ),
    "generic": DeviceModels(
        label="generic headset",
        head=_ASSETS / "headset.glb",
        left_controller=None,
        right_controller=None,
    ),
}
DEFAULT_DEVICE_PROFILE = "pico-4-ultra"


class VisualizationConfigError(ValueError):
    """Raised at backend startup for a malformed EGOCAP_VISER_* value, so a typo stops the process with a message
    naming the bad value instead of silently showing the wrong hardware."""


@dataclass(frozen=True)
class VisualizationConfig:
    """The Viser 3D view. On by default, and bound to loopback unless the operator says otherwise
    (Viser has no authentication of its own). Models come from the device profile shipped in ``assets/viser``
    unless a file is named explicitly; the head model's fixed scale and pose correct that file's own units and
    orientation, not tracking."""

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8085
    public_url: str | None = None
    #: Which shipped device set to depict when the capture status does not name the hardware itself.
    device: str = DEFAULT_DEVICE_PROFILE
    #: Explicit model files win over the profile's; ``None`` means "the profile's file".
    head_model: Path | None = None
    head_model_scale: float = 1.0
    head_model_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    head_model_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    left_controller_model: Path | None = None
    right_controller_model: Path | None = None

    def models_for(self, profile: str) -> DeviceModels:
        """The model files to show for ``profile``, with this configuration's explicit files overriding it."""
        base = DEVICE_MODEL_PROFILES[profile]
        # An operator-supplied file is a deliberate choice for every device; it is never swapped away at runtime.
        return DeviceModels(
            label=base.label,
            head=self.head_model or base.head,
            left_controller=self.left_controller_model or base.left_controller,
            right_controller=self.right_controller_model or base.right_controller,
        )


def _parse_device_profile(raw: str | None) -> str:
    """Parses EGOCAP_VISER_DEVICE (case-insensitive, whitespace ignored); unset or blank means the default."""
    if raw is None or raw.strip() == "":
        return DEFAULT_DEVICE_PROFILE
    name = raw.strip().lower()
    # An unknown name fails fast with the accepted spellings rather than drawing a guessed device.
    if name not in DEVICE_MODEL_PROFILES:
        raise VisualizationConfigError(
            f"EGOCAP_VISER_DEVICE {raw.strip()!r} is not a known device profile; valid names: "
            f"{sorted(DEVICE_MODEL_PROFILES)}"
        )
    return name


def visualization_from_env() -> VisualizationConfig:
    """Reads the 3D view's operator settings; unset variables keep the defaults above. The view is on by
    default -- set EGOCAP_VISER_ENABLED to one of "0"/"false"/"no"/"off" to turn it off for a deployment
    that doesn't want it. Any other non-"1"/"true"/"yes" value is rejected rather than silently treated
    as "on", so a typo can't leave an unauthenticated server running unnoticed."""
    env = os.environ.get
    enabled_raw = env("EGOCAP_VISER_ENABLED", "1").strip().lower()
    if enabled_raw in ("0", "false", "no", "off"):
        # off: the other settings are not read, so a malformed one cannot matter
        return VisualizationConfig(enabled=False)
    if enabled_raw not in ("1", "true", "yes"):
        raise ValueError(f"EGOCAP_VISER_ENABLED={enabled_raw!r} is not a recognized on/off value")
    return VisualizationConfig(
        enabled=True,
        host=env("EGOCAP_VISER_HOST", "127.0.0.1"),
        port=int(env("EGOCAP_VISER_PORT", "8085")),
        public_url=env("EGOCAP_VISER_PUBLIC_URL") or None,
        device=_parse_device_profile(env("EGOCAP_VISER_DEVICE")),
        head_model=Path(model) if (model := env("EGOCAP_VISER_HEAD_MODEL")) else None,
        head_model_scale=float(env("EGOCAP_VISER_HEAD_MODEL_SCALE", "1.0")),
    )

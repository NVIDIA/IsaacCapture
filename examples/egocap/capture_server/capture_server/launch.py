# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Launch composition: the ONE place that selects mock vs real
acquisition dependencies and binds mock-only options (scenario/seed/
faults) into them: ``ServerConfig.mock`` selects dependencies once, here --
adapters themselves do not branch on mock/real, import mock scenarios, test
mock classes or silently fall back to generated values. Real setup never
falls back to the mock, and each side's SDK imports happen only on its own
path: the mock SDK files are imported inside :func:`build_mock_source_factory`,
so real setup never loads them, and mock setup never loads a hardware SDK
or reads ``ISAAC_ROOT``.

Everything downstream of :func:`build_mock_source_factory` -- the
:class:`~capture_server.acquisition.acquisition.Acquisition` it returns,
the two concrete adapters it composes, and all of ``server.py`` -- is
mode-agnostic; this module is the only one (outside the two mock SDK
files themselves) that imports
:mod:`capture_server.mock.mock_isaac_capture`/
:mod:`capture_server.mock.mock_depthai` directly, and the only one that
knows what "scenario", "seed" and fault names mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable

from . import protocol as p
from .acquisition import Acquisition, DepthAIAdapter, IsaacCaptureAdapter, TrackingSelection
from .server import ServerConfig, StationCamera

if TYPE_CHECKING:
    from .gmsl_ipc.feed import SlotMapper, StreamInfo
    from .replay.episode import Episode

__all__ = [
    "build_gmsl_source_factory",
    "build_mock_source_factory",
    "build_real_source_factory",
    "build_replay_source_factory",
    "build_source_factory",
    "gmsl_station_cameras",
    "replay_station_cameras",
]

SourceFactory = Callable[[p.BeginRequest, "tuple[p.CameraStreamConfig, ...]"], Acquisition]

# Replayed frames are re-encoded from the recording at the stream's JPEG quality; this is the station
# default the mock also uses, since a recording carries no JPEG quality of its own.
_REPLAY_JPEG_QUALITY = 80


def _group_selected(begin: p.BeginRequest, source_type: p.SourceType) -> bool:
    return source_type in begin.tracking.sources


def _tracking_adapter(
    begin: p.BeginRequest, *, sdk: ModuleType, session_factory: Callable[[Any], Any]
) -> IsaacCaptureAdapter | None:
    """The IsaacCapture adapter for the requested tracking outputs, or None when none are requested.

    Shared by mock and real composition: only ``sdk`` (the source/config construction surface) and
    ``session_factory`` differ. server.py's Begin validation rejects ``tracking.sources=[]`` combined
    with any enabled plugin before a factory is called, so tracking can be gated on ``sources`` alone.
    """
    if not begin.tracking.sources:
        return None
    manus_plugin_request = next(
        (plugin for plugin in begin.plugins if plugin.plugin == p.PluginType.MANUS and plugin.enabled), None
    )
    selection = TrackingSelection(
        head=_group_selected(begin, p.SourceType.HEAD),
        controller_left=_group_selected(begin, p.SourceType.CONTROLLERS),
        controller_right=_group_selected(begin, p.SourceType.CONTROLLERS),
        hand_left=_group_selected(begin, p.SourceType.HANDS),
        hand_right=_group_selected(begin, p.SourceType.HANDS),
        manus_enabled=manus_plugin_request is not None,
        manus_required=manus_plugin_request.required if manus_plugin_request is not None else False,
    )
    return IsaacCaptureAdapter(
        sdk=sdk, session_factory=session_factory, selection=selection, rate_hz=begin.tracking.rate_hz
    )


def build_mock_source_factory(
    *,
    scenario: str,
    seed: int,
    ic_faults: set[str] | frozenset[str] = frozenset(),
    camera_faults: set[str] | frozenset[str] = frozenset(),
) -> SourceFactory:
    """The default ``CaptureServer`` source factory, wired to the two mock
    SDK files. ``ic_faults``/``camera_faults`` are captured here, once,
    from CLI/test configuration -- "[e]xisting documented local CLI
    controls may delegate to the two mocks without becoming protocol
    fields" -- and baked into each new mock session/pipeline this
    composition builds; they are never read or stored by ``ServerConfig``
    or either adapter."""
    # Imported here, not at module level, so choosing real acquisition never loads the mocks.
    from .mock import mock_depthai, mock_isaac_capture

    def source_factory(begin: p.BeginRequest, cameras: tuple[p.CameraStreamConfig, ...]) -> Acquisition:
        tracking = _tracking_adapter(
            begin,
            sdk=mock_isaac_capture,
            session_factory=partial(
                mock_isaac_capture.TeleopSession, scenario=scenario, seed=seed, faults=set(ic_faults)
            ),
        )

        camera_adapters = tuple(
            DepthAIAdapter(
                sdk=mock_depthai,
                pipeline_factory=partial(
                    mock_depthai.Pipeline,
                    device_id=f"mock-oakd-{camera.stream_id}",
                    seed=seed + camera.stream_id,
                    faults=set(camera_faults),
                ),
                camera=camera,
            )
            for camera in cameras
        )
        return Acquisition(tracking=tracking, cameras=camera_adapters)

    # Lets CaptureServer reject a launch-built factory that disagrees with ServerConfig.mock;
    # untagged custom factories are not checked.
    source_factory.mock = True  # type: ignore[attr-defined]
    return source_factory


def _real_camera_adapters(
    config: ServerConfig, cameras: tuple[p.CameraStreamConfig, ...]
) -> tuple[DepthAIAdapter, ...]:
    """One real OAK-D adapter per requested camera, each bound to its station-configured physical device."""
    if not cameras:
        return ()
    # Every camera needs a station device: report that configuration error before importing the optional SDK.
    stations = {}
    for camera in cameras:
        station = (config.cameras or {}).get(camera.name)
        if station is None:
            raise ValueError(
                f"camera {camera.name!r} has no station device; configure it with --camera NAME=DEVICE_ID"
            )
        stations[camera.name] = station
    import depthai as dai

    return tuple(
        DepthAIAdapter(
            sdk=dai,
            pipeline_factory=partial(
                _open_pipeline, dai, stations[camera.name].device_id, stations[camera.name].max_usb_speed
            ),
            camera=camera,
        )
        for camera in cameras
    )


def _open_pipeline(dai: ModuleType, device_id: str, max_usb_speed: str | None) -> Any:
    """A pipeline bound to exactly the OAK-D with this device ID (never the first one found)."""
    info = dai.DeviceInfo(device_id)
    if max_usb_speed is None:
        return dai.Pipeline(dai.Device(info))
    return dai.Pipeline(dai.Device(info, maxUsbSpeed=dai.UsbSpeed.__members__[max_usb_speed]))


def build_real_source_factory(config: ServerConfig) -> SourceFactory:
    """The real-acquisition source factory: IsaacCapture tracking from ``$ISAAC_ROOT``.

    The SDK is resolved when a session first asks for tracking, so a camera-only session needs no
    IsaacCapture; a resolution failure rejects that Begin with the reason. Each requested camera is a real
    OAK-D selected by the station's device ID (``ServerConfig.cameras``); nothing is ever synthesized.
    """

    def source_factory(begin: p.BeginRequest, cameras: tuple[p.CameraStreamConfig, ...]) -> Acquisition:
        """Build this session's real acquisition: tracking only if requested, one OAK-D per camera."""
        from .acquisition.isaac_sdk import load_isaac_capture

        sdk = load_isaac_capture() if begin.tracking.sources else None
        tracking = _tracking_adapter(begin, sdk=sdk, session_factory=sdk.TeleopSession) if sdk else None
        return Acquisition(tracking=tracking, cameras=_real_camera_adapters(config, cameras))

    source_factory.mock = False  # type: ignore[attr-defined]
    return source_factory


def replay_station_cameras(episode: Episode) -> dict[str, StationCamera]:
    """One station entry per recorded camera, so the server resolves each requested stream to the recording's
    own size and rate (the adapter rejects a frame whose size differs from the accepted stream's) and rejects
    any camera name the episode does not have. The device ID is the MP4 path: stable and unique per camera."""
    return {
        name: StationCamera(str(track.path), track.width, track.height, track.fps, _REPLAY_JPEG_QUALITY)
        for name, track in episode.cameras.items()
    }


def build_replay_source_factory(episode: Episode) -> SourceFactory:
    """The replay source factory: recorded tracking rows and camera frames from *episode*, through the same
    two adapters the mock and real sources use.

    Every session gets one shared playback timeline, so its tracking and camera streams start, align and
    loop together. A requested camera the episode lacks fails construction naming the recorded ones; the
    server normally rejects such a name earlier through :func:`replay_station_cameras`.
    """
    # Imported here, like the mock SDK files, so choosing mock or real acquisition never loads replay.
    from .replay import replay_depthai, replay_isaac_capture
    from .replay.playback import PlaybackClock

    def source_factory(begin: p.BeginRequest, cameras: tuple[p.CameraStreamConfig, ...]) -> Acquisition:
        """Build this session's replay acquisition on one new shared timeline."""
        playback = PlaybackClock(episode.period)
        tracking = _tracking_adapter(
            begin,
            sdk=replay_isaac_capture,
            session_factory=partial(replay_isaac_capture.TeleopSession, episode=episode, playback=playback),
        )
        camera_adapters = []
        for camera in cameras:
            track = episode.cameras.get(camera.name)
            if track is None:
                raise ValueError(
                    f"camera {camera.name!r} is not in the recorded episode; "
                    f"recorded cameras: {sorted(episode.cameras)}"
                )
            camera_adapters.append(
                DepthAIAdapter(
                    sdk=replay_depthai,
                    pipeline_factory=partial(replay_depthai.Pipeline, track=track, playback=playback),
                    camera=camera,
                )
            )
        return Acquisition(tracking=tracking, cameras=tuple(camera_adapters))

    # Replay is recorded data, not live hardware: it is served under ServerConfig.mock=True and reported as
    # CaptureMode.MOCK, so the server's mock/real consistency check treats it as a mock-side factory.
    source_factory.mock = True  # type: ignore[attr-defined]
    return source_factory


@dataclass(frozen=True)
class GmslCamera:
    """One GMSL camera as the operator configured it: the feed stream it reads and the size it is served at."""

    stream: str
    #: Served ``(width, height)``; None serves the sensor's own size.
    output_size: tuple[int, int] | None = None


def gmsl_station_cameras(
    cameras: dict[str, GmslCamera], catalog: dict[str, StreamInfo], *, fps: float, jpeg_quality: int
) -> dict[str, StationCamera]:
    """One station entry per configured camera, sized from its output size or the feed's own stream size.

    The server validates a Begin's camera names and sizes against these entries, and the adapter rejects a
    frame whose size differs from the accepted stream's, so the entry must state the served size exactly.
    A camera naming a stream the feed does not publish is a configuration error, reported here at setup.
    """
    stations: dict[str, StationCamera] = {}
    for name, camera in cameras.items():
        info = catalog.get(camera.stream)
        if info is None:
            raise ValueError(
                f"camera {name!r} reads feed stream {camera.stream!r}, which the frame feed does not publish; "
                f"published streams: {sorted(catalog)}"
            )
        width, height = camera.output_size or (info.width, info.height)
        stations[name] = StationCamera(camera.stream, width, height, fps, jpeg_quality)
    return stations


def build_gmsl_source_factory(
    socket_path: str,
    cameras: dict[str, GmslCamera],
    stations: dict[str, StationCamera],
    *,
    slot_mapper: SlotMapper | None = None,
) -> SourceFactory:
    """The live-station source factory: real IsaacCapture tracking (from ``$ISAAC_ROOT``, only when a session
    requests tracking) plus GMSL cameras read from the receiver's frame feed.

    Each session opens one feed subscription covering exactly the streams its cameras read, closed when the
    acquisition is released; the feed is a broadcast publisher, so this never disturbs other consumers.
    ``slot_mapper`` defaults to the CUDA IPC mapper; tests inject an in-process one.
    """
    # Imported here, like the other source families, so choosing mock, replay or OAK-D never loads this path.
    from .gmsl_ipc import FeedClient
    from .gmsl_ipc import source as gmsl_source

    def source_factory(begin: p.BeginRequest, requested: tuple[p.CameraStreamConfig, ...]) -> Acquisition:
        """Build this session's acquisition: real tracking if requested, one feed subscription for its cameras."""
        from .acquisition.isaac_sdk import load_isaac_capture

        sdk = load_isaac_capture() if begin.tracking.sources else None
        tracking = _tracking_adapter(begin, sdk=sdk, session_factory=sdk.TeleopSession) if sdk else None
        if not requested:
            return Acquisition(tracking=tracking, cameras=())
        for camera in requested:
            if camera.name not in cameras:
                raise ValueError(
                    f"camera {camera.name!r} is not a configured GMSL camera; configured: {sorted(cameras)}"
                )
        mapper = slot_mapper
        if mapper is None:
            from .gmsl_ipc.cuda_slots import CudaSlotMapper

            mapper = CudaSlotMapper()
        streams = tuple(dict.fromkeys(cameras[c.name].stream for c in requested))
        feed = FeedClient(socket_path, streams, slot_mapper=mapper)
        feed.connect()
        adapters = tuple(
            DepthAIAdapter(
                sdk=gmsl_source,
                pipeline_factory=partial(
                    gmsl_source.Pipeline,
                    feed=feed,
                    stream=cameras[camera.name].stream,
                    output_size=(stations[camera.name].width, stations[camera.name].height),
                    jpeg_quality=stations[camera.name].jpeg_quality,
                ),
                camera=camera,
            )
            for camera in requested
        )
        return _FeedAcquisition(tracking=tracking, cameras=adapters, feed=feed)

    # Live cameras and live tracking: this is real acquisition, reported as LIVE.
    source_factory.mock = False  # type: ignore[attr-defined]
    return source_factory


class _FeedAcquisition(Acquisition):
    """An acquisition that also owns the session's feed subscription and closes it after its cameras."""

    def __init__(self, *, tracking: Any, cameras: tuple[DepthAIAdapter, ...], feed: Any) -> None:
        super().__init__(tracking=tracking, cameras=cameras)
        self._feed = feed

    async def close(self) -> Any:
        """Release adapters first (they stop polling), then drop the subscription and unmap the slots."""
        try:
            return await super().close()
        finally:
            self._feed.close()


def build_source_factory(
    config: ServerConfig,
    *,
    scenario: str = "demo",
    seed: int = 0,
    faults: frozenset[str] = frozenset(),
) -> SourceFactory:
    """Select the acquisition implementation once, from ``config.mock``.

    Mock-only options (scenario/seed/faults) reach only the mock composition.
    ``ValueError`` names an unknown mock scenario.
    """
    if not config.mock:
        return build_real_source_factory(config)
    from .mock.mock_isaac_capture import SCENARIOS

    if scenario not in SCENARIOS:
        raise ValueError(f"unknown mock scenario {scenario!r}; choose one of {', '.join(SCENARIOS)}")
    return build_mock_source_factory(
        scenario=scenario, seed=seed, ic_faults=set(faults), camera_faults=set(faults)
    )

"""FastAPI lifetime, the one status endpoint, and static hosting of the
built React app.
"""

from __future__ import annotations

import logging
import os
import secrets
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from egocap.core.data_root import calibration_dir
from recording_service import (
    CalibrationAsset,
    ConfigurationError,
    QcConfig,
    ServiceConfig,
    VideoSourceSpec,
)

from .capture_preview import MAX_CAMERA_JPEG_BYTES
from .capture_status import CaptureStatusMonitor
from .config import VisualizationConfig, config_from_env, visualization_from_env
from .preview_routes import router as preview_router
from .recording_adapter import (
    MANUS_GLOVE_SPEC,
    PICO_CONTROLLER_SPEC,
    RecordingAdapterError,
    validate_tracking_descriptor,
)
from .recording_command import RecordingContext
from .recording_session import RecordingSessionCoordinator, TaskContext
from .recordings import router as recordings_router
from .station_config import ResolvedStation, StationConfigError, load_station_config, station_config_path_from_env
from .task_file_routes import router as task_file_router
from .viser_scene import VisualizationScene

logger = logging.getLogger(__name__)

# The built frontend's output directory (`npm run build` -> `frontend/dist/`).
# Not present in a source checkout until the frontend is built once; see
# frontend/README.md.
_FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
# The task catalogs shipped with the application (read-only; see task_files.py). EGOCAP_TASK_FILES_DIR points the
# API at another directory, for example the one the recorder reads its task library from.
_DEFAULT_TASK_FILE_ROOT = Path(__file__).resolve().parents[2] / "data" / "taskfiles"

# ONE total memory budget covering every queued/in-flight copy this
# pipeline holds AT ONCE, together: the per-take ingress's own raw queue,
# one raw+decoded pair held live during conversion, the real service's own
# decoded-sample queue, AND the service's own writer in-flight copy (the
# one sample it has dequeued and is actively encoding/writing). Sized
# independently, each of these would look individually reasonable while
# still summing to far more memory than intended -- see
# _recording_config_factory below for how the decoded-queue capacity is
# derived from what's left after the raw side's own reservation.
_RECORDING_QUEUE_MEMORY_BUDGET_BYTES = 256 * 1024 * 1024

# RecordingIngress's own raw-item queue, per take -- items queued here are
# the UNconverted CaptureFrame/CameraImage values. It is bounded by bytes,
# not item count: _INGRESS_QUEUE_BYTES of actual compressed JPEG / tracking
# blob (a typical camera frame is a few hundred KB, so this holds seconds
# of 60 Hz tracking plus two or three 30 fps cameras, where an 8-slot count
# held ~50 ms). _INGRESS_QUEUE_CAPACITY is only a safety cap on item count.
# The reservation adds one MAX_CAMERA_JPEG_BYTES for the raw item held live
# in the executor during conversion (capture_preview.py's own cap, not a
# locally-assumed "typical" size). Fixed at coordinator-construction time,
# before any camera is necessarily known, so unlike the decoded queue below
# it cannot be sized from real resolution.
_INGRESS_QUEUE_BYTES = 64 * 1024 * 1024
_INGRESS_QUEUE_CAPACITY = 4096
_INGRESS_RESERVED_BYTES = _INGRESS_QUEUE_BYTES + MAX_CAMERA_JPEG_BYTES


def _recording_output_root(station_data_dir: Path | None = None) -> Path:
    """Server-owned recording output location -- an operator/deployment
    setting, never something the browser supplies (see config.py's own
    docstring for the same rule applied to capture configuration). Resolves
    from the station YAML's ``recording.data_dir`` when configured, else the
    fixed development default."""
    if station_data_dir is not None:
        return station_data_dir
    return Path("./recordings").expanduser()


# Per-station calibration files copied into every new session (and, once there, into each
# routed group) when present at EGOCAP_CALIBRATION_DIR. All optional: a station that hasn't
# captured one yet must not be blocked from recording -- see ServiceConfig.calibration_assets'
# own required=False handling for the same reason the old per-manager copy only ever warned.
_STATION_CALIBRATION_FILES = (
    "head_camera_intrinsics.json",
    "head_camera_extrinsics.json",
    "wrist_calibration_offsets.json",
    "left_wrist_camera_intrinsics.json",
    "right_wrist_camera_intrinsics.json",
)


def _station_calibration_assets() -> tuple[CalibrationAsset, ...]:
    """The station's calibration files, resolved fresh from EGOCAP_CALIBRATION_DIR each call."""
    source_dir = calibration_dir()
    return tuple(
        CalibrationAsset(name=name, path=source_dir / name, required=False) for name in _STATION_CALIBRATION_FILES
    )


def _recording_config_factory(monitor: CaptureStatusMonitor, station_label: str | None, output_root: Path):
    """Builds the real ``ServiceConfig`` lazily, read fresh each time the
    coordinator actually opens/reopens a session (not once at import time)
    -- so it reflects whichever cameras CaptureServer has ACTUALLY accepted
    by then, not a fixed guess made before the capture attempt connected.
    Declares ``intrinsics=False`` for every camera -- per-camera
    intrinsics exist on the accepted descriptor but are not yet threaded
    through the conversion path into each submitted ``VideoSample``
    (``recording_adapter.convert_camera_image``'s ``intrinsics`` param is
    always ``None`` today); declaring a schema that doesn't require them is
    the honest choice until that plumbing is added, rather than manufacturing
    calibration values the backend doesn't actually have."""

    def factory() -> ServiceConfig:
        video_sources = tuple(
            VideoSourceSpec(cam.name, cam.width, cam.height, cam.fps, intrinsics=False)
            for cam in monitor.camera_descriptors
        )
        # A supported profile must record actual tracking AND the
        # selected cameras -- pico_controller and every configured camera
        # are required for a KEPT result. manus_glove is NOT required:
        # a hand legitimately going briefly untracked (the whole-row
        # omission policy in recording_adapter.py) must not by itself
        # invalidate an otherwise-good take.
        required = (PICO_CONTROLLER_SPEC.name, *(v.name for v in video_sources))
        config = ServiceConfig(
            output_root=output_root,
            numeric_sources=(PICO_CONTROLLER_SPEC, MANUS_GLOVE_SPEC),
            video_sources=video_sources,
            qc=QcConfig(required_sources=required),
            calibration_assets=_station_calibration_assets(),
            station_label=station_label,
        )
        largest_decoded_frame_bytes = max((cam.width * cam.height * 3 for cam in video_sources), default=0)
        if largest_decoded_frame_bytes > 0:
            # What's left of the total budget after the raw ingress side's
            # own reservation (above), minus TWO more decoded-frame copies
            # held live at once: one in the executor during conversion,
            # one in the real service's own writer (the sample it has
            # already dequeued and is actively encoding). Genuinely bounds
            # the combined worst case at _RECORDING_QUEUE_MEMORY_BUDGET_BYTES.
            decoded_queue_budget = (
                _RECORDING_QUEUE_MEMORY_BUDGET_BYTES - _INGRESS_RESERVED_BYTES - 2 * largest_decoded_frame_bytes
            )
            capacity = decoded_queue_budget // largest_decoded_frame_bytes
            if capacity < 1:
                # A silent floor here would claim a cap that doesn't
                # actually hold -- this profile's own geometry genuinely
                # cannot fit within the stated budget; refuse rather than
                # pretend.
                raise ConfigurationError(
                    f"camera resolution ({largest_decoded_frame_bytes} bytes/decoded frame) cannot fit even "
                    f"one queued frame within the {_RECORDING_QUEUE_MEMORY_BUDGET_BYTES}-byte recording queue "
                    "memory budget"
                )
            config = replace(config, queue_capacity=capacity)
        # No camera configured keeps the dataclass default -- numeric-only
        # samples are tiny, so that default is already a conservative bound.
        return config

    return factory


#: How old a backend-local-received tracking/camera sample may be and still
#: count as "ready" -- matches preview_routes.py's own staleness threshold
#: for the same preview slots (that constant isn't exported, so this is a
#: local equivalent, not a second, differently-tuned policy).
_READINESS_STALE_SECONDS = 2.0


def _recording_readiness_check(monitor: CaptureStatusMonitor, station_config_error: str | None = None):
    """Real backend-local capture facts, injected into the coordinator --
    it never imports CaptureStatusMonitor/CaptureClient itself (see
    recording_session.py's own module docstring). Checked at start_episode
    admission (BEFORE storage ever opens -- an incompatible descriptor
    must never reach the point of recording a whole take's worth of zero
    tracking data before anyone notices), the post-await activation
    recheck inside _start_episode_impl, AND continuously on every accepted
    capture callback while a take is active (see
    RecordingSessionCoordinator._check_still_ready_while_recording), so a
    source that goes stale or becomes incompatible at any of those points
    is caught. Requires a tracking descriptor this adapter can actually
    decode, capture running, and a tracking/camera sample actually
    received within the staleness window -- not merely "ever arrived
    once."

    ``station_config_error``, when set, unconditionally blocks new recording
    admission with that reason -- a configured-but-invalid station YAML
    must never silently fall back to a development default, but it must
    also never prevent capture preview from running."""

    def check() -> tuple[bool, str | None]:
        if station_config_error is not None:
            return False, station_config_error
        if monitor.tracking_descriptor is None:
            return False, "tracking source not yet accepted"
        try:
            validate_tracking_descriptor(monitor.tracking_descriptor)
        except RecordingAdapterError as exc:
            return False, f"tracking descriptor unsupported for recording: {exc}"
        if monitor.native.status is None or not monitor.native.status.running:
            return False, "capture is not running"
        now = time.monotonic()
        tracking = monitor.preview.tracking()
        if tracking is None or (now - tracking.received_monotonic) > _READINESS_STALE_SECONDS:
            return False, "tracking source is stale or has no sample yet"
        cameras = monitor.camera_descriptors
        if not cameras:
            return False, "no camera selected"
        for cam in cameras:
            slot = monitor.preview.camera(cam.name)
            if slot is None or (now - slot.received_monotonic) > _READINESS_STALE_SECONDS:
                return False, f"camera {cam.name!r} is stale or has no sample yet"
        return True, None

    return check


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Creates and starts the one CaptureStatusMonitor for the station,
    independent of any browser connecting. Multiple viewers share this
    single attempt; closing a browser tab never stops it -- only
    application shutdown does, via an explicit, confirmed stop().

    Also constructs the one RecordingSessionCoordinator (owning the real
    RecordingService) and the RecordingContext wrapping it, the same
    ownership pattern -- every WS connection shares it rather than getting
    its own, so a disconnect/reconnect never loses the active episode/
    task/totals and two simultaneous viewers can never collide on episode
    identity. The coordinator is NOT opened here -- it opens lazily on the
    first real start_episode attempt (see recording_command.py), so its
    config factory reads real, by-then-accepted camera descriptors instead
    of a fixed guess made before capture has had a chance to connect.

    Construction order resolves the real circular dependency (the
    coordinator's config/readiness closures need the monitor; the monitor's
    recording consumer needs the coordinator): the monitor is built first
    with no recording consumer, then the coordinator is built against that
    real monitor, then attached via ``set_recording``.

    Also records the fixed task-catalog directory for the read-only task-file API,
    independently of capture: a missing directory is logged and reported by that API
    (503) while the capture preview still starts.

    A deployment station YAML (``EGOCAP_STATION_CONFIG``, see station_config.py), when
    configured, is loaded exactly once here and seeds the station's label, initial
    operator/instruction and recording output root through the same existing
    config/context mechanisms below -- not a second configuration path. A
    configured-but-invalid file is a real configuration failure, but NOT a backend
    startup crash: capture preview has nothing to do with recording configuration and
    must keep working, so the failure is instead captured as a readiness-check reason
    that blocks new recording admission (see ``_recording_readiness_check``) -- never
    silently replaced by the unset/development default. Unset ``EGOCAP_STATION_CONFIG``
    preserves today's configuration-free development workflow."""
    task_file_root = Path(os.environ.get("EGOCAP_TASK_FILES_DIR") or _DEFAULT_TASK_FILE_ROOT)
    app.state.task_file_root = task_file_root
    if not task_file_root.is_dir():
        logger.error("task file directory %s is missing; /api/task-files is unavailable", task_file_root)
    resolved_station: ResolvedStation | None = None
    # The PUBLIC reason exposed via readiness rejection -- deliberately generic: the real
    # StationConfigError text can contain the configured file's path and raw YAML-parser
    # context, which stays backend-only (see the logged line below).
    station_config_error: str | None = None
    try:
        resolved_station = load_station_config(station_config_path_from_env())
    except StationConfigError as exc:
        station_config_error = "station configuration is invalid; see backend logs"
        logger.error("station configuration failed; new recording is blocked: %s", exc)
    config = config_from_env()
    # The station YAML's label wins when configured, else the fixed development default.
    if resolved_station is not None and resolved_station.station_label:
        config = replace(config, station_label=resolved_station.station_label)
    # The station YAML's video_sources become the cameras requested from CaptureServer -- an
    # explicit empty video_sources list is respected as "no cameras", same as the unconfigured
    # default and the configured-but-invalid-file case above (resolved_station stays None, so
    # config.cameras keeps its empty default -- capture preview then shows no camera tiles,
    # it is not unaffected). Each requested name must already be known to the CaptureServer
    # instance actually running (its own --camera list, mock or real) -- an unknown name rejects
    # the WHOLE Begin, not just that one camera, which also takes tracking preview down with it.
    if resolved_station is not None:
        config = replace(config, cameras=resolved_station.cameras)
    # task_text seeds the committed next-task instruction. operator seeds ONLY a prefill
    # default for the frontend's entry field (via recording_command.py's snapshot
    # "default_operator") -- start_episode's own operator parameter always replaces
    # context.operator with the browser command's own (required) operator before a take can
    # ever start (see recording_session.py's start_episode), so this seeded value is never
    # itself used to admit or attribute a take.
    initial_context = (
        TaskContext(
            task_text=resolved_station.task_description or "",
            operator=resolved_station.operator_name or "",
        )
        if resolved_station is not None
        else None
    )
    # Resolved exactly ONCE here, then shared verbatim with both the recording factory and
    # recordings.py's browse/download routes (via app.state) -- the recorder and the recordings
    # UI must never end up pointed at two different roots mid-process.
    recording_output_root = _recording_output_root(resolved_station.data_dir if resolved_station else None)
    app.state.recording_output_root = recording_output_root
    monitor = CaptureStatusMonitor(config, backend_instance_id=str(uuid.uuid4()))
    coordinator = RecordingSessionCoordinator(
        _recording_config_factory(monitor, config.station_label, recording_output_root),
        queue_capacity=_INGRESS_QUEUE_CAPACITY,
        queue_bytes=_INGRESS_QUEUE_BYTES,
        readiness_check=_recording_readiness_check(monitor, station_config_error),
        initial_context=initial_context,
    )
    monitor.set_recording(coordinator)
    await monitor.start()
    app.state.monitor = monitor
    # Generated once per backend process start (a restart is the only way
    # to issue a new one) and printed here -- this console line is the
    # distribution mechanism, which the operator reads off the machine
    # they started the backend on. A plain print, not logger.info: the
    # documented `uv run uvicorn` launch configures no logging, so a log
    # call here would be silently dropped.
    control_token = secrets.token_urlsafe(32)
    print(f"Recording control token for this run: {control_token}", file=sys.stderr, flush=True)
    auto_fill_operator_name = resolved_station.auto_fill_operator_name if resolved_station is not None else False
    app.state.recording_context = RecordingContext(
        coordinator=coordinator, control_token=control_token, auto_fill_operator_name=auto_fill_operator_name
    )
    visualization: VisualizationScene | None = None
    try:
        visualization = _optional_visualization(monitor)
        app.state.visualization = visualization
        await visualization.start()  # never raises: a failed start only reports the view unavailable
        yield
    finally:
        try:
            if visualization is not None:
                await visualization.stop()
        finally:
            # The recording coordinator settles/closes before the capture
            # owner stops -- it still needs the live capture connection to
            # retire any in-flight take cleanly. An unreleased close is
            # surfaced, not silently discarded.
            close_result = await coordinator.close(final=True)
            if not close_result.released:
                logger.error("recording coordinator did not fully release on shutdown: %s", close_result.issues)
            await monitor.stop()


def _optional_visualization(monitor: CaptureStatusMonitor) -> VisualizationScene:
    """The optional 3D view; settings that cannot be read leave it unavailable instead of stopping the app."""
    try:
        return VisualizationScene(monitor, visualization_from_env())
    except Exception as exc:
        logger.error("3D view settings are invalid: %s", exc)
        scene = VisualizationScene(monitor, VisualizationConfig(enabled=True))
        scene.mark_unavailable(f"3D view settings are invalid: {exc}")
        return scene


app = FastAPI(lifespan=lifespan)
app.include_router(preview_router)
app.include_router(task_file_router)
# PoC demo integration: browse/download recorded files, off unless
# EGOCAP_RECORDINGS_ENABLED=1. Not part of the final system design -- see
# recordings.py's own module docstring.
app.include_router(recordings_router)


@app.get("/api/capture-status")
async def get_capture_status(request: Request) -> JSONResponse:
    """Returns the one current status value. Capture failure is still a
    normal HTTP 200 with an actual failed/stopped status in the body -- a
    healthy HTTP server is distinct from healthy capture."""
    monitor: CaptureStatusMonitor = request.app.state.monitor
    return JSONResponse(monitor.snapshot().to_json(), headers={"Cache-Control": "no-store"})


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def api_not_found(path: str) -> JSONResponse:
    """Any /api path this module didn't explicitly register above is a
    real 404, not the SPA shell -- registered before the static mount so
    it takes precedence over that mount's own HTML fallback."""
    return JSONResponse({"detail": "Not Found"}, status_code=404)


# Static hosting of the built frontend, mounted last so it never shadows
# /api routes above. Only mounted if the frontend has actually been built
# (a source checkout before `npm run build` can still run the backend
# standalone, e.g. for API-only development/tests).
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")

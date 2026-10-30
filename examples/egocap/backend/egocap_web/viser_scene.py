"""A Viser 3D view of the tracked head, controllers and gloves, embedded by the page as an iframe.

One :class:`VisualizationScene` per backend process owns one ``ViserServer`` and one periodic sampler task. The
sampler reads the monitor's existing latest tracking slot (no second CaptureClient), decodes it with the accepted
descriptor, and updates fixed scene nodes. Everything here is presentation: nothing feeds back into capture or
recording, and any failure is reported through :meth:`VisualizationScene.projection` instead of reaching the app.

Coordinate contract (checked against the accepted descriptor, never guessed): meters, right-handed Y-up, OpenXR
XYZW quaternions in, Viser WXYZ out, 26 OpenXR hand joints with explicit validity flags.

The glove skeleton is drawn from the reported joints exactly as they arrive: the tracking-space positions
CaptureServer carries (for the Manus plugin these are already injected as ``root * local`` with the
controller-derived root, so they move with the controller). Nothing is re-mounted, offset or rotated here, so a
producer that does not follow any particular controller relationship is still drawn where it says it is. The
controller mesh follows the GRIP pose.

Device models: the headset and controllers drawn are the ones of a shipped device profile
(:data:`~egocap_web.config.DEVICE_MODEL_PROFILES`). The configured profile is the default; when the capture status
describes the headset by name (for example "Pico 4 Ultra headset"), the scene switches to that device's models so
an operator always sees the hardware actually in use rather than a stand-in.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.transform import Rotation

from capture_server import protocol as p

from .config import DEVICE_MODEL_PROFILES

if TYPE_CHECKING:
    from .capture_status import CaptureStatusMonitor
    from .config import VisualizationConfig

logger = logging.getLogger(__name__)

SAMPLE_INTERVAL_S = 0.1  # the 10 Hz preview cadence
STALE_AFTER_S = 2.0  # backend-local receipt age after which tracking is treated as lost
STOP_TIMEOUT_S = 5.0
STARTUP_TIMEOUT_S = 10.0  # how long the app waits for the optional view to start before serving without it

SIDES = ("left", "right")

# Where a browser's camera starts: behind and above the tracked head, looking at the head and hands. A view
# preference only; the client can orbit freely and tracking geometry is unaffected.
INITIAL_CAMERA_POSITION = (0.6, 1.55, 0.75)
INITIAL_CAMERA_LOOK_AT = (0.0, 1.3, -0.3)

# OpenXR hand joints: 0 PALM, 1 WRIST, thumb 2-5, then index, middle, ring and little 5 joints each.
_WRIST = 1
_CHAINS = ((2, 3, 4, 5), (6, 7, 8, 9, 10), (11, 12, 13, 14, 15), (16, 17, 18, 19, 20), (21, 22, 23, 24, 25))
HAND_BONES = tuple(pair for chain in _CHAINS for pair in zip((_WRIST, *chain), chain, strict=False))

_COLORS = {"left": (60, 140, 255), "right": (255, 140, 60)}

# The tracking fields this view reads, with the dtype and shape the ic_tracking_v1 layout must declare for each.
_HAND_FIELDS = (
    ("present", "u1", ()),
    ("joint_positions", "f4", (26, 3)),
    ("joint_valid", "u1", (26,)),
)
_CONTROLLER_FIELDS = (
    ("present", "u1", ()),
    ("grip_position", "f4", (3,)),
    ("grip_orientation", "f4", (4,)),
    ("grip_is_valid", "u1", ()),
)
_HEAD_FIELDS = (
    ("present", "u1", ()),
    ("position", "f4", (3,)),
    ("orientation", "f4", (4,)),
    ("is_valid", "u1", ()),
)
_NEEDED = (
    [(f"head.{n}", d, s) for n, d, s in _HEAD_FIELDS]
    + [(f"controller_{side}.{n}", d, s) for side in SIDES for n, d, s in _CONTROLLER_FIELDS]
    + [(f"hand_{side}.{n}", d, s) for side in SIDES for n, d, s in _HAND_FIELDS]
)


# Device names as they may appear in a headset's status description, most specific first so "Pico 4 Ultra" is
# never read as a plain Pico 4. Matching is on the normalised description (lower case; spaces, hyphens and
# underscores removed).
_DEVICE_NAME_TOKENS: tuple[tuple[str, str], ...] = (
    ("pico4ultra", "pico-4-ultra"),
    ("pico4u", "pico-4-ultra"),
    ("pico4", "pico-4"),
    ("pico", "pico-4"),
    ("quest", "generic"),
    ("vive", "generic"),
    ("index", "generic"),
)
_MODEL_KEYS = ("head", "left", "right")


def detect_device_profile(devices: Any) -> str | None:
    """The shipped device profile named by the headset's status description, or None when it names no device.

    Only the headset entry decides: the controllers belong to the same product, and a description such as
    "OpenXR headset" (what the live runtime reports today) names nothing, leaving the configured default in force.
    """
    for device in devices or ():
        if getattr(device, "type", None) != "headset":
            continue
        text = "".join(ch for ch in str(getattr(device, "description", "")).lower() if ch not in " -_")
        for token, profile in _DEVICE_NAME_TOKENS:
            if token in text:
                return profile
    return None


class UnsupportedTracking(Exception):
    """The accepted tracking descriptor is not one this view can render faithfully."""


@dataclass(frozen=True)
class Layout:
    """Where each needed field sits in the tracking payload, bound once per accepted descriptor."""

    payload_size: int
    fields: dict[str, tuple[int, np.dtype, tuple[int, ...]]]


def bind_layout(stream: p.TrackingStream) -> Layout:
    """Check the descriptor's semantics and byte layout and bind the fields this view needs."""
    semantics = (
        (stream.schema, p.TrackingSchema.IC_TRACKING_V1, "schema"),
        (stream.pose_representation, p.PoseRepresentation.POSITION_QUATERNION, "pose representation"),
        (stream.quaternion_order, p.QuaternionOrder.XYZW, "quaternion order"),
        (stream.position_unit, p.PositionUnit.METERS, "position unit"),
        (stream.axes, p.Axes.RIGHT_HANDED_Y_UP_MINUS_Z_FORWARD, "axes"),
        (stream.hand_joint_order, p.HandJointOrder.OPENXR_26, "hand joint order"),
    )
    for actual, expected, what in semantics:
        if actual != expected:
            raise UnsupportedTracking(f"unsupported tracking {what}: {actual!r}")
    if not stream.reference_space:
        raise UnsupportedTracking("tracking descriptor declares no reference space")
    # Bind each needed field once: its dtype and shape must be exactly the ic_tracking_v1 ones and it must fit.
    declared = {f.name: f for f in stream.fields}
    fields = {}
    for name, dtype, shape in _NEEDED:
        field = declared.get(name)
        expected_wire = {"u1": p.WireDType.UINT8, "f4": p.WireDType.FLOAT32_LE}[dtype]
        if field is None or field.dtype != expected_wire or field.shape != shape:
            raise UnsupportedTracking(f"tracking field {name} is missing or not {dtype}{list(shape)}")
        nbytes = int(np.dtype(dtype).itemsize * np.prod(shape, dtype=int))
        if field.offset + nbytes > stream.payload_size:
            raise UnsupportedTracking(f"tracking field {name} extends past the declared payload size")
        fields[name] = (field.offset, np.dtype(dtype), shape)
    return Layout(stream.payload_size, fields)


def decode(payload: bytes, layout: Layout) -> dict[str, np.ndarray]:
    """The needed fields of one payload as arrays; a payload of the wrong length is refused."""
    if len(payload) != layout.payload_size:
        raise UnsupportedTracking(f"tracking payload is {len(payload)} bytes, expected {layout.payload_size}")
    return {
        name: np.frombuffer(payload, dtype=dtype, count=int(np.prod(shape, dtype=int)), offset=offset)
        .reshape(shape)
        .astype(np.float64 if dtype.kind == "f" else np.uint8)
        for name, (offset, dtype, shape) in layout.fields.items()
    }


def _matrix(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray | None:
    """A 4x4 pose from a position and XYZW quaternion; None if either is non-finite or the quaternion is zero."""
    norm = np.linalg.norm(quat_xyzw)
    if not (np.isfinite(position).all() and np.isfinite(quat_xyzw).all()) or norm < 1e-6:
        return None
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat(quat_xyzw / norm).as_matrix()
    out[:3, 3] = position
    return out


def _node_pose(matrix: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Viser node position and WXYZ quaternion for a 4x4 pose (the one XYZW -> WXYZ reorder)."""
    x, y, z, w = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return tuple(float(v) for v in matrix[:3, 3]), (float(w), float(x), float(y), float(z))


@dataclass(frozen=True)
class NodePose:
    """Where a node goes: Viser position and WXYZ orientation."""

    position: tuple[float, float, float]
    wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class Glove:
    """A glove skeleton: the reported valid joint points and the bones between valid joints, in tracking space."""

    points: np.ndarray  # (N, 3)
    segments: np.ndarray  # (M, 2, 3)


@dataclass(frozen=True)
class SceneFrame:
    """What the scene should show for one sample; None means hide that object."""

    head: NodePose | None
    controllers: dict[str, NodePose | None]
    gloves: dict[str, Glove | None]


def _pose_of(values: dict[str, np.ndarray], prefix: str, pose: str, valid: str) -> np.ndarray | None:
    """A device's 4x4 pose, or None unless it is present, flagged valid and finite."""
    if not (values[f"{prefix}.present"] and values[f"{prefix}.{valid}"]):
        return None
    return _matrix(
        values[f"{prefix}.{pose}_position" if pose else f"{prefix}.position"],
        values[f"{prefix}.{pose}_orientation" if pose else f"{prefix}.orientation"],
    )


def _glove(values: dict[str, np.ndarray], side: str) -> Glove | None:
    """One side's glove from the reported tracking-space joints; None unless the hand has a valid joint."""
    hand = f"hand_{side}"
    if not values[f"{hand}.present"]:
        return None
    positions = values[f"{hand}.joint_positions"]
    valid = values[f"{hand}.joint_valid"].astype(bool) & np.isfinite(positions).all(axis=1)
    if not valid.any():
        return None
    segments = np.array([(positions[a], positions[b]) for a, b in HAND_BONES if valid[a] and valid[b]])
    return Glove(positions[valid], segments.reshape(-1, 2, 3))


def build_frame(values: dict[str, np.ndarray]) -> SceneFrame:
    """Everything the scene should show for one decoded sample; each object is judged on its own validity."""
    head = _pose_of(values, "head", "", "is_valid")
    controllers = {}
    for side in SIDES:
        grip = _pose_of(values, f"controller_{side}", "grip", "grip_is_valid")
        controllers[side] = None if grip is None else NodePose(*_node_pose(grip))
    return SceneFrame(
        head=None if head is None else NodePose(*_node_pose(head)),
        controllers=controllers,
        gloves={side: _glove(values, side) for side in SIDES},
    )


def browser_url(config: VisualizationConfig, port: int, browser_host: str | None) -> tuple[str | None, str | None]:
    """The URL the browser should embed and a reason when there is none.

    An explicit public URL wins (the HTTPS/proxy deployment). Otherwise the page's own host name is used with the
    port Viser actually bound; a loopback-only server is not advertised to a remote browser.
    """
    if config.public_url:  # the deployment knows the browser-facing address (HTTPS proxy)
        return config.public_url, None
    if not browser_host:
        return None, "no browser-reachable host is known for the 3D view"
    bound = config.host
    loopback = bound in ("127.0.0.1", "::1", "localhost")
    # A server bound to loopback is reachable only from this machine: never point a remote browser at it.
    if loopback and browser_host not in ("127.0.0.1", "::1", "localhost"):
        return None, "the 3D view is bound to loopback; set EGOCAP_VISER_PUBLIC_URL or EGOCAP_VISER_HOST"
    try:
        host = f"[{browser_host}]" if ipaddress.ip_address(browser_host).version == 6 else browser_host
    except ValueError:
        host = browser_host
    return f"http://{host}:{port}/", None


def _frame_scene(client: Any) -> None:
    """Point a newly connected browser's camera at the tracked head and hands."""
    client.camera.position = INITIAL_CAMERA_POSITION
    client.camera.look_at = INITIAL_CAMERA_LOOK_AT


class VisualizationScene:
    """The one Viser server and sampler for this backend process; never raises into the application."""

    def __init__(self, monitor: CaptureStatusMonitor, config: VisualizationConfig) -> None:
        """Nothing starts until :meth:`start`."""
        self._monitor = monitor
        self._config = config
        self._server: Any = None
        self._handles: dict[str, Any] = {}
        self._geometry: dict[str, list[Any]] = {}
        self._profile: str = config.device  # the device profile whose models are currently in the scene
        self._assets: dict[str, dict[str, bytes | None]] = {}  # profile -> model bytes, read once at start
        self._model_handles: dict[str, Any] = {}  # the glb nodes, replaced when the profile changes
        self._startup: asyncio.Task[None] | None = None
        self._stopping = False  # shutdown was requested: a startup that finishes late must clean up, not sample
        self._sampler: asyncio.Task[None] | None = None
        self._stop_task: asyncio.Future[None] | None = None
        self._failure: str | None = None  # the feature itself cannot run
        self._model_note: str | None = None
        self._tracking_note: str | None = None
        self._bound_stream: p.TrackingStream | None = None
        self._layout: Layout | None = None
        self._last_key: tuple[object, int] | None = None

    async def start(self) -> None:
        """Start the server and sampler if enabled; any failure leaves the feature unavailable, not the app.

        The startup work is its own task, retained until it settles, so a cancelled caller cannot strand a
        half-built server that :meth:`stop` never sees."""
        if not self._config.enabled or self._failure is not None or self._startup is not None:
            return
        self._startup = asyncio.create_task(self._start_server(), name="viser-start")
        try:
            await asyncio.wait_for(asyncio.shield(self._startup), STARTUP_TIMEOUT_S)
        except TimeoutError:
            logger.error(
                "the 3D view is still starting after %.0f s; serving without it for now", STARTUP_TIMEOUT_S
            )

    async def _start_server(self) -> None:
        """Read the models, build the server and nodes off the event loop, then begin sampling."""
        try:
            assets = await asyncio.to_thread(self._read_assets)
            await asyncio.to_thread(self._build_server, assets)
        except ModuleNotFoundError as exc:
            if exc.name != "viser":
                self._fail(f"3D view failed to start: {exc}", exc)
            else:
                self._fail("3D view needs the viser package; run `uv sync` to install it", exc)
            await self._stop_server()
            return
        except Exception as exc:
            self._fail(f"3D view failed to start: {exc}", exc)
            await self._stop_server()
            return
        if self._stopping:  # shutdown began while the server was being built: release it instead of sampling
            await self._stop_server()
            return
        self._sampler = asyncio.create_task(self._run(), name="viser-scene")

    async def stop(self) -> None:
        """Let any startup settle (bounded), stop the sampler, then the server, releasing its threads and port.

        The startup task is shielded: cancelling or timing out this wait never cancels it, and it stays owned so a
        later stop can settle it. Shutdown intent is recorded first, so a startup that completes after this call
        gave up waiting releases its own server instead of starting a sampler."""
        self._stopping = True
        if self._startup is not None and not self._startup.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._startup), STOP_TIMEOUT_S)
            except TimeoutError:
                self._fail(f"3D view startup did not settle within {STOP_TIMEOUT_S:.0f} s")
                return
        if self._sampler is not None:
            self._sampler.cancel()
            await asyncio.gather(self._sampler, return_exceptions=True)
            self._sampler = None
        await self._stop_server()

    def mark_unavailable(self, message: str) -> None:
        """Report the feature unavailable without starting it (e.g. its configuration could not be read)."""
        self._fail(message)

    def projection(self, browser_host: str | None) -> dict[str, Any]:
        """The small ``visualization`` object advertised to the page."""
        if not self._config.enabled:
            return {"status": "disabled", "url": None, "diagnostic": None}
        if self._failure is not None or self._server is None:
            starting = self._startup is not None and not self._startup.done()
            return {
                "status": "unavailable",
                "url": None,
                "diagnostic": self._failure or ("3D view is starting" if starting else None),
            }
        url, why = browser_url(self._config, self._server.get_port(), browser_host)
        if url is None:
            return {"status": "unavailable", "url": None, "diagnostic": why}
        notes = [n for n in (self._tracking_note, self._model_note) if n]
        # ``device`` names the hardware the models depict, so a stand-in is never mistaken for the real device;
        # it is a separate member because ``diagnostic`` is reserved for why the view is degraded.
        return {
            "status": "ready",
            "url": url,
            "diagnostic": "; ".join(notes) or None,
            "device": DEVICE_MODEL_PROFILES[self._profile].label,
        }

    def _fail(self, message: str, exc: Exception | None = None) -> None:
        """Mark the feature unavailable and log why."""
        self._failure = message
        logger.error("%s", message, exc_info=exc)

    def _read_assets(self) -> dict[str, bytes | None]:
        """Read every shipped profile's model files once and return the configured profile's.

        All profiles are read up front (about a megabyte in total) so a runtime switch to the device the status
        names is a scene update, never file I/O inside the sampler. A missing or invalid file is a note for the
        configured profile and a silent coordinate-frame fallback for the others, not a failure."""
        for profile in DEVICE_MODEL_PROFILES:
            models = self._config.models_for(profile)
            wanted = {"head": models.head, "left": models.left_controller, "right": models.right_controller}
            assets: dict[str, bytes | None] = {}
            for name, path in wanted.items():
                try:
                    data = Path(path).read_bytes() if path else None
                except OSError:
                    data = None
                # A binary glTF starts with "glTF"; anything else (such as a Git LFS pointer in a checkout without
                # LFS files) would load as an invisible model, so it counts as missing.
                assets[name] = data if data is not None and data[:4] == b"glTF" else None
            self._assets[profile] = assets
        current = self._assets[self._profile]
        missing = [name for name in _MODEL_KEYS if current[name] is None]
        if missing:
            self._model_note = f"no {', '.join(missing)} model; showing coordinate frames"
        return current

    def _build_server(self, assets: dict[str, bytes | None]) -> None:
        """Create the server and every tracked node, hidden. The server is kept as soon as it exists."""
        import viser  # lazy import: skipped entirely when the view is off

        self._server = viser.ViserServer(host=self._config.host, port=self._config.port, verbose=False)
        self._server.on_client_connect(_frame_scene)
        # Dark canvas: the Pico 4 Ultra headset and controllers are white, and on Viser's default white background
        # they are barely visible, which defeats the point of accurate device models.
        self._server.gui.configure_theme(
            control_layout="fixed",
            control_width="small",
            show_logo=False,
            show_share_button=False,
            dark_mode=True,
        )
        self._build_nodes(self._server, assets)

    def _build_nodes(self, server: Any, assets: dict[str, bytes | None]) -> None:
        """Create the fixed scene nodes, all hidden, and load the current profile's models once."""
        # Callers that hand in one profile's assets directly (tests) still get a profile table to switch from.
        self._assets.setdefault(self._profile, assets)
        scene = server.scene
        scene.set_up_direction("+y")
        scene.add_frame("/tracking", show_axes=False)
        head = scene.add_frame("/tracking/head", show_axes=assets["head"] is None, axes_length=0.1, visible=False)
        self._handles["head"] = head
        for side in SIDES:
            grip = scene.add_frame(
                f"/tracking/{side}_controller", show_axes=assets[side] is None, axes_length=0.05, visible=False
            )
            self._handles[f"controller_{side}"] = grip
            self._handles[f"glove_{side}"] = scene.add_frame(
                f"/tracking/{side}_glove", show_axes=False, visible=False
            )
        self._add_models(scene, assets)

    def _add_models(self, scene: Any, assets: dict[str, bytes | None]) -> None:
        """Attach one profile's model meshes under the tracked frames (a missing file leaves the frame's axes)."""
        if assets["head"] is not None:
            self._model_handles["head"] = scene.add_glb(
                "/tracking/head/model",
                assets["head"],
                scale=self._config.head_model_scale,
                wxyz=self._config.head_model_wxyz,
                position=self._config.head_model_position,
            )
        for side in SIDES:
            if assets[side] is not None:
                self._model_handles[side] = scene.add_glb(f"/tracking/{side}_controller/model", assets[side])

    def _sync_models(self, snapshot: Any) -> None:
        """Show the device the capture status names, if it names one and it differs from what is in the scene.

        The status is the only runtime evidence of the hardware; when it is silent the configured profile stands.
        Switching replaces just the model meshes; the tracked frames, their poses and the gloves are untouched.
        """
        status = getattr(snapshot, "status", None)
        wanted = detect_device_profile(getattr(status, "devices", None)) or self._config.device
        if wanted == self._profile or wanted not in self._assets or self._server is None:
            return
        with self._server.atomic():
            for handle in self._model_handles.values():
                handle.remove()
            self._model_handles = {}
            self._add_models(self._server.scene, self._assets[wanted])
        self._profile = wanted
        logger.info(
            "3D view now shows %s models (named by the capture status)", DEVICE_MODEL_PROFILES[wanted].label
        )

    async def _stop_server(self) -> None:
        """Stop Viser through its public API. The server stays owned until that stop is confirmed; a stop that
        outlasts its time limit is reported and kept, so it can be awaited again rather than forgotten."""
        server = self._server
        if server is None:
            return
        if self._stop_task is None:
            self._stop_task = asyncio.ensure_future(asyncio.to_thread(server.stop))
        try:
            await asyncio.wait_for(asyncio.shield(self._stop_task), STOP_TIMEOUT_S)
        except TimeoutError:
            self._fail(f"3D view server did not stop within {STOP_TIMEOUT_S:.0f} s")
            return
        except Exception as exc:  # the server may still be running: keep it, and let a later stop try again
            self._stop_task = None
            self._fail(f"3D view server failed while stopping: {exc}", exc)
            return
        self._server, self._stop_task, self._handles, self._geometry = None, None, {}, {}
        self._model_handles = {}

    async def _run(self) -> None:
        """The periodic sampler. An update error is not retried: the scene is hidden, the feature reports
        itself unavailable once, and sampling stops."""
        while True:
            try:
                self._tick(time.monotonic())
            except Exception as exc:
                self._fail(f"3D view update failed: {exc}", exc)
                try:
                    self._hide_all()
                except Exception:
                    logger.exception("could not hide the 3D scene after an update failure")
                return
            await asyncio.sleep(SAMPLE_INTERVAL_S)

    def _tick(self, now: float) -> None:
        """Judge freshness on every tick; recompute geometry only for a new sample."""
        monitor = self._monitor
        snapshot = monitor.snapshot()
        if snapshot.terminal:
            self._hide("capture ended")
            return
        # The device named in the status can only be known once capture reports it, so it is re-checked per tick.
        self._sync_models(snapshot)
        # Freshness is judged on every tick, even for an unchanged sample, so a stalled source gets hidden.
        stream, slot = monitor.tracking_descriptor, monitor.preview.tracking()
        if stream is None or slot is None:
            self._hide("waiting for tracking")
            return
        if stream is not self._bound_stream:  # a new accepted descriptor: clear the scene and bind its layout
            self._bound_stream, self._layout, self._last_key = stream, None, None
            self._hide(None)
            try:
                self._layout = bind_layout(stream)
            except UnsupportedTracking as exc:
                self._tracking_note = str(exc)
        if self._layout is None:
            return
        if now - slot.received_monotonic > STALE_AFTER_S:
            self._hide("tracking is stale")
            return
        # A sample is the same only within one accepted attempt: a restarted stream can reuse sequence numbers, so
        # the ``started`` identity is part of the key.
        key = (monitor.native.started, slot.sequence)
        if self._last_key is not None and key[0] is self._last_key[0] and key[1] == self._last_key[1]:
            return
        self._apply(build_frame(decode(slot.payload, self._layout)))
        self._last_key, self._tracking_note = key, None

    def _hide(self, note: str | None) -> None:
        """Hide the dynamic geometry and record why."""
        self._tracking_note = note
        self._last_key = None
        self._hide_all()

    def _hide_all(self) -> None:
        """Hide every tracked node in one atomic update."""
        if self._server is None:
            return
        with self._server.atomic():
            for handle in self._handles.values():
                handle.visible = False

    def _apply(self, frame: SceneFrame) -> None:
        """One coherent update: poses, geometry and visibility together."""
        with self._server.atomic():
            self._place("head", frame.head)
            for side in SIDES:
                self._place(f"controller_{side}", frame.controllers[side])
                glove = frame.gloves[side]
                self._set_geometry(side, glove)
                # The glove's points are already in tracking space, so its parent node stays at the origin.
                self._handles[f"glove_{side}"].visible = glove is not None

    def _set_geometry(self, side: str, glove: Glove | None) -> None:
        """Replace one glove's joint points and bones; a bone-less sample must not leave old bones behind."""
        for handle in self._geometry.pop(side, ()):
            handle.remove()
        if glove is None or not len(glove.points):
            return
        scene, base, color = self._server.scene, f"/tracking/{side}_glove", _COLORS[side]
        handles = [scene.add_point_cloud(f"{base}/joints", glove.points, colors=color, point_size=0.006)]
        if len(glove.segments):
            handles.append(scene.add_line_segments(f"{base}/bones", glove.segments, colors=color, line_width=2.0))
        self._geometry[side] = handles

    def _place(self, key: str, pose: NodePose | None) -> None:
        """Move a node and show it, or hide it when there is no valid pose."""
        handle = self._handles[key]
        if pose is None:
            handle.visible = False
            return
        handle.position, handle.wxyz = pose.position, pose.wxyz
        handle.visible = True

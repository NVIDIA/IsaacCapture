# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""SENSING camera source: station configuration, producer ownership and the per-stream adapter.

One native ``camera_plugin_sensing`` producer per rig writes every selected sensor's H.264 stream to its own
file; each stream's adapter tails one file, decodes it in order and JPEG-encodes the newest decoded frame for the
existing CaptureServer image path. Nothing loads the decoder until a stream attaches, so OAK-only and tracking-only
runs never need it.

The producer's existing interface is used unchanged: each selected sensor's H.264 output file
(``--add-stream=sensor=N,output=PATH``). The adapter tails that file, decodes every unit in order on the CPU,
and JPEG-encodes the newest decoded frame. No CUDA is touched in this process.

Native configuration is server-owned: a client's Begin selects logical camera names only, never paths,
masks or arguments.
"""

from __future__ import annotations

import asyncio
import ctypes
import functools
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from capture_server import protocol as p

from ._thread_runner import ThreadRunner
from .clock import Clock, MonotonicClock
from .errors import CameraSourceError, SourceError
from .sensing_h264 import SensingStreamError, SensingStreamReader
from .sensing_jpeg import encode_rgb_jpeg

__all__ = [
    "SensingAdapter",
    "SensingDescription",
    "SensingRig",
    "SensingRigConfig",
    "SensingSensor",
    "SensingStationCamera",
    "inspect_rig",
    "parse_sensor_listing",
    "load_sensing_station",
    "resolve_sensors",
]

_BACKENDS = ("JETSON", "MOCK")
# Producer startup (camera bring-up) and first-frame budget per stream.
_STARTUP_TIMEOUT_S = 30.0
_INSPECT_TIMEOUT_S = 10.0
# A started stream that produces no decoded frame for this long (or five sensor frame periods, if longer) has
# failed, the same rule DepthAIAdapter applies to an OAK camera.
_NO_FRAME_TIMEOUT_S = 5.0
# Shutdown of the SENSING chain fits inside the server's single 5 s source-release budget: the stream adapters
# settle first (_ADAPTER_RELEASE_S), then the producer gets _RIG_CLOSE_BUDGET_S, of which its own steps use
# _RIG_STEPS_S (half of that is the SIGTERM grace before SIGKILL). The OAK and tracking adapters are not part of
# this chain; they close concurrently on their own bounds. The producer's real teardown time on hardware is not
# measured yet; these are bounds, not measurements.
_ADAPTER_RELEASE_S = 1.5
_RIG_CLOSE_BUDGET_S = 3.5
_RIG_STEPS_S = 3.0
_LOG_TAIL_BYTES = 4096
# Linux's prctl(2) option that asks the kernel to deliver a signal to this process when its parent dies, so an
# orphaned producer (this server killed hard -- OOM, SIGKILL, a native crash) is reaped by the kernel instead of
# running unsupervised and filling its tmpfs runtime directory forever. No effect on an ordinary close(): that
# path already terminates the producer itself. Linux-only; harmless no-op everywhere else (see _set_pdeathsig).
_PR_SET_PDEATHSIG = 1


@dataclass(frozen=True)
class SensingRigConfig:
    """Server-owned settings of the one native producer: which binary, backend, platform and link masks."""

    executable: str
    expected_backend: str
    platform_config: str
    config_name: str
    link_masks: tuple[str, ...]
    gpu_id: int = 0
    nito_dir: str | None = None
    # Encoder settings passed to the producer only when set; otherwise its own defaults apply. The decoder handles
    # whatever GOP the producer uses, so none of these is a decoding requirement.
    gop: int | None = None
    bitrate_bps: int | None = None
    # Whether the producer's NV12 uses the full [0,255] luma range; the decoder must convert with the same.
    full_range: bool = False
    # Where per-session stream files live; None prefers /dev/shm. tmpfs is memory, not extra capacity: the files
    # stay bounded only because consumed bytes are reclaimed as they are read (see sensing_h264.py).
    runtime_dir: str | None = None

    def __post_init__(self) -> None:
        """Reject an unknown backend, a missing executable/platform/name, or empty link masks."""
        if self.expected_backend not in _BACKENDS:
            raise ValueError(f"SensingRigConfig.expected_backend must be one of {_BACKENDS}")
        if not (self.executable and self.platform_config and self.config_name and self.link_masks):
            raise ValueError("SensingRigConfig needs executable, platform_config, config_name and link_masks")

    def base_args(self) -> list[str]:
        """The producer's rig arguments as an argument list (never a shell command)."""
        args = [
            self.executable,
            f"--platform-config={self.platform_config}",
            f"--config-name={self.config_name}",
            f"--link-masks={' '.join(self.link_masks)}",
            f"--gpu-id={self.gpu_id}",
        ]
        if self.nito_dir:
            args.append(f"--nito-dir={self.nito_dir}")
        if self.gop is not None:
            args.append(f"--gop={self.gop}")
        if self.bitrate_bps is not None:
            args += [f"--bitrate={self.bitrate_bps}", f"--peak-bitrate={self.bitrate_bps * 3 // 2}"]
        if self.full_range:
            args.append("--full-range")
        return args


@dataclass(frozen=True)
class SensingStationCamera:
    """Server-owned settings of one SENSING sensor, keyed by the logical name clients request."""

    module_name: str
    delivery_fps: float = 30.0
    jpeg_quality: int = 80
    required: bool = True
    # Which sensor of the module, counted in pipeline-id order, when a module has several sensors (a stereo
    # camera's two eyes share one module name). Required exactly when the module name is not unique.
    sensor_index: int | None = None

    def __post_init__(self) -> None:
        """Reject an empty module name, a bad sensor index, a non-positive delivery rate or a bad JPEG quality."""
        if not self.module_name:
            raise ValueError("SensingStationCamera.module_name must be nonempty")
        if self.sensor_index is not None and (type(self.sensor_index) is not int or self.sensor_index < 0):
            raise ValueError(
                f"SensingStationCamera.sensor_index must be a non-negative integer, got {self.sensor_index!r}"
            )
        if not 0 < self.delivery_fps < float("inf"):
            raise ValueError(f"SensingStationCamera.delivery_fps must be positive, got {self.delivery_fps}")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError(f"SensingStationCamera.jpeg_quality must be in 1..100, got {self.jpeg_quality}")


@dataclass(frozen=True)
class SensingSensor:
    """One sensor as the native producer resolved it: module name, pipeline id, native geometry."""

    module_name: str
    pipeline_id: int
    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class SensingDescription:
    """What the producer reports for the configured platform: each sensor's module, pipeline id, size and rate.

    The producer's existing ``--list-sensors`` does not say which backend it was built with, so the backend is
    the station's declared ``expected_backend`` (checked against ``--mock`` at configuration time), not
    something read back from the executable.
    """

    sensors: tuple[SensingSensor, ...]


_RIG_KEYS = {
    "executable",
    "expected_backend",
    "platform_config",
    "config_name",
    "link_masks",
    "gpu_id",
    "nito_dir",
    "gop",
    "bitrate_bps",
    "full_range",
    "runtime_dir",
}
_CAMERA_KEYS = {"type", "module_name", "delivery_fps", "jpeg_quality", "required", "sensor_index"}


# Scalar types accepted for each key. bool is checked separately because it is a subclass of int in Python.
_RIG_TYPES: dict[str, type] = {
    "executable": str,
    "expected_backend": str,
    "platform_config": str,
    "config_name": str,
    "gpu_id": int,
    "nito_dir": str,
    "gop": int,
    "bitrate_bps": int,
    "full_range": bool,
    "runtime_dir": str,
}
_CAMERA_TYPES: dict[str, type | tuple[type, ...]] = {
    "type": str,
    "module_name": str,
    "delivery_fps": (int, float),
    "jpeg_quality": int,
    "required": bool,
    "sensor_index": int,
}


def _check_types(where: str, mapping: dict, types: dict) -> None:
    """Reject a value whose YAML type is not the one the key needs (a bool is never accepted as a number)."""
    for key, value in mapping.items():
        wanted = types[key]
        accepted = wanted if isinstance(wanted, tuple) else (wanted,)
        if (isinstance(value, bool) and bool not in accepted) or not isinstance(value, accepted):
            names = "/".join(t.__name__ for t in accepted)
            raise ValueError(f"{where}.{key} must be {names}, got {type(value).__name__} {value!r}")


def load_sensing_station(path: str) -> tuple[SensingRigConfig, dict[str, SensingStationCamera]]:
    """Load the station's SENSING rig and cameras from a YAML file; the one configuration entry point.

    Expected shape: ``sensing_rig:`` (see :class:`SensingRigConfig`) and ``cameras:`` mapping logical names
    to ``type: SENSING_COLOR`` entries (see :class:`SensingStationCamera`). Unknown keys, wrongly typed values,
    a non-SENSING type, a missing section or a file that is not valid YAML all raise ``ValueError``; nothing is
    defaulted silently. Link masks must be quoted strings (``'0x0000'``): an unquoted ``0x0000`` is a YAML
    integer and would reach the producer as a decimal number.
    """
    import yaml

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"{path}: not valid YAML: {e}") from e
    if not isinstance(data, dict) or set(data) - {"sensing_rig", "cameras"} or not data.get("sensing_rig"):
        raise ValueError(f"{path}: expected only 'sensing_rig' and 'cameras' sections, with a sensing_rig")
    rig = data["sensing_rig"]
    if not isinstance(rig, dict) or set(rig) - _RIG_KEYS:
        raise ValueError(f"{path}: sensing_rig keys must be among {sorted(_RIG_KEYS)}")
    masks = rig.get("link_masks", [])
    if not isinstance(masks, list) or not all(isinstance(m, str) for m in masks):
        raise ValueError(f"{path}: sensing_rig.link_masks must be a list of quoted strings such as '0x0000'")
    scalars = {k: v for k, v in rig.items() if k != "link_masks"}
    try:
        _check_types(f"{path}: sensing_rig", scalars, _RIG_TYPES)
        rig_config = SensingRigConfig(**{**scalars, "link_masks": tuple(masks)})
    except TypeError as e:
        raise ValueError(f"{path}: invalid sensing_rig: {e}") from e
    raw_cameras = data.get("cameras")
    if not isinstance(raw_cameras, dict):
        raise ValueError(f"{path}: cameras must be a mapping of logical name to camera settings")
    cameras: dict[str, SensingStationCamera] = {}
    for name, entry in raw_cameras.items():
        if not isinstance(entry, dict) or set(entry) - _CAMERA_KEYS or entry.get("type") != "SENSING_COLOR":
            raise ValueError(
                f"{path}: camera {name!r} must be type SENSING_COLOR with keys among {sorted(_CAMERA_KEYS)}"
            )
        _check_types(f"{path}: cameras.{name}", entry, _CAMERA_TYPES)
        try:
            cameras[str(name)] = SensingStationCamera(**{k: v for k, v in entry.items() if k != "type"})
        except TypeError as e:
            raise ValueError(f"{path}: invalid camera {name!r}: {e}") from e
    if not cameras:
        raise ValueError(f"{path}: no SENSING cameras configured")
    return rig_config, cameras


# One line of the producer's ``--list-sensors``: "  sensor=<pipeline id>  <module>  <w>x<h> @ <fps> fps".
_SENSOR_LINE = re.compile(r"^\s*sensor=(\d+)\s+(\S+)\s+(\d+)x(\d+)\s+@\s+([0-9.]+)\s+fps\s*$")


def parse_sensor_listing(text: str) -> SensingDescription:
    """Parse the producer's ``--list-sensors`` output; raises ``ValueError`` on anything unexpected.

    The first line (``CONFIG: N sensor(s)``) states how many sensor lines must follow, so a truncated or
    garbled listing is rejected instead of silently describing fewer sensors.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    header = re.match(r"^\S+: (\d+) sensor\(s\)$", lines[0].strip()) if lines else None
    if header is None:
        raise ValueError(f"malformed sensor listing: unexpected first line {lines[0] if lines else ''!r}")
    sensors = []
    for line in lines[1:]:
        match = _SENSOR_LINE.match(line)
        if match is None:
            raise ValueError(f"malformed sensor listing: unexpected line {line!r}")
        pipeline_id, module, width, height, fps = match.groups()
        if int(width) <= 0 or int(height) <= 0 or not 0 < float(fps) < float("inf"):
            raise ValueError(f"malformed sensor listing: non-positive geometry or rate in {line!r}")
        sensors.append(SensingSensor(module, int(pipeline_id), int(width), int(height), float(fps)))
    if len(sensors) != int(header.group(1)):
        raise ValueError(f"malformed sensor listing: header says {header.group(1)} sensors, found {len(sensors)}")
    return SensingDescription(tuple(sensors))


def _producer_env() -> dict[str, str]:
    """The environment for every run of the producer binary, listing and capture alike.

    The reduced-size test knob must never reach either, or the listed geometry and the captured geometry could
    differ from the station's.
    """
    return {k: v for k, v in os.environ.items() if k != "SENSING_MOCK_SCALE_DIV"}


@functools.cache
def _bind_prctl() -> Callable[[int, int, int, int, int], int] | None:
    """The C library's ``prctl`` with explicit argument types, bound once; ``None`` where it does not exist
    (anywhere but Linux). Called in the parent before each spawn (see ``_start_sync``), so the ``preexec_fn``
    below only ever calls an already-resolved C function pointer -- see ``_set_pdeathsig``.
    """
    try:
        prctl = ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        return None
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    return prctl


def _set_pdeathsig() -> None:
    """``subprocess.Popen``'s ``preexec_fn``: ask the kernel to SIGTERM the producer if this server dies first.

    Runs in the forked child between ``fork()`` and ``exec()``. It calls only a ctypes function pointer already
    bound in the parent, with primitive integer arguments, so it does no dlopen or ctypes construction there.
    The kernel delivers the signal when the *spawning thread* (the rig's executor worker), not the process,
    exits, and exec clears the setting for a setuid or file-capability binary. A failed or unsupported call (not
    Linux, or the kernel refuses it) is silently skipped: PR_SET_PDEATHSIG only narrows the orphaned-producer
    window from "unbounded" to "none", it is not the only thing cleaning up the producer on an ordinary close().
    """
    if sys.platform != "linux":
        return
    prctl = _bind_prctl()
    if prctl is not None:
        prctl(_PR_SET_PDEATHSIG, 15, 0, 0, 0)  # 15 == SIGTERM; avoids importing signal for one constant


def inspect_rig(
    config: SensingRigConfig,
    *,
    run: Callable[..., Any] = subprocess.run,
    timeout_s: float = _INSPECT_TIMEOUT_S,
) -> SensingDescription:
    """Ask the native executable which sensors the configured platform resolves to; blocking and bounded.

    Uses the producer's existing ``--list-sensors`` (hardware-free; it never starts capture). Run it off the
    event loop.
    """
    try:
        done = run(
            config.base_args() + ["--list-sensors"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=_producer_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SourceError(f"could not inspect {config.executable}: {e}") from e
    if done.returncode != 0:
        raise SourceError(
            f"{config.executable} --list-sensors failed ({done.returncode}): {done.stderr.strip()[-500:]}"
        )
    try:
        return parse_sensor_listing(done.stdout)
    except ValueError as e:
        raise SourceError(str(e)) from e


def resolve_sensors(
    description: SensingDescription, stations: dict[str, SensingStationCamera]
) -> dict[str, SensingSensor]:
    """Map each logical camera name to its resolved sensor by module name.

    A module with several sensors needs ``sensor_index``. Rejects a missing module, an ambiguous or out-of-range
    selection, and two logical names selecting the same physical sensor.
    """
    resolved: dict[str, SensingSensor] = {}
    for name, station in stations.items():
        # Pipeline-id order makes sensor_index stable: the link mask renumbers pipelines, not their order.
        matches = sorted(
            (s for s in description.sensors if s.module_name == station.module_name), key=lambda s: s.pipeline_id
        )
        if not matches:
            raise SourceError(f"camera {name!r}: module {station.module_name!r} is not in the platform config")
        if station.sensor_index is None:
            if len(matches) > 1:
                raise SourceError(
                    f"camera {name!r}: module {station.module_name!r} has {len(matches)} sensors; "
                    "select one with sensor_index"
                )
            resolved[name] = matches[0]
        elif station.sensor_index >= len(matches):
            raise SourceError(
                f"camera {name!r}: module {station.module_name!r} has {len(matches)} sensor(s), "
                f"no sensor_index {station.sensor_index}"
            )
        else:
            resolved[name] = matches[station.sensor_index]
    pipelines = [s.pipeline_id for s in resolved.values()]
    dup = sorted({i for i in pipelines if pipelines.count(i) > 1})
    if dup:
        raise SourceError(f"physical sensor(s) selected by more than one logical camera: pipeline {dup}")
    return resolved


class SensingRig:
    """Owns one producer process, its private runtime directory and per-stream H.264 output file paths.

    ``start`` and ``close`` run on the rig's own dedicated worker, one job at a time, so a spawn still in
    flight when Stop or cancellation arrives is always finished and then reaped by the queued close; it can
    never escape cleanup. The producer is launched with one ``--add-stream=...,output=FILE`` per selected
    sensor, and with PR_SET_PDEATHSIG (Linux; a silent no-op elsewhere) so a hard death of this server
    process -- OOM, SIGKILL, a native crash -- still gets the producer reaped by the kernel, rather than
    left running unsupervised with nothing left to bound its tmpfs output. Diagnostics are drained
    continuously into a byte-bounded tail so a full pipe can never stall capture. A start failure is
    recorded in ``start_error`` for each adapter to report under its own required/optional policy rather
    than ending the attempt here.
    """

    def __init__(
        self,
        config: SensingRigConfig,
        pipeline_ids: tuple[int, ...],
        *,
        popen: Callable[..., Any] = subprocess.Popen,
        runtime_root: str | None = None,
    ) -> None:
        """Records the config and selected pipeline ids; nothing is started or created until :meth:`start`."""
        self._config = config
        self._pipeline_ids = pipeline_ids
        self._popen = popen
        self._runtime_root = runtime_root
        self._dir: str | None = None
        self._proc: Any = None
        self._drain: threading.Thread | None = None
        self._tail = b""
        self._abort_reason = ""
        self._runner = ThreadRunner(release_timeout_s=_RIG_CLOSE_BUDGET_S)
        self.start_error: str | None = None
        # Sticky: once any close failed to confirm release, no later call reports success.
        self.cleanup_ok = True

    def stream_path(self, pipeline_id: int) -> str:
        """The H.264 output file for one selected sensor; valid after a successful :meth:`start`."""
        assert self._dir is not None, "rig not started"
        return os.path.join(self._dir, f"sensor{pipeline_id}.h264")

    async def start(self) -> None:
        """Launch the producer on the rig's worker; a failure is recorded in ``start_error``, not raised."""
        try:
            await self._runner.run(self._start_sync)
        except SourceError as e:
            self.start_error = str(e)

    def _start_sync(self) -> None:
        # Local filesystem failures are start failures too (an optional stream then degrades instead of
        # aborting healthy OAK/tracking). The directory, once created, stays owned for close() to remove.
        """Create the private runtime directory and spawn the producer on the rig worker.

        Raises ``SourceError`` on failure.
        """
        try:
            root = self._runtime_root or self._config.runtime_dir
            if root is None and os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
                root = "/dev/shm"
            self._dir = tempfile.mkdtemp(prefix="sensing-", dir=root)
            os.chmod(self._dir, 0o700)
        except OSError as e:
            raise SourceError(f"could not prepare the SENSING runtime directory: {e}") from e
        args = self._config.base_args() + [
            f"--add-stream=sensor={i},output={self.stream_path(i)}" for i in self._pipeline_ids
        ]
        env = _producer_env()
        _bind_prctl()  # resolve in the parent, so the post-fork preexec_fn does no dlopen
        try:
            self._proc = self._popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=_set_pdeathsig,
            )
        except OSError as e:
            self._remove_dir()
            raise SourceError(f"could not start {self._config.executable}: {e}") from e
        self._drain = threading.Thread(target=self._drain_output, name="sensing-rig-log", daemon=True)
        self._drain.start()

    def _drain_output(self) -> None:
        """Log-drain thread body.

        Keeps the producer's most recent output so its pipe can never fill and stall it.
        """
        stream = self._proc.stdout
        try:
            while chunk := stream.read1(4096):
                # Byte-bounded (a newline-free flood cannot grow it); one atomic assignment, so readers
                # always see a consistent snapshot.
                self._tail = (self._tail + chunk)[-_LOG_TAIL_BYTES:]
        except (OSError, ValueError):
            pass  # the pipe was closed under us during shutdown
        finally:
            stream.close()

    def exit_code(self) -> int | None:
        """The producer's exit code once it has exited on its own, else None."""
        return None if self._proc is None else self._proc.poll()

    def diagnostics(self) -> str:
        """The producer's most recent output, decoded leniently, for error messages."""
        tail = self._tail.decode("utf-8", errors="replace").strip().replace("\n", " | ")
        return f"stopped by the server: {self._abort_reason} | {tail}" if self._abort_reason else tail

    def abort(self, reason: str) -> None:
        """Stop the producer because its output can no longer be bounded; ``close`` still reaps it."""
        self._abort_reason = self._abort_reason or reason
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()

    async def close(self) -> bool:
        """Terminate and reap the producer, join the log drain, remove the runtime directory.

        Runs on the rig's worker after any pending start, within one bounded budget. Returns True only when
        every step is confirmed, and False on every later call once any attempt failed or timed out.
        """
        try:
            done = await self._runner.close(self._close_sync)
        except Exception:
            self.cleanup_ok = False
            raise
        if not done:
            self.cleanup_ok = False
        return done and self.cleanup_ok

    def _close_sync(self) -> None:
        """Terminate, reap and clean up within ``_RIG_STEPS_S`` on the rig worker.

        Sets ``cleanup_ok`` False if anything is unconfirmed.
        """
        deadline = time.monotonic() + _RIG_STEPS_S
        ok = True
        proc = self._proc
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=_RIG_STEPS_S / 2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=max(deadline - time.monotonic(), 0.05))
                    except subprocess.TimeoutExpired:
                        ok = False
            if ok and self._drain is not None:
                self._drain.join(timeout=max(deadline - time.monotonic(), 0.05))
                ok = not self._drain.is_alive()
        if ok:
            ok = self._remove_dir()
        if not ok:
            self.cleanup_ok = False

    def _remove_dir(self) -> bool:
        """Remove this rig's own runtime directory only; False if it could not be removed."""
        if self._dir is None:
            return True
        try:
            shutil.rmtree(self._dir)
        except FileNotFoundError:
            pass
        except OSError:
            return False
        self._dir = None
        return True


class SensingAdapter:
    """One SENSING stream: tail its H.264 file, deliver the newest frame as JPEG at the delivery rate.

    The same instance shape the server expects of every camera adapter. All blocking file, decode and
    JPEG work runs on this adapter's dedicated worker; the reader is created, used and closed there.
    """

    # Acquisition closes adapters that borrow the shared producer before it reaps the producer.
    borrows_rig = True

    def __init__(
        self,
        *,
        rig: SensingRig,
        sensor: SensingSensor,
        camera: p.CameraStreamConfig,
        full_range: bool = False,
        clock: Clock | None = None,
        startup_timeout_s: float = _STARTUP_TIMEOUT_S,
    ) -> None:
        """Bind one sensor of the shared rig to one accepted camera stream.

        The reader is created later, in :meth:`enter`.
        """
        self._rig = rig
        self._sensor = sensor
        self._camera = camera
        self._interval = 1.0 / camera.fps
        self._clock = clock or MonotonicClock()
        self._startup_timeout_s = startup_timeout_s
        self._runner = ThreadRunner(release_timeout_s=_ADAPTER_RELEASE_S)
        self._full_range = full_range
        # Built in enter(): the stream file path exists only once Acquisition has started the rig.
        self._reader: SensingStreamReader | None = None
        self._deadline: float | None = None
        self._dead = False
        self._flushed = False
        # Set synchronously by close(), before its job queues behind the worker, so startup, reads and the final
        # flush stop within a fraction of a second.
        self._stop = threading.Event()
        # Host time of the last decoded frame (or of the first confirmed one); drives stall detection.
        self._last_frame_s: float | None = None
        self._degraded = True
        self._last_error = "camera not yet started"
        # Sticky: a close that failed to confirm release is never forgotten, whatever happens to the stream.
        self.cleanup_failed = False

    @property
    def stream_id(self) -> int:
        """The server-assigned stream id this adapter delivers."""
        return self._camera.stream_id

    @property
    def required(self) -> bool:
        """Whether this stream's failure ends the whole session."""
        return self._camera.required

    def _fail(self, message: str) -> CameraSourceError:
        """Mark the stream dead and degraded, and return the ``CameraSourceError`` for the caller to raise."""
        self._dead, self._degraded, self._last_error = True, True, message
        return CameraSourceError(message, camera_name=self._camera.name)

    def _enter_sync(self) -> None:
        """Worker-side startup: wait for the stream file and a first complete, decodable frame.

        Raises a camera error if either does not arrive.
        """
        if self._rig.start_error is not None:
            # A shared-producer failure degrades exactly the streams that depend on it, each under its own
            # required/optional policy in Acquisition.enter.
            raise self._fail(f"SENSING producer did not start: {self._rig.start_error}")
        deadline = time.monotonic() + self._startup_timeout_s
        self._reader = reader = SensingStreamReader(
            self._rig.stream_path(self._sensor.pipeline_id),
            width=self._sensor.width,
            height=self._sensor.height,
            full_range=self._full_range,
            stop=self._stop,
        )
        # The file existing is not readiness: attach, then wait for a complete, decodable frame.
        while True:
            if self._stop.is_set():
                raise self._fail("stopped during startup")
            if (code := self._rig.exit_code()) is not None:
                raise self._fail(f"SENSING producer exited with {code}: {self._rig.diagnostics()}")
            left = deadline - time.monotonic()
            if left <= 0:
                raise self._fail(f"timed out waiting for the SENSING stream: {self._rig.diagnostics()}")
            try:
                reader.connect(min(left, 2.0))
                break
            except OSError:
                self._stop.wait(0.1)  # the producer has not created the file yet
            except SensingStreamError as e:
                if self._stop.is_set():
                    raise self._fail("stopped during startup") from e
                raise self._fail(f"SENSING stream rejected: {e}") from e
        try:
            while reader.read_latest(min(max(deadline - time.monotonic(), 0.0), 0.5)) is None:
                if (code := self._rig.exit_code()) is not None:
                    raise self._fail(f"SENSING producer exited with {code}: {self._rig.diagnostics()}")
                if time.monotonic() >= deadline:
                    raise self._fail("no frame from the SENSING stream before the startup deadline")
        except SensingStreamError as e:
            if self._stop.is_set():
                raise self._fail("stopped during startup") from e
            raise self._fail(f"SENSING stream failed at startup: {e}") from e
        self._last_frame_s = self._clock.monotonic()

    async def enter(self) -> None:
        """Attach and confirm a usable frame; the deadline is set first so an optional failure stays pollable."""
        self._deadline = self._clock.monotonic() + self._interval
        await self._runner.run(self._enter_sync)
        self._degraded = False
        self._last_error = ""

    def _poll_sync(self) -> tuple[int, bytes] | None:
        """Worker-side delivery: JPEG-encode the newest decoded frame, or None when nothing new has arrived."""
        if self._reader is None:
            return None
        if self._dead:
            # The shared producer keeps writing this sensor's file; keep handing the bytes back (no decoding).
            try:
                self._reader.reclaim_all()
            except SensingStreamError as e:
                # Unreclaimable spool: stop the shared producer (every stream then reports its exit) rather
                # than let its output fill memory.
                self._rig.abort(str(e))
            return None
        try:
            if (code := self._rig.exit_code()) is not None:
                # The producer ended: flush the decoder once for its final frame, then report the exit.
                if not self._flushed:
                    self._flushed = True
                    if (last := self._reader.finish()) is not None:
                        return self._to_jpeg(last)
                raise self._fail(f"SENSING producer exited with {code}: {self._rig.diagnostics()}")
            got = self._reader.read_latest(0.0)
            if got is None:
                self._check_stall()
                return None
            return self._to_jpeg(got)
        except (SensingStreamError, SourceError, ValueError, OSError) as e:
            raise self._fail(str(e) or type(e).__name__) from e

    def _check_stall(self) -> None:
        """Fail the stream if the producer is alive but no decoded frame has arrived for too long.

        The clock starts at the first confirmed frame, so startup keeps its own deadline; a producer that stops
        writing without exiting (link loss, an encoder or plugin hang) is otherwise indistinguishable from a
        quiet camera, and a required camera would never end the session.
        """
        if self._last_frame_s is None:
            return
        silent = self._clock.monotonic() - self._last_frame_s
        if silent > max(_NO_FRAME_TIMEOUT_S, 5.0 / self._sensor.fps):
            raise SensingStreamError(
                f"no frame from the SENSING stream for {silent:.1f}s: {self._rig.diagnostics()}"
            )

    def _to_jpeg(self, got: tuple[Any, int]) -> tuple[int, bytes]:
        """Encode a decoded RGB frame at the station quality and mark the stream live: ``(host_ns, jpeg)``."""
        rgb, selected_ns = got
        jpeg = encode_rgb_jpeg(rgb, self._camera.jpeg_quality)
        self._last_frame_s = self._clock.monotonic()
        self._degraded, self._last_error = False, ""
        return selected_ns, jpeg

    async def next_camera_jpeg(self) -> tuple[int, bytes] | None:
        """Sleep to the next delivery tick, then deliver the newest frame as ``(host_ns, jpeg)`` or None.

        None means nothing new this tick (or the stream already failed; later ticks stay paced and empty
        rather than looping on exceptions). The timestamp is host monotonic time at frame selection.
        """
        assert self._deadline is not None, "next_camera_jpeg() called before enter()"
        await asyncio.sleep(max(0.0, self._deadline - self._clock.monotonic()))
        now = self._clock.monotonic()
        while self._deadline <= now:
            self._deadline += self._interval
        return await self._runner.run(self._poll_sync)

    def camera_status(self) -> p.CameraStatus:
        """This stream's honest current status; never CONNECTED before a frame was confirmed."""
        return p.CameraStatus(
            name=self._camera.name,
            stream_id=self._camera.stream_id,
            status=p.DeviceState.DEGRADED if self._degraded else p.DeviceState.CONNECTED,
            reason=p.StatusReason.NO_CURRENT_DATA if self._degraded else p.StatusReason.NONE,
            error=self._last_error if self._degraded else "",
            intrinsics=None,
        )

    async def close(self) -> bool:
        """Close the reader on its worker; False (sticky) if release could not be confirmed.

        The stop signal is raised here, synchronously, before the close job queues behind whatever the worker is
        doing, so a startup or read in progress ends promptly instead of delaying the release.
        """
        self._stop.set()

        def _close() -> None:
            """Worker-side release of the reader; a failure propagates to the awaiting caller below."""
            if self._reader is not None:
                self._reader.close()

        try:
            confirmed = await self._runner.close(_close)
        except Exception:
            self.cleanup_failed = True
            raise
        if not confirmed:
            self.cleanup_failed = True
        return confirmed and not self.cleanup_failed

# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""SENSING configuration, inspection, rig ownership, adapter lifecycle and Acquisition composition.

CPU-only: a fake producer writing real intra-only H.264 files stands in for the native binary,
so these do not replace the desktop MOCK-binary or Orin integration runs.
"""

from __future__ import annotations

import asyncio
import io
import os
import subprocess
import threading
import time
import types

import pytest
from PIL import Image

from capture_server import protocol as p
from capture_server.acquisition import Acquisition, CameraSourceError
from capture_server.acquisition import sensing as sn
from capture_server.acquisition import sensing_h264 as h264
from capture_server.acquisition.errors import SourceError
from capture_server.server import CaptureServer, ServerConfig, StationCamera

from .h264_fixtures import FileProducer, allocated_bytes, encode_units

W, H = 64, 32

RIG = dict(
    executable="/opt/x/camera_plugin_sensing",
    expected_backend="MOCK",
    platform_config="/opt/x/p.json",
    config_name="CFG",
    link_masks=("0x0000", "0x1100"),
)
DESCRIPTION = sn.SensingDescription((sn.SensingSensor("L", 1, W, H, 30.0), sn.SensingSensor("R", 0, W, H, 30.0)))


def _listing(*lines: str, header: str = "CFG: 2 sensor(s)") -> str:
    """The native plugin's ``--list-sensors`` text: a header, then one line per sensor."""
    return "\n".join([header, *lines]) + "\n"


GOOD_LISTING = _listing(
    f"  sensor=1  L  {W}x{H} @ 30 fps",
    f"  sensor=0  R  {W}x{H} @ 30 fps",
)


# --- configuration --------------------


@pytest.mark.parametrize(
    "over", [dict(expected_backend="CPU"), dict(executable=""), dict(link_masks=()), dict(config_name="")]
)
def test_rig_config_rejects_bad_values(over):
    with pytest.raises(ValueError):
        sn.SensingRigConfig(**{**RIG, **over})


@pytest.mark.parametrize("over", [dict(module_name=""), dict(delivery_fps=0), dict(jpeg_quality=101)])
def test_station_camera_rejects_bad_values(over):
    with pytest.raises(ValueError):
        sn.SensingStationCamera(**{"module_name": "L", **over})


def test_server_config_dedups_within_each_identity_domain_only():
    rig = sn.SensingRigConfig(**RIG)
    cams = {
        "a": StationCamera("mxid-1"),
        "b": sn.SensingStationCamera("mxid-1"),  # same string, different identity domain: fine
    }
    ServerConfig(mock=True, cameras=cams, sensing_rig=rig)
    with pytest.raises(ValueError, match="more than one logical name"):
        ServerConfig(
            mock=True,
            cameras={"a": sn.SensingStationCamera("L"), "b": sn.SensingStationCamera("L")},
            sensing_rig=rig,
        )


def test_server_config_allows_both_eyes_of_one_module_but_not_the_same_eye_twice():
    rig = sn.SensingRigConfig(**RIG)
    eyes = {
        "head_left": sn.SensingStationCamera("HEAD", sensor_index=0),
        "head_right": sn.SensingStationCamera("HEAD", sensor_index=1),
    }
    ServerConfig(mock=True, cameras=eyes, sensing_rig=rig)
    twice = {
        "a": sn.SensingStationCamera("HEAD", sensor_index=0),
        "b": sn.SensingStationCamera("HEAD", sensor_index=0),
    }
    with pytest.raises(ValueError, match="more than one logical name"):
        ServerConfig(mock=True, cameras=twice, sensing_rig=rig)
    # One name selecting the module as a whole and another selecting one eye is also a conflict to resolve later.
    mixed = {"a": sn.SensingStationCamera("HEAD"), "b": sn.SensingStationCamera("HEAD", sensor_index=1)}
    ServerConfig(mock=True, cameras=mixed, sensing_rig=rig)  # config-level distinct; resolve_sensors rejects it


def test_server_config_requires_rig_and_matching_backend():
    with pytest.raises(ValueError, match="no sensing_rig"):
        ServerConfig(mock=True, cameras={"a": sn.SensingStationCamera("L")})
    with pytest.raises(ValueError, match="must be JETSON"):
        ServerConfig(mock=False, sensing_rig=sn.SensingRigConfig(**RIG))
    with pytest.raises(ValueError, match="must be MOCK"):
        ServerConfig(mock=True, sensing_rig=sn.SensingRigConfig(**{**RIG, "expected_backend": "JETSON"}))


def test_every_documented_rig_key_loads_from_yaml(tmp_path):
    f = tmp_path / "s.yaml"
    f.write_text(
        "sensing_rig:\n  executable: /x/e\n  expected_backend: MOCK\n  platform_config: /x/p.json\n"
        "  config_name: C\n  link_masks: ['0x0000']\n  gpu_id: 1\n  nito_dir: /n\n"
        "  bitrate_bps: 50000000\n  full_range: true\n  runtime_dir: /r\n"
        "cameras:\n  left: {type: SENSING_COLOR, module_name: L}\n"
    )
    rig, _ = sn.load_sensing_station(str(f))
    assert (rig.gpu_id, rig.nito_dir, rig.bitrate_bps, rig.full_range, rig.runtime_dir) == (
        1,
        "/n",
        50_000_000,
        True,
        "/r",
    )
    args = rig.base_args()
    assert "--full-range" in args and "--bitrate=50000000" in args and "--peak-bitrate=75000000" in args


def test_configured_encoder_settings_reach_the_launch_arguments():
    args = sn.SensingRigConfig(**RIG, gop=30, bitrate_bps=20_000_000).base_args()
    assert "--gop=30" in args and "--bitrate=20000000" in args and "--peak-bitrate=30000000" in args


def test_load_station_yaml(tmp_path):
    f = tmp_path / "s.yaml"
    f.write_text(
        "sensing_rig:\n  executable: /x/e\n  expected_backend: MOCK\n  platform_config: /x/p.json\n"
        "  config_name: C\n  link_masks: ['0x0000', '0x1100']\n"
        "cameras:\n  left: {type: SENSING_COLOR, module_name: L, delivery_fps: 15}\n"
    )
    rig, cams = sn.load_sensing_station(str(f))
    assert rig.link_masks == ("0x0000", "0x1100") and cams["left"].delivery_fps == 15.0


@pytest.mark.parametrize(
    "text",
    [
        "cameras: {}\n",
        "sensing_rig: {executable: e}\ncameras: {}\n",
        "sensing_rig: {executable: e, bogus: 1}\ncameras: {l: {type: SENSING_COLOR, module_name: L}}\n",
    ],
)
def test_load_station_yaml_rejects_bad_files(tmp_path, text):
    f = tmp_path / "s.yaml"
    f.write_text(text)
    with pytest.raises(ValueError):
        sn.load_sensing_station(str(f))


def test_load_station_yaml_rejects_non_sensing_camera(tmp_path):
    f = tmp_path / "s.yaml"
    f.write_text(
        "sensing_rig: {executable: e, expected_backend: MOCK, platform_config: p, config_name: c,"
        " link_masks: ['0x0']}\n"
        "cameras: {l: {type: OAK_D_COLOR, module_name: L}}\n"
    )
    with pytest.raises(ValueError, match="SENSING_COLOR"):
        sn.load_sensing_station(str(f))


# --- inspection and resolution --------------------


def _fake_run(stdout="", returncode=0, stderr="", exc=None):
    def run(args, **kw):
        assert args[-1] == "--list-sensors" and kw["timeout"] > 0 and "shell" not in kw
        if exc:
            raise exc
        return subprocess.CompletedProcess(args, returncode, stdout, stderr)

    return run


def test_inspect_rig_happy_path_uses_argument_list_and_the_existing_listing():
    d = sn.inspect_rig(sn.SensingRigConfig(**RIG), run=_fake_run(GOOD_LISTING))
    assert [(s.module_name, s.pipeline_id, s.width, s.height) for s in d.sensors] == [
        ("L", 1, W, H),
        ("R", 0, W, H),
    ]


@pytest.mark.parametrize(
    "run,msg",
    [
        (_fake_run(returncode=2, stderr="bad masks"), "bad masks"),
        (_fake_run("not a listing"), "malformed"),
        (_fake_run(_listing(f"  sensor=0  L  {W}x{H} @ 30 fps")), "header says 2 sensors, found 1"),
        (_fake_run(_listing("  sensor=0  L  garbage", "  sensor=1  R  1x1 @ 1 fps")), "unexpected line"),
        (
            _fake_run(_listing(f"  sensor=0  L  0x{H} @ 30 fps", f"  sensor=1  R  {W}x{H} @ 30 fps")),
            "non-positive",
        ),
        (_fake_run(exc=subprocess.TimeoutExpired("x", 1)), "could not inspect"),
        (_fake_run(exc=FileNotFoundError("nope")), "could not inspect"),
    ],
)
def test_inspect_rig_failures(run, msg):
    with pytest.raises(SourceError, match=msg):
        sn.inspect_rig(sn.SensingRigConfig(**RIG), run=run)


@pytest.mark.parametrize(
    "bad", ["", "\n\n", "CFG: x sensor(s)\n", "CFG: 1 sensor(s)\n  sensor=-1  L  1x1 @ 30 fps\n"]
)
def test_parse_sensor_listing_rejects_instead_of_guessing(bad):
    with pytest.raises(ValueError, match="malformed"):
        sn.parse_sensor_listing(bad)


def test_resolve_sensors_uses_module_names_not_positions():
    got = sn.resolve_sensors(
        DESCRIPTION, {"left": sn.SensingStationCamera("L"), "right": sn.SensingStationCamera("R")}
    )
    assert got["left"].pipeline_id == 1 and got["right"].pipeline_id == 0


def test_sensor_index_selects_one_eye_of_a_stereo_module_in_pipeline_order():
    stereo = sn.SensingDescription(
        (
            sn.SensingSensor("HEAD", 1, W, H, 30.0),
            sn.SensingSensor("HEAD", 0, W, H, 30.0),
            sn.SensingSensor("WRIST", 2, W, H, 30.0),
        ),
    )
    got = sn.resolve_sensors(
        stereo,
        {
            "head_left_eye": sn.SensingStationCamera("HEAD", sensor_index=0),
            "head_right_eye": sn.SensingStationCamera("HEAD", sensor_index=1),
            "wrist": sn.SensingStationCamera("WRIST"),
        },
    )
    assert [got[n].pipeline_id for n in ("head_left_eye", "head_right_eye", "wrist")] == [0, 1, 2]
    with pytest.raises(SourceError, match="select one with sensor_index"):
        sn.resolve_sensors(stereo, {"head": sn.SensingStationCamera("HEAD")})
    with pytest.raises(SourceError, match="no sensor_index 2"):
        sn.resolve_sensors(stereo, {"head": sn.SensingStationCamera("HEAD", sensor_index=2)})
    with pytest.raises(SourceError, match="more than one logical"):
        sn.resolve_sensors(
            stereo,
            {
                "a": sn.SensingStationCamera("HEAD", sensor_index=0),
                "b": sn.SensingStationCamera("HEAD", sensor_index=0),
            },
        )
    with pytest.raises(ValueError):
        sn.SensingStationCamera("HEAD", sensor_index=-1)


def test_resolve_sensors_rejects_missing_ambiguous_and_duplicate():
    with pytest.raises(SourceError, match="not in the platform"):
        sn.resolve_sensors(DESCRIPTION, {"x": sn.SensingStationCamera("Z")})
    amb = sn.SensingDescription((sn.SensingSensor("L", 0, W, H, 30.0), sn.SensingSensor("L", 1, W, H, 30.0)))
    with pytest.raises(SourceError, match="select one with sensor_index"):
        sn.resolve_sensors(amb, {"x": sn.SensingStationCamera("L")})
    with pytest.raises(SourceError, match="more than one logical"):
        sn.resolve_sensors(DESCRIPTION, {"a": sn.SensingStationCamera("L"), "b": sn.SensingStationCamera("L")})


# --- rig ownership --------------------


class FakeProc:
    def __init__(self, *, ignore_terminate=False, ignore_kill=False) -> None:
        self.code: int | None = None
        self.stdout = io.BytesIO(b"line one\nline two\n")
        self.ignore_terminate, self.ignore_kill = ignore_terminate, ignore_kill
        self.calls: list[str] = []

    def poll(self):
        return self.code

    def terminate(self):
        self.calls.append("terminate")
        if not self.ignore_terminate:
            self.code = -15

    def kill(self):
        self.calls.append("kill")
        if not self.ignore_kill:
            self.code = -9

    def wait(self, timeout=None):
        if self.code is None:
            raise subprocess.TimeoutExpired("p", timeout)
        return self.code


def _rig(tmp_path, proc, popen=None):
    captured = {}

    def default_popen(args, **kw):
        captured["args"], captured["kw"] = args, kw
        return proc

    rig = sn.SensingRig(
        sn.SensingRigConfig(**RIG), (1, 0), popen=popen or default_popen, runtime_root=str(tmp_path)
    )
    return rig, captured


@pytest.mark.asyncio
async def test_rig_start_builds_file_output_argument_list_and_cleans_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSING_MOCK_SCALE_DIV", "4")
    rig, cap = _rig(tmp_path, FakeProc())
    await rig.start()
    args = cap["args"]
    assert isinstance(args, list) and args[0] == RIG["executable"] and rig.start_error is None
    assert "--link-masks=0x0000 0x1100" in args
    # Encoder settings are passed only when configured; the decoder does not require any particular GOP.
    assert not any(a.startswith(("--gop", "--bitrate", "--peak-bitrate")) for a in args)
    streams = [a for a in args if a.startswith("--add-stream=")]
    assert len(streams) == 2 and all(",output=" in a and "ipc=" not in a for a in streams)
    assert streams[0].startswith("--add-stream=sensor=1,") and rig.stream_path(1) in streams[0]
    assert "SENSING_MOCK_SCALE_DIV" not in cap["kw"]["env"] and "shell" not in cap["kw"]
    run_dir = os.path.dirname(rig.stream_path(1))
    assert os.stat(run_dir).st_mode & 0o777 == 0o700
    await asyncio.sleep(0.05)  # let the log drain read the fake output
    assert "line two" in rig.diagnostics()
    assert await rig.close() and rig.cleanup_ok
    assert not os.path.exists(run_dir)


@pytest.mark.asyncio
async def test_rig_close_escalates_to_kill_and_removes_only_its_directory(tmp_path):
    other = tmp_path / "keep"
    other.mkdir()
    proc = FakeProc(ignore_terminate=True)
    rig, _ = _rig(tmp_path, proc)
    await rig.start()
    run_dir = os.path.dirname(rig.stream_path(1))
    assert await rig.close() and proc.calls == ["terminate", "kill"] and other.exists()
    assert not os.path.exists(run_dir)


@pytest.mark.asyncio
async def test_rig_unconfirmed_exit_stays_false_on_every_later_call(tmp_path, monkeypatch):
    monkeypatch.setattr(sn, "_RIG_STEPS_S", 0.1)
    rig, _ = _rig(tmp_path, FakeProc(ignore_terminate=True, ignore_kill=True))
    await rig.start()
    assert await rig.close() is False and rig.cleanup_ok is False
    assert await rig.close() is False  # never reported as released just because a wait ended


@pytest.mark.asyncio
async def test_rig_close_stays_inside_one_total_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(sn, "_RIG_STEPS_S", 0.3)
    rig, _ = _rig(tmp_path, FakeProc(ignore_terminate=True, ignore_kill=True))
    await rig.start()
    started = time.monotonic()
    assert await rig.close() is False
    assert time.monotonic() - started < 1.5


@pytest.mark.asyncio
async def test_rig_start_failure_is_recorded_not_raised_and_leaves_no_directory(tmp_path):
    def boom(*a, **k):
        raise FileNotFoundError("no binary")

    rig, _ = _rig(tmp_path, None, popen=boom)
    await rig.start()
    assert "could not start" in rig.start_error and list(tmp_path.iterdir()) == []
    assert await rig.close() is True


@pytest.mark.asyncio
async def test_slow_spawn_still_in_flight_at_stop_is_reaped_by_the_queued_close(tmp_path):
    release = threading.Event()
    proc = FakeProc()

    def slow_popen(args, **kw):
        release.wait(5)
        return proc

    rig, _ = _rig(tmp_path, None, popen=slow_popen)
    start = asyncio.ensure_future(rig.start())
    await asyncio.sleep(0.05)
    start.cancel()  # Stop/cancellation while the spawn is still running on the worker
    threading.Timer(0.2, release.set).start()
    assert await rig.close() is True
    assert proc.calls == ["terminate"] and list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_log_tail_is_byte_bounded_decoded_leniently_and_closed(tmp_path):
    proc = FakeProc()
    proc.stdout = io.BytesIO(b"x" * 100_000 + b"\xff\xfe tail-marker")
    rig, _ = _rig(tmp_path, proc)
    await rig.start()
    await asyncio.sleep(0.1)
    text = rig.diagnostics()
    assert len(text) <= sn._LOG_TAIL_BYTES and text.endswith("tail-marker")
    assert proc.stdout.closed
    await rig.close()


# --- adapter lifecycle -------------------------------------------------------------------------------------------


class FakeRig:
    start_error: str | None = None

    def __init__(self, path: str) -> None:
        self.path, self.code = path, None
        self.aborted: list[str] = []

    def abort(self, reason: str) -> None:
        self.aborted.append(reason)

    def stream_path(self, _pid: int) -> str:
        return self.path

    def exit_code(self):
        return self.code

    def diagnostics(self) -> str:
        return "producer log"


def _camera(required=True, fps=200.0) -> p.CameraStreamConfig:
    return p.CameraStreamConfig(
        stream_id=1,
        name="cam",
        type=p.CameraType.SENSING_COLOR,
        required=required,
        width=W,
        height=H,
        fps=fps,
        jpeg_quality=80,
    )


def _adapter(tmp_path, **kw):
    rig = FakeRig(str(tmp_path / "sensor1.h264"))
    adapter = sn.SensingAdapter(
        rig=rig,
        sensor=sn.SensingSensor("M", 1, W, H, 30.0),
        camera=kw.pop("camera", _camera()),
        startup_timeout_s=kw.pop("startup_timeout_s", 5.0),
    )
    return adapter, rig


RED, GREEN, BLUE = (220, 20, 20), (20, 220, 20), (20, 20, 220)


@pytest.mark.asyncio
async def test_adapter_delivers_decodable_jpeg_and_closes_cleanly(tmp_path):
    adapter, rig = _adapter(tmp_path)
    assert adapter.camera_status().status == p.DeviceState.DEGRADED  # never CONNECTED before a frame
    producer = FileProducer(rig.path, encode_units(W, H, [RED, GREEN, BLUE]), loop=True).start()
    try:
        await adapter.enter()
        assert adapter.camera_status().status == p.DeviceState.CONNECTED
        got = None
        for _ in range(100):
            got = await adapter.next_camera_jpeg()
            if got:
                break
        ts, jpeg = got
        img = Image.open(io.BytesIO(jpeg))
        assert img.size == (W, H) and ts > 0
        assert await adapter.close() is True and not adapter.cleanup_failed
    finally:
        producer.stop()


@pytest.mark.asyncio
async def test_producer_exit_during_startup_is_a_camera_error(tmp_path):
    adapter, rig = _adapter(tmp_path, startup_timeout_s=3.0)
    rig.code = 3
    with pytest.raises(CameraSourceError, match="exited with 3: producer log"):
        await adapter.enter()
    assert adapter.camera_status().status == p.DeviceState.DEGRADED
    assert await adapter.close() is True


@pytest.mark.asyncio
async def test_startup_deadline_without_a_frame(tmp_path):
    adapter, rig = _adapter(tmp_path, startup_timeout_s=0.5)
    open(rig.path, "wb").close()  # the file exists but no unit is ever written
    with pytest.raises(CameraSourceError, match="no frame"):
        await adapter.enter()
    await adapter.close()


@pytest.mark.asyncio
async def test_undecodable_producer_output_is_a_camera_error_not_a_synthetic_frame(tmp_path):
    adapter, rig = _adapter(tmp_path, startup_timeout_s=2.0)
    # Annex-B shaped but not video: the placeholder bytes an older mock encoder wrote.
    with open(rig.path, "wb") as f:
        f.write((b"\x00\x00\x00\x01\x65MOCK" + b"U" * 300) * 6)
    with pytest.raises(CameraSourceError, match="failed at startup|no frame"):
        await adapter.enter()
    assert adapter.camera_status().status == p.DeviceState.DEGRADED
    await adapter.close()


@pytest.mark.asyncio
async def test_mid_session_producer_exit_then_paced_empty_polls(tmp_path):
    adapter, rig = _adapter(tmp_path, camera=_camera(required=False, fps=50.0))
    producer = FileProducer(rig.path, encode_units(W, H, [RED, GREEN]), loop=True).start()
    try:
        await adapter.enter()
        rig.code = 1
        with pytest.raises(CameraSourceError, match="exited with 1"):
            for _ in range(20):
                await adapter.next_camera_jpeg()
        assert adapter.camera_status().status == p.DeviceState.DEGRADED
        started = time.monotonic()
        for _ in range(3):  # after the failure: no exception loop, still one tick per interval
            assert await adapter.next_camera_jpeg() is None
        assert time.monotonic() - started >= 0.03
        await adapter.close()
    finally:
        producer.stop()


@pytest.mark.asyncio
async def test_failed_close_is_sticky_even_after_stream_degradation(tmp_path):
    adapter, rig = _adapter(tmp_path)
    producer = FileProducer(rig.path, encode_units(W, H, [RED, GREEN]), loop=True).start()
    try:
        await adapter.enter()

        def leaky_close():
            raise OSError("could not release")

        adapter._reader.close = leaky_close
        with pytest.raises(OSError, match="could not release"):
            await adapter.close()
        assert adapter.cleanup_failed
    finally:
        producer.stop()


# --- Acquisition ownership --------------------


class SpyAdapter:
    stream_id, required = 1, False

    def __init__(
        self,
        order,
        name,
        fail=False,
        gate: threading.Event | None = None,
        enter_error=None,
        required=False,
        borrows_rig=True,
    ):
        self.order, self.name, self.fail, self.gate = order, name, fail, gate
        self.enter_error, self.required, self.borrows_rig = enter_error, required, borrows_rig

    async def enter(self):
        if self.enter_error:
            raise CameraSourceError(self.enter_error, camera_name=self.name)

    async def close(self):
        self.order.append(f"{self.name}:close-start")
        if self.gate is not None:
            await asyncio.get_running_loop().run_in_executor(None, self.gate.wait, 5)
        self.order.append(f"{self.name}:close-done")
        if self.fail:
            raise RuntimeError("release failed")
        return True

    def camera_status(self):
        raise AssertionError


class SpyRig:
    def __init__(self, order, ok=True, start_error=None):
        self.order, self.ok, self.started, self.start_error = order, ok, False, start_error

    async def start(self):
        self.started = True

    async def close(self):
        self.order.append("rig:close")
        return self.ok


@pytest.mark.asyncio
async def test_acquisition_starts_rig_then_closes_adapters_before_rig():
    order: list[str] = []
    rig = SpyRig(order)
    acq = Acquisition(tracking=None, cameras=(SpyAdapter(order, "a"),), sensing_rig=rig)
    await acq.enter()
    assert rig.started
    assert await acq.close() is True and order == ["a:close-start", "a:close-done", "rig:close"]


@pytest.mark.asyncio
async def test_rig_is_not_terminated_while_an_adapter_close_is_still_held():
    order: list[str] = []
    gate = threading.Event()
    acq = Acquisition(tracking=None, cameras=(SpyAdapter(order, "a", gate=gate),), sensing_rig=SpyRig(order))
    closing = asyncio.ensure_future(acq.close())
    await asyncio.sleep(0.2)
    assert order == ["a:close-start"]  # producer teardown has NOT begun while the adapter is still closing
    gate.set()
    assert await closing is True
    assert order == ["a:close-start", "a:close-done", "rig:close"]


@pytest.mark.asyncio
async def test_acquisition_reaps_rig_even_if_an_adapter_close_raises_and_reports_unconfirmed_rig():
    order: list[str] = []
    acq = Acquisition(tracking=None, cameras=(SpyAdapter(order, "a", fail=True),), sensing_rig=SpyRig(order))
    with pytest.raises(RuntimeError, match="release failed"):
        await acq.close()
    assert order[-1] == "rig:close"
    acq2 = Acquisition(tracking=None, cameras=(SpyAdapter([], "a"),), sensing_rig=SpyRig([], ok=False))
    assert await acq2.close() is False


@pytest.mark.asyncio
async def test_shared_producer_start_failure_degrades_optional_and_fails_required_streams(tmp_path):
    class DeadRig(FakeRig):
        start_error = "could not start /x: no binary"

    def make(required):
        return sn.SensingAdapter(
            rig=DeadRig("/nope"),
            sensor=sn.SensingSensor("M", 1, W, H, 30.0),
            camera=_camera(required=required),
        )

    healthy_oak = SpyAdapter([], "oak")
    optional = make(False)
    await Acquisition(tracking=None, cameras=(healthy_oak, optional), sensing_rig=None).enter()  # tolerated
    assert optional.camera_status().status == p.DeviceState.DEGRADED
    assert "did not start" in optional.camera_status().error
    with pytest.raises(CameraSourceError, match="did not start"):
        await Acquisition(tracking=None, cameras=(healthy_oak, make(True)), sensing_rig=None).enter()
    for adapter in (optional,):
        assert await adapter.close() is True


# --- server type validation and launch selection --------------------


def _server(mock=True, cameras=None):
    cfg = ServerConfig(mock=mock, cameras=cameras, sensing_rig=sn.SensingRigConfig(**RIG) if cameras else None)
    return CaptureServer(cfg, source_factory=lambda *_: None)


def _begin(*cams):
    return p.BeginRequest(
        session_id="s1",
        tracking=p.TrackingConfig(sources=(), rate_hz=30.0),
        cameras=tuple(p.CameraRequest(name=n, type=t) for n, t in cams),
    )


@pytest.mark.asyncio
async def test_prepare_sensing_rejects_type_mismatch_before_inspection(monkeypatch):
    monkeypatch.setattr("capture_server.server.inspect_rig", lambda *_: pytest.fail("must not inspect"))
    server = _server(cameras={"left": sn.SensingStationCamera("L")})
    err = await server._prepare_sensing(_begin(("left", p.CameraType.OAK_D_COLOR)))
    assert "SENSING_COLOR station camera" in err


@pytest.mark.asyncio
async def test_prepare_sensing_inspects_once_and_caches(monkeypatch):
    calls = []

    def fake_inspect(cfg):
        calls.append(cfg)
        return DESCRIPTION

    monkeypatch.setattr("capture_server.server.inspect_rig", fake_inspect)
    server = _server(cameras={"left": sn.SensingStationCamera("L")})
    begin = _begin(("left", p.CameraType.SENSING_COLOR))
    assert await server._prepare_sensing(begin) is None and await server._prepare_sensing(begin) is None
    assert len(calls) == 1
    cam = server._camera_stream_config(1, begin.cameras[0])
    assert (cam.width, cam.height, cam.fps) == (W, H, 30.0) and cam.type == p.CameraType.SENSING_COLOR


@pytest.mark.asyncio
async def test_prepare_sensing_failure_is_reported_and_not_cached(monkeypatch):
    def boom(_cfg):
        raise SourceError("backend mismatch")

    monkeypatch.setattr("capture_server.server.inspect_rig", boom)
    server = _server(cameras={"left": sn.SensingStationCamera("L")})
    err = await server._prepare_sensing(_begin(("left", p.CameraType.SENSING_COLOR)))
    assert "backend mismatch" in err and server.config.sensing_description is None


@pytest.mark.asyncio
async def test_oak_only_begin_needs_no_sensing(monkeypatch):
    monkeypatch.setattr("capture_server.server.inspect_rig", lambda *_: pytest.fail("must not inspect"))
    server = _server(cameras={"w": StationCamera("mxid")})
    assert await server._prepare_sensing(_begin(("w", p.CameraType.OAK_D_COLOR))) is None


def test_launch_builds_no_rig_for_oak_only_sessions():
    from capture_server.launch import _sensing_composition

    cfg = ServerConfig(mock=True)
    assert _sensing_composition(cfg, (_camera(),)) == (None, {})


def test_launch_mixes_sensing_and_mock_oak_in_one_acquisition():
    from capture_server.launch import build_source_factory

    cfg = ServerConfig(
        mock=True,
        cameras={"left": sn.SensingStationCamera("L")},
        sensing_rig=sn.SensingRigConfig(**RIG),
        sensing_description=DESCRIPTION,
    )
    factory = build_source_factory(cfg)
    begin = _begin(("left", p.CameraType.SENSING_COLOR), ("wrist", p.CameraType.OAK_D_COLOR))
    cams = (
        p.CameraStreamConfig(1, "left", p.CameraType.SENSING_COLOR, True, W, H, 30.0, 80),
        p.CameraStreamConfig(2, "wrist", p.CameraType.OAK_D_COLOR, True, 1280, 720, 30.0, 80),
    )
    acq = factory(begin, cams)
    assert isinstance(acq.cameras[0], sn.SensingAdapter) and not isinstance(acq.cameras[1], sn.SensingAdapter)
    assert acq.sensing_rig is not None


# --- Begin admission through the real server -----------


async def _begin_rejection(url: str, camera_type: p.CameraType, name: str = "left") -> p.RequestRejected:
    """Send one real Begin with a single camera over a raw socket; return the RequestRejected it draws."""
    import json

    import websockets

    begin = p.BeginRequest(
        tracking=p.TrackingConfig(sources=()), cameras=(p.CameraRequest(name=name, type=camera_type),)
    )
    async with websockets.connect(url) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": int(p.MessageType.BEGIN),
                    "version": p.PROTOCOL_VERSION,
                    "session_id": "sensing-begin-1",
                    **p.effective_config(begin).to_json(),
                }
            )
        )
        for _ in range(20):
            raw = await asyncio.wait_for(ws.recv(), timeout=3)
            if isinstance(raw, bytes):
                continue
            event = p.parse_server_message(json.loads(raw))
            if isinstance(event, p.RequestRejected):
                return event
    raise AssertionError("no RequestRejected arrived")


@pytest.mark.asyncio
async def test_sensing_request_without_a_station_entry_is_rejected_not_built_as_a_mock_oak(unused_tcp_port):
    from tests.support.capture_server import start_capture_server, stop_capture_server

    running = await start_capture_server(unused_tcp_port)  # permissive legacy config: cameras=None
    try:
        rejected = await _begin_rejection(f"ws://127.0.0.1:{unused_tcp_port}", p.CameraType.SENSING_COLOR)
        assert rejected.reason == p.RejectionReason.INVALID_CONFIG
        assert "no SENSING station entry" in rejected.error_message
        assert running.server._active_session is None
    finally:
        await stop_capture_server(running)


@pytest.mark.asyncio
async def test_shutdown_during_a_paused_inspection_never_admits_a_session(unused_tcp_port, monkeypatch):
    from tests.support.capture_server import start_capture_server, stop_capture_server

    started, release = threading.Event(), threading.Event()

    def slow_inspect(_cfg):
        started.set()
        release.wait(5)
        return DESCRIPTION

    monkeypatch.setattr("capture_server.server.inspect_rig", slow_inspect)
    cfg = ServerConfig(
        mock=True, cameras={"left": sn.SensingStationCamera("L")}, sensing_rig=sn.SensingRigConfig(**RIG)
    )
    running = await start_capture_server(unused_tcp_port, config=cfg)
    try:
        pending = asyncio.ensure_future(
            _begin_rejection(f"ws://127.0.0.1:{unused_tcp_port}", p.CameraType.SENSING_COLOR)
        )
        await asyncio.get_running_loop().run_in_executor(None, started.wait, 5)
        running.server.request_stop()  # shutdown begins while the Begin is parked in inspection
        release.set()
        rejected = await pending
        assert rejected.reason == p.RejectionReason.BUSY and "shutting down" in rejected.error_message
        assert running.server._active_session is None
    finally:
        release.set()
        await stop_capture_server(running)


@pytest.mark.asyncio
async def test_concurrent_first_begins_share_one_inspection(monkeypatch):
    calls: list[int] = []
    release = threading.Event()

    def slow_inspect(_cfg):
        calls.append(1)
        release.wait(5)
        return DESCRIPTION

    monkeypatch.setattr("capture_server.server.inspect_rig", slow_inspect)
    server = _server(cameras={"left": sn.SensingStationCamera("L")})
    begin = _begin(("left", p.CameraType.SENSING_COLOR))
    both = asyncio.gather(server._prepare_sensing(begin), server._prepare_sensing(begin))
    await asyncio.sleep(0.1)
    release.set()
    assert await both == [None, None] and len(calls) == 1


@pytest.mark.asyncio
async def test_a_begin_that_is_already_busy_triggers_no_inspection(monkeypatch):
    monkeypatch.setattr("capture_server.server.inspect_rig", lambda *_: pytest.fail("must not inspect"))
    server = _server(cameras={"left": sn.SensingStationCamera("L")})
    server._active_session = object()  # a session is already running
    assert await server._prepare_sensing(_begin(("left", p.CameraType.SENSING_COLOR))) is None


# --- shared shutdown budget and local start failures -------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_slow_oak_release_is_not_cut_short_by_the_producer_shutdown(monkeypatch):
    from capture_server.acquisition import acquisition as acq_mod

    monkeypatch.setattr(acq_mod, "_CLOSE_BUDGET_S", 0.8)
    order: list[str] = []
    gate = threading.Event()
    threading.Timer(0.45, gate.set).start()  # a release taking 0.45 s of a 0.8 s budget
    oak = SpyAdapter(order, "oak", gate=gate, borrows_rig=False)
    acq = Acquisition(tracking=None, cameras=(oak, SpyAdapter(order, "sensing")), sensing_rig=SpyRig(order))
    started = time.monotonic()
    assert await acq.close() is True  # the old shared budget cut OAK at budget minus the producer's reserve
    assert time.monotonic() - started < 0.8
    assert order.count("oak:close-done") == 1 and order[-1] in ("rig:close", "oak:close-done")


@pytest.mark.asyncio
async def test_a_held_oak_release_does_not_delay_the_producer_reap_and_stays_unconfirmed(monkeypatch):
    from capture_server.acquisition import acquisition as acq_mod

    monkeypatch.setattr(acq_mod, "_CLOSE_BUDGET_S", 0.4)
    order: list[str] = []
    gate = threading.Event()
    oak = SpyAdapter(order, "oak", gate=gate, borrows_rig=False)
    acq = Acquisition(tracking=None, cameras=(oak, SpyAdapter(order, "sensing")), sensing_rig=SpyRig(order))
    started = time.monotonic()
    assert await acq.close() is False  # the OAK release is still pending: unconfirmed, admission stays blocked
    assert time.monotonic() - started < 0.9
    assert order.index("rig:close") < order.index("oak:close-done") if "oak:close-done" in order else True
    assert "rig:close" in order and "sensing:close-done" in order  # the producer chain was not held up by OAK
    gate.set()  # the retained OAK close finishes later and is observed, not dropped
    await asyncio.sleep(0.1)
    assert order.count("oak:close-done") == 1


@pytest.mark.asyncio
async def test_runtime_directory_failures_are_start_failures_and_leave_nothing_behind(tmp_path, monkeypatch):
    rig = sn.SensingRig(
        sn.SensingRigConfig(**RIG), (1,), popen=lambda *a, **k: FakeProc(), runtime_root=str(tmp_path / "missing")
    )
    await rig.start()
    assert "runtime directory" in rig.start_error and await rig.close() is True

    monkeypatch.setattr(os, "chmod", lambda *a, **k: (_ for _ in ()).throw(PermissionError("denied")))
    rig2 = sn.SensingRig(
        sn.SensingRigConfig(**RIG), (1,), popen=lambda *a, **k: FakeProc(), runtime_root=str(tmp_path)
    )
    await rig2.start()
    assert "denied" in rig2.start_error and list(tmp_path.iterdir()) != []  # created, so retained for cleanup
    assert await rig2.close() is True and list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_cancelled_close_still_reaps_the_producer_and_observes_late_failures():
    import gc

    unobserved: list[dict] = []
    asyncio.get_running_loop().set_exception_handler(lambda _loop, ctx: unobserved.append(ctx))
    order: list[str] = []
    gate = threading.Event()
    acq = Acquisition(
        tracking=None,
        cameras=(SpyAdapter(order, "slow", gate=gate), SpyAdapter(order, "late-fail", fail=True, gate=gate)),
        sensing_rig=SpyRig(order),
    )
    closing = asyncio.ensure_future(acq.close())
    await asyncio.sleep(0.1)
    closing.cancel()  # the caller gives up while both adapter closes are still held
    with pytest.raises(asyncio.CancelledError):
        await closing
    gate.set()  # the held closes finish later; one of them fails with nobody awaiting it
    for _ in range(100):
        if "rig:close" in order and "late-fail:close-done" in order:
            break
        await asyncio.sleep(0.02)
    assert "rig:close" in order  # the producer was still reaped
    del acq, closing
    gc.collect()
    await asyncio.sleep(0.05)
    assert unobserved == []  # no "exception was never retrieved"


# --- strict station file, shared producer environment, factory misuse, cancellation precedence -------------------


def _yaml(tmp_path, rig_extra="", camera="{type: SENSING_COLOR, module_name: L}", masks="['0x0000']"):
    f = tmp_path / "s.yaml"
    f.write_text(
        "sensing_rig: {executable: e, expected_backend: MOCK, platform_config: p, config_name: c,"
        f" link_masks: {masks}{rig_extra}}}\ncameras: {{l: {camera}}}\n"
    )
    return str(f)


@pytest.mark.parametrize(
    "kw,msg",
    [
        (dict(masks="[0x0000, 0x1100]"), "quoted strings"),  # unquoted masks are YAML integers
        (dict(rig_extra=", full_range: yes-please"), "full_range must be bool"),
        (dict(rig_extra=", gpu_id: true"), "gpu_id must be int"),
        (dict(camera="{type: SENSING_COLOR, module_name: L, jpeg_quality: 80.5}"), "jpeg_quality must be int"),
        (dict(camera="{type: SENSING_COLOR, module_name: L, required: 1}"), "required must be bool"),
        (
            dict(camera="{type: SENSING_COLOR, module_name: L, delivery_fps: true}"),
            "delivery_fps must be int/float",
        ),
        (dict(camera="{type: SENSING_COLOR, module_name: 7}"), "module_name must be str"),
    ],
)
def test_station_yaml_scalars_are_type_checked(tmp_path, kw, msg):
    with pytest.raises(ValueError, match=msg):
        sn.load_sensing_station(_yaml(tmp_path, **kw))


def test_a_file_that_is_not_yaml_is_a_value_error_not_a_traceback(tmp_path):
    f = tmp_path / "s.yaml"
    f.write_text("sensing_rig: [unclosed\n")
    with pytest.raises(ValueError, match="not valid YAML"):
        sn.load_sensing_station(str(f))


def test_the_listing_and_the_capture_run_share_one_sanitized_environment(monkeypatch):
    monkeypatch.setenv("SENSING_MOCK_SCALE_DIV", "4")
    seen = {}

    def run(args, **kw):
        seen["env"] = kw["env"]
        return subprocess.CompletedProcess(args, 0, GOOD_LISTING, "")

    sn.inspect_rig(sn.SensingRigConfig(**RIG), run=run)
    assert "SENSING_MOCK_SCALE_DIV" not in seen["env"] and seen["env"] == sn._producer_env()


def test_a_factory_without_the_servers_sensor_description_says_so():
    from capture_server.launch import _sensing_composition

    cfg = ServerConfig(
        mock=True, cameras={"left": sn.SensingStationCamera("L")}, sensing_rig=sn.SensingRigConfig(**RIG)
    )  # no sensing_description: what a copied config would look like
    camera = p.CameraStreamConfig(1, "left", p.CameraType.SENSING_COLOR, True, W, H, 30.0, 80)
    with pytest.raises(ValueError, match="no rig or sensor description"):
        _sensing_composition(cfg, (camera,))


@pytest.mark.asyncio
async def test_a_producer_that_goes_quiet_but_stays_alive_fails_the_stream(tmp_path, monkeypatch):
    monkeypatch.setattr(sn, "_NO_FRAME_TIMEOUT_S", 0.3)
    adapter, rig = _adapter(tmp_path, camera=_camera(required=True, fps=100.0))
    producer = FileProducer(rig.path, encode_units(W, H, [RED, GREEN, BLUE]), loop=False).start()
    try:
        await adapter.enter()
        assert adapter.camera_status().status == p.DeviceState.CONNECTED
        with pytest.raises(CameraSourceError, match="no frame from the SENSING stream for"):
            for _ in range(200):  # rig.exit_code() stays None: the producer is alive, just silent
                await adapter.next_camera_jpeg()
        assert adapter.camera_status().status == p.DeviceState.DEGRADED
        assert await adapter.close() is True
    finally:
        producer.stop()


def test_reclaim_all_hands_back_a_failed_streams_file_without_decoding(tmp_path):
    path = tmp_path / "s.h264"
    path.write_bytes(os.urandom(4 * 1024 * 1024))  # real, allocated bytes the producer keeps appending to
    reader = h264.SensingStreamReader(str(path), width=W, height=H)
    reader.connect(1.0)
    before = allocated_bytes(str(path))
    reader.reclaim_all()
    assert allocated_bytes(str(path)) < before / 2 and os.stat(path).st_size == 4 * 1024 * 1024
    with pytest.raises(h264.SensingStreamError, match="failed earlier"):
        reader.read_latest(0.0)  # reclaiming never revives the stream
    with open(path, "ab") as f:
        f.write(os.urandom(2 * 1024 * 1024))
    reader.reclaim_all()  # more output arrived: reclaimed again on the next call
    assert allocated_bytes(str(path)) < 1024 * 1024
    reader.close()


def test_reclaim_all_tolerates_a_failing_native_call_only_up_to_the_spool_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(h264, "_MAX_UNRECLAIMED_BYTES", 2 * 1024 * 1024)
    path = tmp_path / "s.h264"
    path.write_bytes(os.urandom(1024 * 1024))
    reader = h264.SensingStreamReader(str(path), width=W, height=H, fallocate=lambda *a: -1)
    reader.connect(1.0)
    reader.reclaim_all()
    assert reader._punched == 0  # nothing marked reclaimed; tried again next tick
    with open(path, "ab") as f:
        f.write(os.urandom(2 * 1024 * 1024))
    with pytest.raises(h264.SensingStreamError, match="cannot be handed back"):
        reader.reclaim_all()  # 3 MiB the producer keeps growing and we cannot return: the spool is unbounded
    reader.close()


def test_reclaim_all_reclaims_a_file_created_after_the_stream_failed_before_attaching(tmp_path):
    path = tmp_path / "late.h264"
    reader = h264.SensingStreamReader(str(path), width=W, height=H)
    reader.reclaim_all()  # no file yet: nothing accumulating, nothing raised
    path.write_bytes(os.urandom(4 * 1024 * 1024))  # the producer creates it later
    reader.reclaim_all()
    assert allocated_bytes(str(path)) < 1024 * 1024
    reader.close()


def test_reclaim_all_bounds_a_late_file_it_cannot_open(tmp_path, monkeypatch):
    monkeypatch.setattr(h264, "_MAX_UNRECLAIMED_BYTES", 1024)
    path = tmp_path / "late.h264"
    path.write_bytes(os.urandom(64 * 1024))
    path.chmod(0o400)  # readable only: O_RDWR open fails, so nothing can be handed back
    if os.access(path, os.W_OK):
        pytest.skip("running as a user that can write read-only files")
    reader = h264.SensingStreamReader(str(path), width=W, height=H)
    with pytest.raises(h264.SensingStreamError, match="cannot be handed back"):
        reader.reclaim_all()
    reader.close()


@pytest.mark.asyncio
async def test_an_unreclaimable_dead_stream_stops_the_shared_producer(tmp_path):
    adapter, rig = _adapter(tmp_path, camera=_camera(required=False, fps=200.0))

    def unbounded():
        raise h264.SensingStreamError("spool is unbounded")

    adapter._reader = types.SimpleNamespace(reclaim_all=unbounded, close=lambda: None)
    adapter._dead = True
    adapter._deadline = time.monotonic()
    assert await adapter.next_camera_jpeg() is None
    assert rig.aborted == ["spool is unbounded"]
    await adapter.close()


@pytest.mark.asyncio
async def test_a_dead_stream_keeps_reclaiming_on_every_paced_poll(tmp_path):
    calls = []
    adapter, _ = _adapter(tmp_path, camera=_camera(required=False, fps=200.0))
    adapter._reader = types.SimpleNamespace(reclaim_all=lambda: calls.append(1), close=lambda: None)
    adapter._dead = True
    adapter._deadline = time.monotonic()
    for _ in range(3):
        assert await adapter.next_camera_jpeg() is None
    assert len(calls) == 3
    await adapter.close()


@pytest.mark.asyncio
async def test_stop_during_bring_up_ends_the_startup_and_the_release_is_confirmed_promptly(tmp_path):
    adapter, rig = _adapter(tmp_path, startup_timeout_s=30.0)  # the stream file never appears
    entering = asyncio.ensure_future(adapter.enter())
    await asyncio.sleep(0.3)
    started = time.monotonic()
    assert await adapter.close() is True  # without the stop signal this queues behind the 30 s startup
    assert time.monotonic() - started < 1.0
    with pytest.raises(CameraSourceError, match="stopped during startup"):
        await entering


def test_a_stopped_reader_ends_every_blocking_operation_with_the_reason(tmp_path):
    stop = threading.Event()
    path = tmp_path / "s.h264"
    path.write_bytes(b"".join(encode_units(W, H, [RED, GREEN, BLUE]) * 50))
    reader = h264.SensingStreamReader(str(path), width=W, height=H, stop=stop)
    reader.connect(1.0)
    stop.set()
    with pytest.raises(h264.SensingStreamError, match="stopped"):
        reader.read_latest(5.0)
    with pytest.raises(h264.SensingStreamError, match="stopped"):
        h264.SensingStreamReader(str(tmp_path / "never"), width=W, height=H, stop=stop).connect(5.0)
    finisher = h264.SensingStreamReader(str(path), width=W, height=H, stop=threading.Event())
    finisher.connect(1.0)
    finisher._stop.set()
    with pytest.raises(h264.SensingStreamError, match="stopped"):
        finisher.finish()

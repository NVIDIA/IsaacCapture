# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The start-of-day service: the facts it reads, the steps it reports, and the gate recording asks."""

from __future__ import annotations

import time
from types import SimpleNamespace

from capture_server import protocol as p
from egocap_web.config import CaptureStatusConfig
from egocap_web.setup_checks import CameraFact, CaptureFacts, DeviceFact
from egocap_web.setup_service import SetupService, capture_facts
from egocap_web.station_service import StationService, session_adapter
from egocap_web.stations import StationStore

BENCH = {"name": "Bench A", "camera_roles": {"ego": "cam_head", "wrist_left": "cam_left"}}
CALIBRATION = (
    "head_camera_intrinsics.json",
    "head_camera_extrinsics.json",
    "left_wrist_camera_intrinsics.json",
    "wrist_calibration_offsets.json",
)


def _healthy(**overrides) -> CaptureFacts:
    base = dict(
        running=True,
        session_id="s1",
        requested=frozenset({"headset"}),
        tracking_fresh=True,
        devices=(DeviceFact("headset", "Headset", True),),
        cameras={"cam_head": CameraFact(True, True), "cam_left": CameraFact(True, True)},
    )
    return CaptureFacts(**{**base, **overrides})


class _World:
    """The capture facts, clocks and disk a test can change between calls."""

    def __init__(self, tmp_path):
        self.facts = _healthy()
        self.now = 1_000.0
        self.free = 50 * 1024**3
        self.fact_reads = 0
        self.tmp_path = tmp_path

    def read_facts(self) -> CaptureFacts:
        self.fact_reads += 1
        return self.facts


async def _rotate() -> None:
    """Nothing to rotate in these tests."""


def _service(tmp_path) -> tuple[SetupService, StationService, _World]:
    world = _World(tmp_path)
    stations = StationService(
        StationStore(tmp_path / "stations.json"),
        session_adapter(
            offered_cameras=lambda: frozenset({"cam_head", "cam_left"}),
            change_blocked=lambda: None,
            show_label=lambda label: None,
            rotate_batch=_rotate,
            leave_station=lambda label: None,
        ),
        "Deployment",
    )
    setup = SetupService(
        facts=world.read_facts,
        session_id=lambda: world.facts.session_id,
        stations=stations,
        calibration_dir=lambda: tmp_path / "calibration",
        output_root=tmp_path / "recordings",
        free_bytes=lambda path: world.free,
        clock=lambda: world.now,
        monotonic=lambda: world.now,
    )
    return setup, stations, world


async def _with_station_in_use(tmp_path, calibrated: bool = True):
    setup, stations, world = _service(tmp_path)
    await stations.select((await stations.add(BENCH)).station_id)
    if calibrated:
        (tmp_path / "calibration").mkdir()
        for name in CALIBRATION:
            (tmp_path / "calibration" / name).write_text("{}")
    return setup, stations, world


def _steps(
    setup: SetupService, saved_count: int = 0, headset_address_changed: bool = False
) -> dict[str, tuple[str, str | None]]:
    status = setup.status(saved_count=saved_count, headset_address_changed=headset_address_changed)
    return {step["id"]: (step["status"], step["detail"]) for step in status["steps"]}


async def test_with_no_stations_the_flow_advises_and_never_blocks(tmp_path):
    setup, _, world = _service(tmp_path)
    assert setup.block_reason() is None
    status = setup.status(saved_count=0, headset_address_changed=False)
    assert status["enforced"] is False and status["blocked_reason"] is None and status["connection_test"] is None
    assert [(step["id"], step["label"], step["status"]) for step in status["steps"]] == [
        ("station", "Station", "skipped"),
        ("devices", "Devices", "todo"),
        ("calibration", "Calibration", "skipped"),
        ("record", "Record", "todo"),
        ("wrap_up", "Wrap up", "todo"),
    ]
    # The test can still be run: it then tests every camera the session offers.
    test = setup.run_connection_test()
    assert test.passed and (test.station, test.ran_at) == ("Deployment", 1_000.0)
    assert _steps(setup)["devices"] == ("done", "connection test passed")
    assert setup.block_reason() is None


async def test_a_station_in_use_cannot_record_until_the_connection_test_has_passed(tmp_path):
    setup, _, world = await _with_station_in_use(tmp_path)
    assert setup.block_reason() == "run the device connection test before recording"
    assert _steps(setup) == {
        "station": ("done", "Bench A"),
        "devices": ("todo", "run the connection test"),
        "calibration": ("done", None),
        "record": ("todo", None),
        "wrap_up": ("todo", None),
    }
    world.facts = _healthy(
        cameras={"cam_head": CameraFact(True, True), "cam_left": CameraFact(False, False, "unplugged")}
    )
    failed = setup.run_connection_test()
    assert not failed.passed
    assert setup.block_reason() == "the device connection test failed: Left wrist: cam_left: unplugged"
    assert _steps(setup)["devices"] == ("attention", "Left wrist: cam_left: unplugged")
    world.facts = _healthy()
    assert setup.run_connection_test().passed
    assert setup.block_reason() is None
    status = setup.status(saved_count=3, headset_address_changed=False)
    assert status["enforced"] is True and status["blocked_reason"] is None
    assert status["connection_test"]["passed"] is True and status["connection_test"]["station"] == "Bench A"
    assert [item["status"] for item in status["connection_test"]["items"]] == ["pass", "pass", "pass", "skip"]
    assert _steps(setup, saved_count=3)["record"] == ("done", "3 saved")


async def test_a_passed_test_does_not_carry_to_another_station_or_capture_session(tmp_path):
    setup, stations, world = await _with_station_in_use(tmp_path)
    setup.run_connection_test()
    assert setup.block_reason() is None
    world.facts = _healthy(session_id="s2")  # CaptureServer restarted: these devices were never looked at
    assert setup.current_test() is None
    assert setup.block_reason() == "run the device connection test before recording"
    setup.run_connection_test()
    cart = await stations.add({"name": "Cart B", "camera_roles": {"ego": "cam_head"}})
    await stations.select(cart.station_id)
    assert setup.block_reason() == "run the device connection test before recording"
    assert _steps(setup)["devices"] == ("todo", "run the connection test")


async def test_missing_calibration_blocks_a_take_and_shows_on_its_step(tmp_path):
    setup, _, world = await _with_station_in_use(tmp_path, calibrated=False)
    setup.run_connection_test()
    missing = "Camera calibration: missing: " + ", ".join(CALIBRATION)
    assert setup.block_reason() == missing
    assert _steps(setup)["calibration"] == ("attention", missing)
    status = setup.status(saved_count=0, headset_address_changed=False)
    assert status["blocked_reason"] == missing
    assert {item["id"]: item["status"] for item in status["system"]} == {
        "calibration": "fail",
        "gloves": "skip",
        "disk": "pass",
        "producer": "skip",
    }


async def test_low_disk_space_blocks_a_take_and_shows_on_the_devices_step(tmp_path):
    setup, _, world = await _with_station_in_use(tmp_path)
    setup.run_connection_test()
    world.free = 1024**3
    world.now += 5  # past the cache
    assert setup.block_reason() == "Free disk space: 1.0 GB free; 10 GB needed"
    assert _steps(setup)["devices"] == ("attention", "Free disk space: 1.0 GB free; 10 GB needed")


async def test_the_system_checks_are_cached_briefly_because_the_gate_is_asked_constantly(tmp_path):
    setup, _, world = await _with_station_in_use(tmp_path)
    setup.run_connection_test()
    reads = world.fact_reads
    for _ in range(50):
        assert setup.block_reason() is None
    assert world.fact_reads == reads + 1  # one read filled the cache; the other forty-nine reused it
    world.free = 1024**3
    assert setup.block_reason() is None  # still the cached answer
    world.now += 2.5
    assert setup.block_reason() is not None


async def test_the_station_step_follows_the_station_in_use(tmp_path):
    setup, stations, world = _service(tmp_path)
    bench = await stations.add({"name": "Bench A"})
    assert _steps(setup)["station"] == ("todo", "select a station before recording")
    await stations.select(bench.station_id)
    assert _steps(setup)["station"] == (
        "attention",
        "station 'Bench A': no camera is assigned to the ego (head) role",
    )
    await stations.update(bench.station_id, {"camera_roles": {"ego": "cam_head"}})
    assert _steps(setup)["station"] == ("done", "Bench A")
    # Advisory only: the gate does not judge the headset's network.
    assert _steps(setup, headset_address_changed=True)["station"] == (
        "attention",
        "this host's address changed; update the headset",
    )
    world.facts = _healthy()
    (tmp_path / "calibration").mkdir()
    for name in CALIBRATION[:2]:
        (tmp_path / "calibration" / name).write_text("{}")
    setup.run_connection_test()
    world.now += 5
    assert setup.block_reason() is None


async def test_calibration_is_skipped_for_a_mock_session(tmp_path):
    setup, _, world = await _with_station_in_use(tmp_path, calibrated=False)
    world.facts = _healthy(mock=True)
    setup.run_connection_test()
    assert setup.block_reason() is None
    assert _steps(setup)["calibration"] == ("skipped", "mock capture has nothing to calibrate")


# --- reading the capture session ---------------------------------------------------------------------------------


def _device(kind: str, description: str, state: p.DeviceState, error: str = "") -> p.DeviceStatus:
    return p.DeviceStatus(
        id=f"device-{description}",
        path=f"/{kind}",
        type=kind,
        description=description,
        provider_id="provider",
        status=state,
        reason=p.StatusReason.NONE,
        error=error,
    )


def _provider(plugin, state: p.ProviderState, error: str = "") -> p.ProviderStatus:
    kind = p.ProviderType.PLUGIN if plugin is not None else p.ProviderType.OPENXR_RUNTIME
    return p.ProviderStatus(
        id="provider",
        type=kind,
        plugin=plugin,
        name="provider",
        status=state,
        reason=p.StatusReason.NONE,
        error=error,
    )


def _livekit(name: str) -> p.LiveKitCameraStream:
    return p.LiveKitCameraStream(
        stream_id=16,
        name=name,
        url="wss://lk",
        room="r",
        publisher="o",
        track="t",
        timestamp_basis=p.LiveKitTimestampBasis.SENSOR_CAPTURE_MONOTONIC_US,
        width=8,
        height=4,
        fps=30.0,
    )


def _monitor(
    *, status=None, running=True, started=True, stopped=None, mode=p.CaptureMode.LIVE, streams=(), fresh=()
):
    """A stand-in monitor holding exactly the facts ``capture_facts`` reads."""
    now = time.monotonic()
    slot = SimpleNamespace(received_monotonic=now)
    stale = SimpleNamespace(received_monotonic=now - 60)
    return SimpleNamespace(
        native=SimpleNamespace(
            started=SimpleNamespace(session_id="s1", mode=mode, streams=streams) if started else None,
            status=SimpleNamespace(running=running) if started else None,
            stopped=stopped,
            latest_status=status,
        ),
        preview=SimpleNamespace(
            tracking=lambda: slot if "tracking" in fresh else None,
            camera=lambda name: slot if name in fresh else (stale if name == "stale_cam" else None),
        ),
        snapshot=lambda: SimpleNamespace(error_message="connection refused" if not started else None),
    )


def test_capture_facts_read_devices_cameras_and_freshness_from_the_session():
    status = p.StatusSnapshot(
        updated_at_ns=1,
        providers=(_provider(None, p.ProviderState.AVAILABLE),),
        devices=(
            _device("headset", "Headset", p.DeviceState.CONNECTED),
            _device("controller", "Left controller", p.DeviceState.DISCONNECTED, "battery empty"),
            _device("controller", "", p.DeviceState.DEGRADED),
            _device("hand", "Left hand", p.DeviceState.DISABLED),
        ),
        cameras=(
            p.CameraStatus("cam_head", 1, p.DeviceState.CONNECTED, p.StatusReason.NONE, ""),
            p.CameraStatus("stale_cam", 2, p.DeviceState.CONNECTED, p.StatusReason.NONE, ""),
            p.CameraStatus("cam_left", 3, p.DeviceState.FAILED, p.StatusReason.NONE, "unplugged"),
        ),
    )
    facts = capture_facts(_monitor(status=status, fresh=("tracking", "cam_head")), CaptureStatusConfig())
    assert (facts.running, facts.error, facts.mock, facts.session_id) == (True, None, False, "s1")
    assert facts.requested == frozenset({"headset", "controller"}) and facts.tracking_fresh
    assert facts.devices == (
        DeviceFact("headset", "Headset", True, "connected"),
        DeviceFact("controller", "Left controller", False, "battery empty"),
        DeviceFact("controller", "device-", False, "degraded"),
    )
    assert facts.cameras == {
        "cam_head": CameraFact(True, True, "connected"),
        "stale_cam": CameraFact(True, False, "connected"),
        "cam_left": CameraFact(False, False, "unplugged"),
    }
    assert (facts.livekit_cameras, facts.livekit_blocked, facts.glove_plugin_requested) == (
        frozenset(),
        None,
        False,
    )


def test_capture_facts_before_a_session_and_after_it_stopped():
    never = capture_facts(_monitor(started=False), CaptureStatusConfig())
    assert (never.running, never.error, never.session_id, never.devices, never.cameras) == (
        False,
        "connection refused",
        None,
        (),
        {},
    )
    stopped = capture_facts(_monitor(stopped=object()), CaptureStatusConfig())
    assert (stopped.running, stopped.error) == (False, "capture is not running")
    assert capture_facts(_monitor(mode=p.CaptureMode.MOCK), CaptureStatusConfig()).mock is True


def test_capture_facts_report_the_glove_plugin_and_livekit_cameras():
    config = CaptureStatusConfig(plugins=(p.PluginType.MANUS,), capture_server_url="ws://127.0.0.1:8082")

    def glove_error(*providers) -> str | None:
        status = p.StatusSnapshot(updated_at_ns=1, providers=providers)
        return capture_facts(_monitor(status=status), config).glove_plugin_error

    assert glove_error(_provider(p.PluginType.MANUS, p.ProviderState.AVAILABLE)) is None
    assert glove_error(_provider(p.PluginType.MANUS, p.ProviderState.FAILED, "licence dongle not found")) == (
        "licence dongle not found"
    )
    assert glove_error(_provider(p.PluginType.MANUS, p.ProviderState.INITIALIZING)) == "plugin is initializing"
    assert glove_error(_provider(None, p.ProviderState.AVAILABLE)) == "not reported by the capture session"
    assert capture_facts(_monitor(), config).glove_plugin_error == "not reported by the capture session"
    facts = capture_facts(_monitor(streams=(_livekit("lk_head"),)), config)
    assert "hand" in facts.requested and facts.glove_plugin_requested
    assert (facts.livekit_cameras, facts.livekit_blocked) == (frozenset({"lk_head"}), None)
    remote = CaptureStatusConfig(capture_server_url="ws://192.168.1.50:8082")
    assert capture_facts(_monitor(streams=(_livekit("lk_head"),)), remote).livekit_blocked is not None

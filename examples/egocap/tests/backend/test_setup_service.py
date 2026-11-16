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
        tracking_fresh=True,
        devices=(DeviceFact("Headset", True, headset=True),),
        cameras={"cam_head": CameraFact(True, True), "cam_left": CameraFact(True, True)},
    )
    return CaptureFacts(**{**base, **overrides})


class _World:
    """The capture facts, clock and disk a test can change between calls."""

    def __init__(self):
        self.facts: CaptureFacts | Exception = _healthy()
        self.now = 1_000.0
        self.free = 50 * 1024**3
        self.fact_reads = 0

    def read_facts(self) -> CaptureFacts:
        self.fact_reads += 1
        if isinstance(self.facts, Exception):
            raise self.facts
        return self.facts


async def _rotate() -> None:
    """Nothing to rotate in these tests."""


def _service(tmp_path) -> tuple[SetupService, StationService, _World]:
    world = _World()
    stations = StationService(
        StationStore(tmp_path / "stations.json"),
        session_adapter(
            offered_cameras=lambda: frozenset({"cam_head", "cam_left"}),
            change_blocked=lambda: None,
            show_label=lambda label: None,
            rotate_batch=_rotate,
            renamed=lambda previous, name: None,
        ),
        "Deployment",
    )
    setup = SetupService(
        facts=world.read_facts,
        stations=stations,
        calibration_dir=lambda: tmp_path / "calibration",
        output_root=tmp_path / "recordings",
        free_bytes=lambda path: world.free,
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


def _steps(setup: SetupService, **overrides) -> dict[str, tuple[str, str | None]]:
    arguments = {
        "took_today": False,
        "saved_count": 0,
        "report_saved": False,
        "headset_address_changed": False,
        **overrides,
    }
    return {step["id"]: (step["status"], step["detail"]) for step in setup.status(**arguments)["steps"]}


async def test_with_no_stations_the_checklist_advises_and_never_blocks(tmp_path):
    setup, _, world = _service(tmp_path)
    world.facts = _healthy(cameras={"cam_head": CameraFact(False, False, "unplugged")})
    assert setup.block_reason() is None
    status = setup.status(took_today=False, saved_count=0, report_saved=False, headset_address_changed=False)
    assert status["enforced"] is False and status["blocked_reason"] is None
    assert [(step["id"], step["label"], step["status"]) for step in status["steps"]] == [
        ("station", "Station", "skipped"),
        ("devices", "Devices", "attention"),
        ("record", "Record", "todo"),
        ("wrap_up", "Wrap up", "todo"),
    ]
    assert _steps(setup)["devices"] == ("attention", "cam_head: unplugged")
    assert [(item["id"], item["status"]) for item in status["checks"]] == [
        ("device:0", "pass"),
        ("camera:cam_head", "fail"),
        ("calibration", "skip"),
        ("disk", "pass"),
    ]


async def test_a_station_in_use_is_held_while_any_item_fails_and_released_when_it_recovers(tmp_path):
    setup, _, world = await _with_station_in_use(tmp_path)
    assert setup.block_reason() is None  # everything is connected: nothing to run, nothing to wait for
    assert _steps(setup) == {
        "station": ("done", "Bench A"),
        "devices": ("done", "all checks pass"),
        "record": ("todo", None),
        "wrap_up": ("todo", None),
    }
    world.facts = _healthy(
        cameras={"cam_head": CameraFact(True, True), "cam_left": CameraFact(False, False, "unplugged")}
    )
    world.now += 2  # past the cache
    assert setup.block_reason() == "Left wrist: cam_left: unplugged"
    status = setup.status(took_today=False, saved_count=0, report_saved=False, headset_address_changed=False)
    assert status["enforced"] is True and status["blocked_reason"] == "Left wrist: cam_left: unplugged"
    assert _steps(setup)["devices"] == ("attention", "Left wrist: cam_left: unplugged")
    world.facts = _healthy()
    world.now += 2
    assert setup.block_reason() is None


async def test_missing_calibration_and_low_disk_space_hold_a_take(tmp_path):
    setup, _, world = await _with_station_in_use(tmp_path, calibrated=False)
    missing = "Camera calibration: missing: " + ", ".join(CALIBRATION)
    assert setup.block_reason() == missing and _steps(setup)["devices"] == ("attention", missing)
    (tmp_path / "calibration").mkdir()
    for name in CALIBRATION:
        (tmp_path / "calibration" / name).write_text("{}")
    world.free = 1024**3
    world.now += 2
    assert setup.block_reason() == "Free disk space: 1.0 GB free; 10 GB needed"


async def test_the_checklist_is_cached_briefly_and_follows_an_edit_of_the_station_at_once(tmp_path):
    setup, stations, world = await _with_station_in_use(tmp_path)
    reads = world.fact_reads
    for _ in range(50):
        assert setup.block_reason() is None
    assert world.fact_reads == reads + 1  # one evaluation served all fifty
    world.free = 1024**3
    assert setup.block_reason() is None  # still the cached answer
    # The station was given a camera the session does not offer: what is judged changed, so it is judged again.
    await stations.update(stations.registry.selected_id, {"camera_roles": {"wrist_right": "cam_gone"}})
    assert (
        setup.block_reason()
        == "station 'Bench A': camera 'cam_gone' (right wrist) is not offered by the capture session"
    )
    await stations.update(
        stations.registry.selected_id, {"camera_roles": {"wrist_right": None, "wrist_left": None}}
    )
    assert setup.block_reason() == "Free disk space: 1.0 GB free; 10 GB needed"


async def test_a_checklist_that_cannot_be_evaluated_holds_the_take_and_never_raises(tmp_path, caplog):
    setup, _, world = await _with_station_in_use(tmp_path)
    world.facts = RuntimeError("status snapshot is malformed")
    for _ in range(3):
        world.now += 2
        assert setup.block_reason() == "Station checks: could not be evaluated; see backend logs"
    assert caplog.text.count("could not be evaluated") == 1  # logged once, not on every evaluation
    assert setup.headset_connected() is False
    world.facts = _healthy()
    world.now += 2
    assert setup.block_reason() is None


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


async def test_the_record_and_wrap_up_steps_follow_the_day(tmp_path):
    setup, _, _ = await _with_station_in_use(tmp_path)
    assert _steps(setup)["record"] == ("todo", None)
    assert _steps(setup, took_today=True, saved_count=3)["record"] == (
        "done",
        "recorded today · 3 saved in this batch",
    )
    # A new batch restarts its count; the day still has its takes.
    assert _steps(setup, took_today=True)["record"] == ("done", "recorded today")
    # A batch left open overnight has a count, but the new day has no take yet.
    assert _steps(setup, saved_count=3)["record"] == ("todo", None)
    assert _steps(setup, report_saved=True)["wrap_up"] == ("done", "report saved")


async def test_whether_a_headset_is_connected_comes_from_the_same_evaluation(tmp_path):
    setup, _, world = await _with_station_in_use(tmp_path)
    assert setup.headset_connected() is True
    world.facts = _healthy(devices=(DeviceFact("Headset", False, "disconnected", headset=True),))
    world.now += 2
    assert setup.headset_connected() is False


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
        devices=(
            _device("headset", "Headset", p.DeviceState.CONNECTED),
            _device("controller", "Left controller", p.DeviceState.DISCONNECTED, "battery empty"),
            # A glove is whatever the capture session calls it; the checklist never depends on that label.
            _device("hand_tracker", "", p.DeviceState.DEGRADED),
            _device("hand_tracker", "Right glove", p.DeviceState.DISABLED),
        ),
        cameras=(
            p.CameraStatus("cam_head", 1, p.DeviceState.CONNECTED, p.StatusReason.NONE, ""),
            p.CameraStatus("stale_cam", 2, p.DeviceState.CONNECTED, p.StatusReason.NONE, ""),
            p.CameraStatus("cam_left", 3, p.DeviceState.FAILED, p.StatusReason.NONE, "unplugged"),
        ),
    )
    facts = capture_facts(_monitor(status=status, fresh=("tracking", "cam_head")), CaptureStatusConfig())
    assert (facts.running, facts.error, facts.mock, facts.tracking_fresh) == (True, None, False, True)
    assert facts.devices == (
        DeviceFact("Headset", True, "connected", headset=True),
        DeviceFact("Left controller", False, "battery empty"),
        DeviceFact("device-", False, "degraded"),
    )
    assert facts.headset_connected is True
    assert facts.cameras == {
        "cam_head": CameraFact(True, True, "connected"),
        "stale_cam": CameraFact(True, False, "connected"),
        "cam_left": CameraFact(False, False, "unplugged"),
    }
    assert (facts.livekit_cameras, facts.livekit_blocked) == (frozenset(), None)


def test_capture_facts_before_a_session_and_after_it_stopped():
    never = capture_facts(_monitor(started=False), CaptureStatusConfig())
    assert (never.running, never.error, never.devices, never.cameras) == (False, "connection refused", (), {})
    stopped = capture_facts(_monitor(stopped=object()), CaptureStatusConfig())
    assert (stopped.running, stopped.error) == (False, "capture is not running")
    assert capture_facts(_monitor(mode=p.CaptureMode.MOCK), CaptureStatusConfig()).mock is True


def test_capture_facts_report_livekit_cameras_and_why_they_cannot_be_recorded():
    local = CaptureStatusConfig(capture_server_url="ws://127.0.0.1:8082")
    facts = capture_facts(_monitor(streams=(_livekit("lk_head"),)), local)
    assert (facts.livekit_cameras, facts.livekit_blocked) == (frozenset({"lk_head"}), None)
    remote = CaptureStatusConfig(capture_server_url="ws://192.168.1.50:8082")
    assert capture_facts(_monitor(streams=(_livekit("lk_head"),)), remote).livekit_blocked is not None

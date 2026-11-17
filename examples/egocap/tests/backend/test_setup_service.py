# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The start-of-day service: the facts it reads, the steps it reports, and the gate recording asks."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

from capture_server import protocol as p
from egocap_web.config import CaptureStatusConfig
from egocap_web.setup_checks import CameraFact, CaptureFacts, DeviceFact
from egocap_web.setup_service import CHECKS_PENDING, SetupService, capture_facts
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
    """The capture facts, clock, disk and deployment setting a test can change between evaluations."""

    def __init__(self):
        self.facts: CaptureFacts | Exception = _healthy()
        self.now = 1_000.0
        self.free = 50 * 1024**3
        self.fact_reads = 0
        self.require_calibration = False
        self.free_reads_on: list[int] = []
        #: Set to make the disk read wait, as a stalled mount would.
        self.disk_gate: threading.Event | None = None
        self.disk_entered = threading.Event()

    def read_facts(self) -> CaptureFacts:
        self.fact_reads += 1
        if isinstance(self.facts, Exception):
            raise self.facts
        return self.facts

    def free_bytes(self, path) -> int:
        self.free_reads_on.append(threading.get_ident())
        self.disk_entered.set()
        if self.disk_gate is not None:
            self.disk_gate.wait(5)
        return self.free


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
        require_calibration=lambda: world.require_calibration,
        free_bytes=world.free_bytes,
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
    await setup.refresh()
    return setup, stations, world


def _status(setup: SetupService, **overrides) -> dict:
    arguments = {
        "took_today": False,
        "saved_count": 0,
        "report_saved": False,
        "headset_address_changed": False,
        **overrides,
    }
    return setup.status(**arguments)


def _steps(setup: SetupService, **overrides) -> dict[str, tuple[str, str | None]]:
    return {step["id"]: (step["status"], step["detail"]) for step in _status(setup, **overrides)["steps"]}


async def test_with_no_stations_the_checklist_advises_and_never_blocks(tmp_path):
    setup, _, world = _service(tmp_path)
    world.facts = _healthy(cameras={"cam_head": CameraFact(False, False, "unplugged")})
    assert setup.block_reason() is None  # not even before the first evaluation
    await setup.refresh()
    assert setup.block_reason() is None
    status = _status(setup)
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
    await setup.refresh()
    assert setup.block_reason() == "Left wrist: cam_left: unplugged"
    status = _status(setup)
    assert status["enforced"] is True and status["blocked_reason"] == "Left wrist: cam_left: unplugged"
    assert _steps(setup)["devices"] == ("attention", "Left wrist: cam_left: unplugged")
    world.facts = _healthy()
    await setup.refresh()
    assert setup.block_reason() is None


async def test_missing_calibration_is_advisory_unless_the_deployment_requires_it(tmp_path):
    setup, _, world = await _with_station_in_use(tmp_path, calibrated=False)
    missing = "Camera calibration: missing: " + ", ".join(CALIBRATION)
    # Off by default: the item warns, the step is done, and nothing holds the take.
    assert setup.block_reason() is None
    status = _status(setup)
    assert status["blocked_reason"] is None
    assert _steps(setup)["devices"] == ("done", f"{missing}; takes are not held")
    calibration = next(item for item in status["checks"] if item["id"] == "calibration")
    assert (calibration["label"], calibration["status"]) == ("Camera calibration", "warn")
    assert calibration["detail"] == "missing: " + ", ".join(CALIBRATION)
    # Required: the same item fails and holds recording.
    world.require_calibration = True
    await setup.refresh()
    assert setup.block_reason() == missing and _steps(setup)["devices"] == ("attention", missing)
    calibration = next(item for item in _status(setup)["checks"] if item["id"] == "calibration")
    assert (calibration["label"], calibration["status"]) == ("Camera calibration", "fail")
    (tmp_path / "calibration").mkdir()
    for name in CALIBRATION:
        (tmp_path / "calibration" / name).write_text("{}")
    await setup.refresh()
    assert setup.block_reason() is None and _steps(setup)["devices"] == ("done", "all checks pass")


async def test_low_disk_space_holds_a_take(tmp_path):
    setup, _, world = await _with_station_in_use(tmp_path)
    world.free = 1024**3
    await setup.refresh()
    assert setup.block_reason() == "Free disk space: 1.0 GB free; 10 GB needed"


async def test_a_device_that_is_not_connected_does_not_hold_a_take(tmp_path):
    setup, _, world = await _with_station_in_use(tmp_path)
    world.facts = _healthy(
        devices=(DeviceFact("Headset", True, headset=True), DeviceFact("Glove", False, "degraded"))
    )
    await setup.refresh()
    assert setup.block_reason() is None
    assert _steps(setup)["devices"] == ("done", "Glove: degraded; takes are not held")


async def test_the_gate_answers_from_memory_and_the_files_are_read_in_a_worker_thread(tmp_path):
    setup, _, world = await _with_station_in_use(tmp_path)
    reads, disk_reads = world.fact_reads, len(world.free_reads_on)
    world.free = 1024**3
    for _ in range(50):
        assert setup.block_reason() is None
        _status(setup)
        setup.headset_connected()
    # No reader evaluated anything: the answer stays what the last evaluation found.
    assert (world.fact_reads, len(world.free_reads_on)) == (reads, disk_reads)
    await setup.refresh()
    assert setup.block_reason() == "Free disk space: 1.0 GB free; 10 GB needed"
    assert world.free_reads_on and threading.get_ident() not in world.free_reads_on


async def test_a_stalled_disk_never_stalls_the_event_loop_or_the_gate(tmp_path):
    setup, _, world = await _with_station_in_use(tmp_path)
    world.disk_gate, world.free = threading.Event(), 1024**3
    world.disk_entered.clear()
    evaluation = asyncio.create_task(setup.refresh())
    # The loop keeps running while the disk read hangs, and the gate still answers at once.
    while not world.disk_entered.is_set():
        await asyncio.sleep(0.01)
    assert not evaluation.done() and setup.block_reason() is None
    # Callers that arrive meanwhile wait for that one evaluation instead of each taking a thread.
    waiting = [asyncio.create_task(setup.refresh()) for _ in range(3)]
    await asyncio.sleep(0.05)
    reads = len(world.free_reads_on)
    world.disk_gate.set()
    results = await asyncio.gather(evaluation, *waiting)
    assert len(world.free_reads_on) == reads and all(result is results[0] for result in results)
    assert setup.block_reason() == "Free disk space: 1.0 GB free; 10 GB needed"


async def test_a_result_is_as_old_as_its_facts_however_long_the_disk_took(tmp_path):
    setup, _, world = await _with_station_in_use(tmp_path)
    world.disk_gate = threading.Event()
    world.disk_entered.clear()
    evaluation = asyncio.create_task(setup.refresh())
    while not world.disk_entered.is_set():
        await asyncio.sleep(0.01)
    # The facts were sampled before the stall, so a result that arrives this late is already too old to admit.
    world.now += 11
    world.disk_gate.set()
    late = await evaluation
    assert late.items and setup.block_reason() == CHECKS_PENDING


async def test_a_take_is_held_until_the_station_in_use_has_been_evaluated(tmp_path):
    setup, stations, world = _service(tmp_path)
    bench = await stations.add(BENCH)
    await stations.select(bench.station_id)
    # Before the first evaluation nothing is known, so nothing is admitted and nothing is claimed.
    assert setup.block_reason() == CHECKS_PENDING
    status = _status(setup)
    assert status["blocked_reason"] == CHECKS_PENDING and status["checks"] == []
    assert _steps(setup)["devices"] == ("todo", CHECKS_PENDING)
    assert setup.headset_connected() is False
    await setup.refresh()
    assert setup.block_reason() is None
    # A change of what is judged makes the last result another station's: held until it is judged again.
    await stations.update(bench.station_id, {"camera_roles": {"wrist_left": None}})
    assert setup.block_reason() == CHECKS_PENDING
    await setup.refresh()
    assert setup.block_reason() is None
    # A result that stopped being renewed (an evaluation stuck on a stalled disk) no longer admits a take.
    world.now += 11
    assert setup.block_reason() == CHECKS_PENDING


async def test_a_caller_that_waited_out_an_evaluation_of_another_station_judges_again(tmp_path):
    setup, stations, world = await _with_station_in_use(tmp_path)
    world.disk_gate = threading.Event()
    world.disk_entered.clear()
    first = asyncio.create_task(setup.refresh())
    while not world.disk_entered.is_set():
        await asyncio.sleep(0.01)
    await stations.update(stations.registry.selected_id, {"camera_roles": {"wrist_left": None}})
    second = asyncio.create_task(setup.refresh())
    await asyncio.sleep(0.05)
    world.disk_gate.set()
    stale, fresh = await asyncio.gather(first, second)
    assert stale is not fresh and dict(fresh.key)["wrist_left"] is None
    assert setup.block_reason() is None


async def test_a_station_problem_outranks_the_checklist(tmp_path):
    setup, stations, _ = await _with_station_in_use(tmp_path)
    await stations.update(stations.registry.selected_id, {"camera_roles": {"wrist_right": "cam_gone"}})
    assert (
        setup.block_reason()
        == "station 'Bench A': camera 'cam_gone' (right wrist) is not offered by the capture session"
    )


async def test_a_checklist_that_cannot_be_evaluated_holds_the_take_and_never_raises(tmp_path, caplog):
    setup, _, world = await _with_station_in_use(tmp_path)
    world.facts = RuntimeError("status snapshot is malformed")
    for _ in range(3):
        await setup.refresh()
        assert setup.block_reason() == "Station checks: could not be evaluated; see backend logs"
    assert caplog.text.count("could not be evaluated") == 1  # logged once, not on every evaluation
    assert setup.headset_connected() is False
    world.facts = _healthy()
    await setup.refresh()
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
    await setup.refresh()
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
    await setup.refresh()
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

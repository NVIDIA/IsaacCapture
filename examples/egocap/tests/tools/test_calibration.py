# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import zarr
from scipy.spatial.transform import Rotation

# tools/calibration/ is a plain scripts directory, not a package; the root makes ``tests`` importable when
# this directory is run on its own.
_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_ROOT / "tools" / "calibration"), str(_ROOT)]
import calib_common as common  # noqa: E402
import calibrate_aim_to_coil as aim_to_coil  # noqa: E402
import calibrate_coil_to_wrist as coil_to_wrist  # noqa: E402
import record  # noqa: E402
from tests.backend.test_recording_adapter import default_tracking_values  # noqa: E402

from capture_server import protocol  # noqa: E402
from capture_server.client import CaptureFrame  # noqa: E402

RATE_HZ = 60.0
TRUTH = common.transform(Rotation.from_rotvec([0.4, -1.1, 0.7]).as_matrix(), [0.03, -0.07, 0.05])


def _write_source(episode: Path, source: str, fields: dict[str, np.ndarray]) -> None:
    group = zarr.open_group(str(episode / source), mode="w")
    for name, values in fields.items():
        group.create_array(name, data=values)


def _poses(rotations: Rotation, positions: np.ndarray) -> np.ndarray:
    poses = np.zeros((*positions.shape[:-1], 4, 4))
    poses[..., :3, :3] = rotations.as_matrix().reshape(*positions.shape[:-1], 3, 3)
    poses[..., :3, 3] = positions
    poses[..., 3, 3] = 1.0
    return poses


def _wave(times: np.ndarray, amplitude: float, seed: int, shape: tuple[int, ...] = ()) -> np.ndarray:
    """Smooth, reproducible motion: (len(times), *shape, 3) sums of two slow sinusoids."""
    rng = np.random.default_rng(seed)
    freq = rng.uniform(0.15, 0.5, (2, *shape, 3))
    phase = rng.uniform(0, 2 * np.pi, (2, *shape, 3))
    return (
        amplitude * np.sin(2 * np.pi * freq * times.reshape(-1, *[1] * (len(shape) + 2)) + phase).sum(axis=1) / 2
    )


def _coil_to_wrist_episode(tmp_path: Path) -> Path:
    times = np.arange(0, 40.0, 1 / RATE_HZ)
    rng = np.random.default_rng(0)
    offsets = rng.uniform(-0.01, 0.01, (5, 3))
    # Wrist-frame joints: static everywhere except the five moving fingertips.
    joints = np.tile(rng.uniform(-0.05, 0.05, (25, 3)), (len(times), 1, 1))
    joints[:, coil_to_wrist.TIP] += _wave(times, 0.06, 1, (5,))
    sensor_rotations = Rotation.from_rotvec(_wave(times, 1.6, 2, (5,)).reshape(-1, 3))
    tips_in_coil = joints[:, coil_to_wrist.TIP] @ TRUTH[:3, :3].T + TRUTH[:3, 3]
    sensor_positions = tips_in_coil - np.einsum(
        "nkij,kj->nki", sensor_rotations.as_matrix().reshape(-1, 5, 3, 3), offsets
    )
    episode = tmp_path / "session" / "episode_0"
    _write_source(
        episode,
        "manus_glove",
        {
            "capture_time": 1000.0 + times,
            "left_hand_pose": _poses(Rotation.identity(len(times) * 25), joints),
            "left_sensor_pose": _poses(sensor_rotations, sensor_positions),
        },
    )
    return episode


def _aim_to_coil_episode(tmp_path: Path, delay_s: float = 0.05, turn_rad: float = 2.2) -> Path:
    times = np.arange(0, 40.0, 1 / RATE_HZ)
    clamp = np.array([0.2, 1.1, -0.4])

    def aim(at: np.ndarray) -> tuple[Rotation, np.ndarray]:
        return Rotation.from_rotvec(_wave(at, turn_rad, 3)), clamp + _wave(at, 0.15, 4)

    # The sensor row stamped t holds what the coil measured when the controller was where it was at t - delay.
    rotation, position = aim(times - delay_s)
    in_aim = np.einsum("nji,nj->ni", rotation.as_matrix(), clamp - position)
    in_coil = (in_aim - TRUTH[:3, 3]) @ TRUTH[:3, :3]
    sensor_positions = np.zeros((len(times), 5, 3))
    sensor_positions[:, 2] = in_coil
    episode = tmp_path / "session" / "episode_0"
    _write_source(episode, "pico_controller", {"capture_time": 1000.0 + times, "right_pose": _poses(*aim(times))})
    _write_source(
        episode,
        "manus_glove",
        {
            "capture_time": 1000.0 + times,
            "right_sensor_pose": _poses(Rotation.identity(len(times) * 5), sensor_positions),
        },
    )
    return episode


def _run(tool, episode: Path, *args: str) -> tuple[int, dict]:
    """The exit status and the results file's content after solving ``episode``."""
    results = episode.parents[1] / "results.json"
    status = tool.main([str(episode), *args, "--results", str(results)])
    return status, json.loads(results.read_text()) if results.is_file() else {}


def test_coil_to_wrist_recovers_the_transform_and_saves_it_in_the_station_file(tmp_path, monkeypatch):
    # The station's file may hold more than these tools write; that is kept.
    monkeypatch.setenv("EGOCAP_CALIBRATION_DIR", str(tmp_path / "calibration"))
    station = tmp_path / "calibration" / "wrist_calibration_offsets.json"
    station.parent.mkdir()
    station.write_text('{"other": 1}')
    assert coil_to_wrist.main([str(_coil_to_wrist_episode(tmp_path)), "--side", "left"]) == 0
    results = json.loads(station.read_text())
    assert results == {"other": 1, "left": {"T_coil_wrist": results["left"]["T_coil_wrist"]}}
    assert np.allclose(np.array(results["left"]["T_coil_wrist"]), TRUTH, atol=1e-4)


def test_a_failed_take_says_what_to_change_and_keeps_the_earlier_result(tmp_path, capsys):
    episode = _coil_to_wrist_episode(tmp_path)
    _, earlier = _run(coil_to_wrist, episode, "--side", "left")
    group = zarr.open_group(str(episode / "manus_glove"), mode="r+")
    hand, sensor = group["left_hand_pose"][:], group["left_sensor_pose"][:]
    hand[:, coil_to_wrist.TIP[0]] = hand[0, coil_to_wrist.TIP[0]]
    sensor[:, 0] = sensor[0, 0]
    group["left_hand_pose"][:], group["left_sensor_pose"][:] = hand, sensor
    status, results = _run(coil_to_wrist, episode, "--side", "left")
    assert status == 1 and results == earlier
    printed = capsys.readouterr().out
    assert "[FAIL] least-turned sensor: thumb" in printed and "-> Bend the thumb more." in printed
    assert "[FAIL] least-moved fingertip: thumb" in printed and "FAIL. Nothing was saved; record again." in printed


def test_aim_to_coil_recovers_the_transform(tmp_path):
    status, results = _run(aim_to_coil, _aim_to_coil_episode(tmp_path), "--side", "right", "--finger", "middle")
    solved = np.array(results["right"]["T_controller_manus"])
    assert status == 0 and list(results["right"]) == ["T_controller_manus"]
    assert common.rotation_between_deg(TRUTH, solved) < 0.2 and np.allclose(solved[:3, 3], TRUTH[:3, 3], atol=5e-4)


def test_aim_to_coil_fails_a_take_whose_controller_barely_turned(tmp_path, capsys):
    status, results = _run(
        aim_to_coil, _aim_to_coil_episode(tmp_path, turn_rad=0.3), "--side", "right", "--finger", "middle"
    )
    assert status == 1 and results == {}
    assert "[FAIL] how far the controller turned" in capsys.readouterr().out


class _FakeCaptureClient:
    """Stands in for CaptureClient: delivers the given tracking values at 60 Hz until stopped."""

    values: dict = {}

    def __init__(self, url: str) -> None:
        self._task: asyncio.Task | None = None

    async def start(self, *, sources, plugins, on_capture_frame, on_stopped) -> None:
        description = protocol.tracking_stream_descriptor(tuple(sources), RATE_HZ)

        async def run() -> None:
            sequence = 0
            while True:
                stamp = time.monotonic_ns()
                payload = protocol.encode_tracking_frame(self.values)
                on_capture_frame(CaptureFrame("s", description, 0, sequence, stamp, stamp, payload))
                sequence += 1
                await asyncio.sleep(1 / RATE_HZ)

        self._task = asyncio.ensure_future(run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()


def _record(tmp_path, monkeypatch, values, record_s=0.5, controllers=True):
    monkeypatch.setattr(record, "CaptureClient", _FakeCaptureClient)
    monkeypatch.setattr(_FakeCaptureClient, "values", values)
    # Enter is "pressed" at once to start and after ``record_s`` to stop.
    delays = iter((0.0, record_s))

    async def enter(prompt):
        await asyncio.sleep(next(delays))

    monkeypatch.setattr(record, "_enter", enter)
    return asyncio.run(record.record_take("ws://x", tmp_path / "takes", "task", "left", controllers, "PROCEDURE"))


def test_a_recorded_take_is_an_episode_the_solvers_read(tmp_path, monkeypatch, capsys):
    episode = _record(tmp_path, monkeypatch, default_tracking_values())
    assert episode.parent.parent == tmp_path / "takes" and "PROCEDURE" in capsys.readouterr().out
    glove = common.read_source(episode, "manus_glove", ("left_hand_pose", "left_sensor_pose"), "")
    controller = common.read_source(episode, "pico_controller", ("left_pose",), "")
    assert 15 < len(glove["capture_time"]) == len(controller["capture_time"])
    assert not (episode / "manus_glove" / "right_hand_pose").exists()
    assert (glove["left_sensor_pose"][:, :, 3, 3] == 1).all()


def test_recording_waits_for_the_devices_and_says_which_are_missing(tmp_path, monkeypatch, capsys):
    values = default_tracking_values() | {"manus_left.sensor_present": 0, "controller_left.aim_is_valid": 0}
    monkeypatch.setattr(record, "LIVE_S", 0.05)
    waited = record._wait_until_live

    async def give_up(feed, needed, side):
        await asyncio.wait_for(waited(feed, needed, side), timeout=0.5)

    monkeypatch.setattr(record, "_wait_until_live", give_up)
    with pytest.raises(TimeoutError):
        _record(tmp_path, monkeypatch, values)
    printed = capsys.readouterr().out
    assert "Waiting for the left controller. Turn it on" in printed
    assert "Waiting for the left glove's fingertip sensors." in printed
    assert "Waiting for the left glove. Turn it on" not in printed
    assert not list((tmp_path / "takes").glob("*/episode_*"))


def test_a_take_needs_only_the_calibrated_side(tmp_path, monkeypatch):
    values = default_tracking_values() | {"controller_right.aim_is_valid": 0, "head.is_valid": 0}
    values["hand_right.joint_valid"] = np.zeros(26, dtype=np.uint8)
    episode = _record(tmp_path, monkeypatch, values)
    assert len(common.read_source(episode, "pico_controller", ("left_pose",), "")["capture_time"]) > 15
    assert len(common.read_source(episode, "manus_glove", ("left_sensor_pose",), "")["capture_time"]) > 15

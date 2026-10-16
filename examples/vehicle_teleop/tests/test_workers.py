# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pty
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from isaaccapture.retargeters import (
    VehicleControlRetargeter,
    VehicleControlRetargeterConfig,
)

from vehicle_teleop.isaac_keyboard_control_worker import (
    IsaacKeyboardControlState,
    IsaacKeyboardControlWorker,
    TerminalKeyReader,
)
from vehicle_teleop.isaac_remote_steering_worker import (
    IsaacRemoteSteeringWorker,
    build_parser,
)
from vehicle_teleop.vehicle_command import VehicleControlCommand


@pytest.fixture
def wheel_worker(monkeypatch):
    monkeypatch.setattr(time, "monotonic_ns", lambda: 1_000_000_000)
    sample = SimpleNamespace(
        steering=0.0,
        throttle=0.0,
        brake=0.0,
        connected=True,
        sample_time_monotonic_ns=1_000_000_000,
    )
    worker = IsaacRemoteSteeringWorker.__new__(IsaacRemoteSteeringWorker)
    worker._tracker = Mock()
    worker._tracker.get_wheel_data.return_value = sample
    worker._deviceio_session = Mock()
    worker._sample_timeout_ns = 500_000_000
    worker._collection_id = "steering_wheel"
    worker._sequence = 0
    worker._logger = None
    worker._verbose = False
    worker._publish = Mock()
    worker._retargeter = VehicleControlRetargeter(
        VehicleControlRetargeterConfig(steer_scale=-1, throttle_scale=2),
        steering_neutral=0.2,
    )
    return worker, sample


@pytest.mark.parametrize(
    "invalid", ["disconnected", "stale", "future", "missing_time", "missing_data"]
)
def test_invalid_wheel_input_publishes_neutral_before_raising(wheel_worker, invalid):
    worker, sample = wheel_worker
    if invalid == "disconnected":
        sample.connected = False
    elif invalid == "missing_data":
        worker._tracker.get_wheel_data.return_value = None
    else:
        sample.sample_time_monotonic_ns = {
            "stale": 499_999_999,
            "future": 1_000_000_001,
            "missing_time": 0,
        }[invalid]
    with pytest.raises(RuntimeError):
        worker._publish_next_command()
    command = worker._publish.call_args.args[0]
    assert command == VehicleControlCommand.neutral(timestamp_ns=command.timestamp_ns)
    worker._publish.assert_called_once()


def test_unchanged_axes_allowed_until_sample_expires(wheel_worker, monkeypatch):
    worker, sample = wheel_worker
    worker._publish_next_command()
    monkeypatch.setattr(time, "monotonic_ns", lambda: 1_500_000_000)
    worker._publish_next_command()
    monkeypatch.setattr(time, "monotonic_ns", lambda: 1_500_000_001)
    with pytest.raises(RuntimeError):
        worker._publish_next_command()
    assert [call.args[0].accel for call in worker._publish.call_args_list] == [
        0.5,
        0.5,
        0.0,
    ]
    sample.sample_time_monotonic_ns = 1_500_000_001
    assert worker._read_isaac_sample() is sample


@pytest.mark.parametrize("keys", ["wq", "wr", "wc", "w\x1b"])
def test_keyboard_bursts_do_not_stay_in_text_buffer(keys):
    master, slave = pty.openpty()
    try:
        with os.fdopen(slave) as stream, TerminalKeyReader(stream) as reader:
            os.write(master, keys.encode())
            assert reader.read_key() == keys[0]
            assert reader.read_key() == keys[1]
            assert reader.read_key() is None
    finally:
        os.close(master)


def test_quit_does_not_publish_another_active_command():
    worker = IsaacKeyboardControlWorker.__new__(IsaacKeyboardControlWorker)
    worker._running = True
    worker._state = IsaacKeyboardControlState(gas_brake=1.0)
    worker._publish_next_command = Mock()
    worker._run_loop(Mock(read_key=Mock(return_value="q")), 0.02)
    worker._publish_next_command.assert_not_called()


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1"])
def test_sample_timeout_must_be_finite_positive(value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--sample-timeout-s", value])

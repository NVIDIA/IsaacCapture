# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise example entry points with simulated input and native session boundaries."""

import importlib.util
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from repo_paths import repo_root


EXAMPLES = [
    ("gripper_retargeting_example_simple", None),
    ("se3_retargeting_example", "1"),
    ("se3_retargeting_example", "2"),
    ("se3_retargeting_example", "3"),
    ("se3_retargeting_example", "4"),
]


@pytest.fixture(
    params=EXAMPLES,
    ids=["gripper", "hand-abs", "controller-abs", "hand-rel", "controller-rel"],
)
def example(request, monkeypatch, tmp_path):
    name, choice = request.param
    state = SimpleNamespace(
        steps=0,
        now=0.0,
        wall_time=1000.0,
        events=[],
        stop_after=25,
        step_error=None,
        cleanup_error=None,
        launcher_cleanup_error=None,
        adjust_wallclock=False,
    )

    class Session:
        def __init__(self, config):
            self.frame_count = 0

        def __enter__(self):
            state.events.append("session-enter")
            return self

        def __exit__(self, *args):
            state.events.append("session-exit")
            if state.cleanup_error:
                raise state.cleanup_error

        def step(self):
            state.steps += 1
            self.frame_count += 1
            if state.steps == state.stop_after:
                raise state.step_error or KeyboardInterrupt()
            return {
                "gripper_command": [1.0],
                "ee_pose": [np.array([0, 0, 0, 0, 0, 0, 1])],
                "ee_delta": [np.zeros(6)],
            }

        def get_elapsed_time(self):
            return state.now

    class Launcher:
        @staticmethod
        def add_launcher_arguments(parser):
            pass

        @staticmethod
        def launch_context(args):
            return Launcher()

        def __enter__(self):
            state.events.append("launcher-enter")
            return self

        def __exit__(self, *args):
            state.events.append("launcher-exit")
            if state.launcher_cleanup_error:
                raise state.launcher_cleanup_error

    def module(name, **attributes):
        result = ModuleType(name)
        result.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, result)
        return result

    root = module("isaaccapture")
    root.__path__ = []
    root.deviceio = module(
        "isaaccapture.deviceio", HandTracker=MagicMock(), ControllerTracker=MagicMock()
    )
    module("isaaccapture.cloudxr", CloudXRLauncher=Launcher)
    module(
        "isaaccapture.retargeters",
        GripperRetargeter=MagicMock(),
        GripperRetargeterConfig=SimpleNamespace,
        Se3AbsRetargeter=MagicMock(),
        Se3RelRetargeter=MagicMock(),
        Se3RetargeterConfig=SimpleNamespace,
    )
    module(
        "isaaccapture.teleop_session_manager",
        TeleopSession=Session,
        TeleopSessionConfig=SimpleNamespace,
        PluginConfig=SimpleNamespace,
        create_standard_inputs=lambda trackers: {
            "hands": MagicMock(),
            "controllers": MagicMock(),
        },
    )
    module("isaaccapture.retargeting_engine").__path__ = []
    module(
        "isaaccapture.retargeting_engine.deviceio_source_nodes",
        HandsSource=MagicMock(),
        ControllersSource=MagicMock(),
    )
    directory = repo_root() / "examples" / "teleop" / "python"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    monkeypatch.setattr(
        loaded, "PLUGIN_ROOT_DIR", tmp_path / "no-plugins", raising=False
    )
    monkeypatch.setattr("builtins.input", lambda prompt: choice)

    def sleep(_):
        state.now += 1.0
        # Wall clock corrections must not change a requested duration.
        state.wall_time += -100.0 if state.adjust_wallclock else 1.0

    clock = SimpleNamespace(
        monotonic=lambda: state.now, time=lambda: state.wall_time, sleep=sleep
    )
    if hasattr(loaded, "time"):
        monkeypatch.setattr(loaded, "time", clock)
    if "_example_loop" in sys.modules:
        monkeypatch.setattr(sys.modules["_example_loop"], "time", clock)

    def run(*args):
        monkeypatch.setattr(sys, "argv", [f"{name}.py", *args])
        return loaded.main()

    return state, run


def test_default_runs_until_interrupted_and_closes_in_order(example, capsys):
    state, run = example
    assert run() == 0
    assert state.steps == 25  # Continue beyond the old 20-second cutoff.
    assert state.events == [
        "launcher-enter",
        "session-enter",
        "session-exit",
        "launcher-exit",
    ]
    output = capsys.readouterr().out
    assert "Ctrl+C" in output
    assert "Stop requested" in output
    assert "Session closed." in output


def test_duration_uses_monotonic_time_and_reports_completion(example, capsys):
    state, run = example
    state.adjust_wallclock = True
    assert run("--duration", "3") == 0
    assert state.steps == 3
    assert state.events[-2:] == ["session-exit", "launcher-exit"]
    output = capsys.readouterr().out
    assert "Duration reached" in output
    assert "Session closed." in output


def test_zero_duration_runs_until_interrupted(example):
    state, run = example
    assert run("--duration", "0") == 0
    assert state.steps == state.stop_after


def test_stream_failure_is_not_reported_as_success(example, capsys):
    state, run = example
    state.stop_after = 2
    state.step_error = RuntimeError("stream connection lost")
    with pytest.raises(RuntimeError, match="stream connection lost"):
        run()
    assert state.events[-2:] == ["session-exit", "launcher-exit"]
    assert "Session closed." not in capsys.readouterr().out


@pytest.mark.parametrize("boundary", ["session", "launcher"])
def test_cleanup_failure_is_not_reported_as_success(example, capsys, boundary):
    state, run = example
    error = RuntimeError("teardown failed")
    if boundary == "session":
        state.cleanup_error = error
    else:
        state.launcher_cleanup_error = error
    with pytest.raises(RuntimeError, match="teardown failed"):
        run("--duration", "1")
    assert state.events[-2:] == ["session-exit", "launcher-exit"]
    assert "Session closed." not in capsys.readouterr().out


@pytest.mark.parametrize("duration", ["-1", "nan", "inf", "-inf"])
def test_invalid_duration_fails_before_starting_runtime(example, duration):
    state, run = example
    with pytest.raises(SystemExit) as error:
        run(f"--duration={duration}")
    assert error.value.code == 2
    assert state.events == []

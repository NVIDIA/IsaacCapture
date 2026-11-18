# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent program's options of the experiment's own launcher.

--batch-milestones, --record-debug, --debug-max-gib, --no-timing and
--motion-approval, and the program's own --debug-map-hz for the debug
recording. The experiment is the folder of the rig's earlier, single
machine program; its launcher is thor_runtime.py. Stand-ins take the place of
the four modules the options load (thor_timing, thor_model_trace,
thor_recording, thor_milestones), and the runs put the launcher on APAI's mock
scene as test_launcher.py does. The last test runs the experiment's real
modules, when its folder is there. Before it: three faults of the full stack
that the first runs on the real arm showed.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import signal
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
import test_launcher
from isaaccapture_examples.agentic_remote.agent import config as agent_config
from isaaccapture_examples.agentic_remote.agent import checkahead, launcher, task_gate
from isaaccapture_examples.agentic_remote.agent.arm_adapter import (
    AGENT_TERMINAL,
    RobotLinkArm,
)
from isaaccapture_examples.agentic_remote.agent.config import RunOptions
from isaaccapture_examples.agentic_remote.agent.link_client import AgentLink
from test_launcher import QUICK, Stage, settings_file

from conftest import EXPERIMENT_DIR_ENV, import_experiment

apai = pytest.mark.apai
experiment = pytest.mark.experiment
# The fixtures of test_launcher.py, for the tests of this file.
make_stage = test_launcher.make_stage
stage = test_launcher.stage

GIB = 1024**3
# Every option on: what the owner's command line of the experiment asked for.
ALL = RunOptions(
    timing=True,
    model_trace=True,
    record_debug=True,
    debug_max_gib=2.0,
    batch_milestones=True,
)
# The spans (timed stretches) the launcher adds, in the order it adds them:
# (class, method, span, category, the options the timing's wrap() gets). On
# the robot link arm a move and its parts, on the check ahead what comes
# before a MOVE and each of its passes, on the task gate the taking and the
# giving back of the arm, on the link's stereo camera a stereo pair.
LINK_SPANS = [
    ("RobotLinkArm", "stream_to", "motor.stream_to", "Execution", {}),
    (
        "RobotLinkArm",
        "close_gripper_two_stage",
        "gripper.close_two_stage",
        "Execution",
        {},
    ),
    ("RobotLinkArm", "_fresh_state", "motor.fresh_joint_readings", "Execution", {}),
    ("RobotLinkArm", "_await_first", "motor.move_to_first_result", "Execution", {}),
    ("RobotLinkArm", "_await_armed", "motor.move_to_armed", "Execution", {}),
    ("RobotLinkArm", "_permit_loop", "motor.permits", "Execution", {}),
    # A span without length per point of the permit loop, named by the point.
    ("RobotLinkArm", "_mark", "motor.mark", "Execution", {"named_argument": 1}),
    ("CheckAhead", "wait_fresh_grid", "safety.wait_fresh_grid", "Safety", {}),
    ("CheckAhead", "vet_whole", "safety.check_whole_move", "Safety", {}),
    # Counted in bulk inside the span of its move's permits.
    ("MoveCheck", "step", "safety.check_ahead_pass", "Safety", {"hot": True}),
    ("TaskGate", "_start", "task.take_control", "Task", {}),
    ("TaskGate", "_end", "task.give_back_control", "Task", {}),
    ("LinkStereoCamera", "_grab", "camera.pair_wait_read_depth", "Look", {}),
]
# What the program says at its start, one line per option.
SAID = ("timing: ", "model transcript: ", "debug recording: ", "milestone batching: ")
# What an option leaves in the run folder.
LEFT = ("run-options.json", "timing.jsonl", "model-trace", "debug")


class Recorder:
    """In place of the experiment's timing recorder: what it wrapped and timed."""

    def __init__(self) -> None:
        self.hooks: list[str] = []
        # (span name, category, the options wrap() got)
        self.wrapped: list[tuple[str, str, dict[str, Any]]] = []
        self.calls: list[str] = []  # the span name of each timed call

    def wrap(
        self, function: Callable[..., Any], name: str, category: str, **options: Any
    ) -> Callable[..., Any]:
        self.wrapped.append((name, category, options))

        @functools.wraps(function)
        def measured(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(name)
            return function(*args, **kwargs)

        return measured


class Evidence:
    """In place of the experiment's debug recorder: the events it was given."""

    def __init__(self) -> None:
        self.hooks: list[str] = []
        self.events: list[tuple[str, dict[str, Any]]] = []  # (kind, data)

    def record(self, kind: str, builder: Callable[[], dict[str, Any]]) -> int:
        self.events.append((kind, builder()))
        return len(self.events)


def state_fields(state: Any) -> dict[str, Any]:
    """In place of the experiment's state_fields: the pose and its time."""
    return {"q": [float(value) for value in state.q], "t": float(state.t)}


class Experiment:
    """Stand-ins for the four modules of the experiment behind the options.

    note(text) gets "begin:<session>" and "end:<session>" for the three
    sessions and "batch" for the milestone batching, in the order they happen.
    Each stand-in takes the arguments the experiment's own function takes.
    """

    def __init__(self, note: Callable[[str], None]) -> None:
        self.note = note
        self.recorder = Recorder()
        # What the timing session yields; None is the experiment's own answer
        # when its recorder cannot be made.
        self.timing: Any = self.recorder
        self.began: dict[str, dict[str, Any]] = {}  # session -> its arguments
        self.fails = ""  # the session that cannot begin
        self.fails_at_end = ""  # the session that cannot end
        self.confirmations = 0  # questions asked through the timed question
        self.tools: list[str] = []  # tools run through the timed gate
        # Per agent that got the batching: whether its run_task was the task
        # gate's wrapper already.
        self.batched: list[bool] = []
        # What the recording session yields: None, or an Evidence.
        self.evidence: Any = None

    def module(self, name: str) -> SimpleNamespace:
        """In place of launcher.experiment_module."""
        return {
            "thor_timing": SimpleNamespace(timing_session=self.timing_session),
            "thor_model_trace": SimpleNamespace(
                model_trace_session=self.model_trace_session
            ),
            "thor_recording": SimpleNamespace(
                recording_session=self.recording_session, state_fields=state_fields
            ),
            "thor_milestones": SimpleNamespace(
                install_batched_milestones=self.install_batched_milestones
            ),
        }[name]

    @contextmanager
    def _session(self, name: str, **arguments: Any) -> Iterator[None]:
        if self.fails == name:
            raise FileExistsError(f"the {name} session cannot begin")
        assert name not in self.began, f"the {name} session began twice"
        self.note(f"begin:{name}")
        self.began[name] = arguments
        try:
            yield
        finally:
            self.note(f"end:{name}")
            if self.fails_at_end == name:
                raise RuntimeError(f"the {name} session cannot end")

    @contextmanager
    def timing_session(self, directory: Any, runtime_module: Any = None) -> Iterator:
        """Times runtime_module.Admission.execute and .operator_answer."""
        with self._session("timing", directory=directory, module=runtime_module):
            gate, ask = runtime_module.Admission, runtime_module.operator_answer
            execute = gate.execute

            def timed_execute(self_: Any, original: Any, name: str, args: Any) -> Any:
                self.tools.append(name)
                return execute(self_, original, name, args)

            def timed_ask(prompt: str) -> str:
                self.confirmations += 1
                return ask(prompt)

            with (
                mock.patch.object(gate, "execute", timed_execute),
                mock.patch.object(runtime_module, "operator_answer", timed_ask),
            ):
                yield self.timing

    @contextmanager
    def model_trace_session(
        self, directory: Any, timing_recorder: Any = None
    ) -> Iterator[None]:
        with self._session("model_trace", directory=directory, timing=timing_recorder):
            yield

    @contextmanager
    def recording_session(self, run_dir: Any, *, max_bytes: int) -> Iterator[Any]:
        with self._session("recording", directory=run_dir, max_bytes=max_bytes):
            yield self.evidence

    def install_batched_milestones(self, agent: Any) -> None:
        self.note("batch")
        self.batched.append(hasattr(agent.run_task, "__wrapped__"))


@pytest.fixture
def stand_ins(monkeypatch: pytest.MonkeyPatch) -> Callable[..., Experiment]:
    """stand_ins(note, options=ALL): the launcher loads an Experiment's modules.

    The run takes `options` whatever its file's stack is: the mock scene runs
    on the minimal stack, which the arguments themselves are refused with.
    """

    def make(note: Callable[[str], None], options: RunOptions = ALL) -> Experiment:
        made = Experiment(note)
        monkeypatch.setattr(launcher, "experiment_module", made.module)
        monkeypatch.setattr(launcher, "_run_options", lambda args, settings: options)
        return made

    return make


def noting_attach(stage: Stage) -> Callable[..., AgentLink]:
    """stage.attach, which notes "attach" in stage.order."""

    def attach(socket_path: str | None = None, *, timeout_s: float) -> AgentLink:
        stage.order.append("attach")
        return stage.attach(socket_path, timeout_s=timeout_s)

    return attach


def never_attached(*args: Any, **kwargs: Any) -> AgentLink:
    raise AssertionError("attached to the robot link service")


def parsed(settings: Any, *arguments: str) -> RunOptions:
    """The options of a run with these arguments on these settings."""
    args = launcher._parser().parse_args(["--config", "unused", *arguments])
    return launcher._run_options(args, settings)


def map_hz(settings: Any, *arguments: str) -> float:
    """The map pictures a second the debug recording of such a run stores."""
    args = launcher._parser().parse_args(["--config", "unused", *arguments])
    return launcher._debug_map_hz(args, settings)


def at_limit(settings: Any, *arguments: str) -> str:
    """What the debug recording of such a run does at its size."""
    args = launcher._parser().parse_args(["--config", "unused", *arguments])
    return launcher._debug_at_limit(args, settings)


# What a run takes, from its arguments and its file


def test_the_full_stack_times_and_transcribes_by_default_and_the_minimal_does_not(
    tmp_path: Path,
) -> None:
    full = agent_config.load_settings(
        settings_file(tmp_path, experiment={"stack": "full"})
    )
    # As the experiment's launcher: timing and model transcript on, the rest off.
    assert parsed(full) == RunOptions(timing=True, model_trace=True)
    assert parsed(full).debug_max_gib == 16.0
    assert parsed(full, "--no-timing") == RunOptions(model_trace=True)
    assert parsed(full, "--batch-milestones") == RunOptions(
        timing=True, model_trace=True, batch_milestones=True
    )
    assert parsed(full, "--record-debug", "--debug-max-gib", "2.5") == RunOptions(
        timing=True, model_trace=True, record_debug=True, debug_max_gib=2.5
    )
    # Without --record-debug the size does nothing, as in the experiment's launcher.
    assert parsed(full, "--debug-max-gib", "4").record_debug is False
    # The debug recording stores 2 pictures of the obstacle map a second
    # unless the run names another rate; 0 stores every one. The rate is the
    # program's own: it is no field of what the experiment's launcher takes.
    assert map_hz(full) == map_hz(full, "--record-debug") == 2.0
    assert map_hz(full, "--record-debug", "--debug-map-hz", "0") == 0.0
    assert map_hz(full, "--debug-map-hz", "0.5") == 0.5
    assert parsed(full, "--record-debug", "--debug-map-hz", "0") == parsed(
        full, "--record-debug"
    )
    # At its size the debug recording drops its oldest map pictures unless
    # the run asks for the experiment's stop. Also the program's own.
    assert at_limit(full) == at_limit(full, "--record-debug") == "drop-oldest"
    assert at_limit(full, "--debug-at-limit", "drop-oldest") == "drop-oldest"
    assert at_limit(full, "--record-debug", "--debug-at-limit", "stop") == "stop"
    assert parsed(full, "--record-debug", "--debug-at-limit", "stop") == parsed(
        full, "--record-debug"
    )
    with pytest.raises(SystemExit) as refused:
        at_limit(full, "--debug-at-limit", "drop-newest")
    assert refused.value.code == 2
    recorded = parsed(full, "--record-debug")
    assert launcher._options_row(recorded, 0.5) == {
        **dataclasses.asdict(recorded),
        "debug_map_hz": 0.5,
        "debug_at_limit": "drop-oldest",
    }
    assert launcher._options_row(recorded, 0.0, "stop") == {
        **dataclasses.asdict(recorded),
        "debug_map_hz": 0.0,
        "debug_at_limit": "stop",
    }
    plain = RunOptions(timing=True, model_trace=True)
    assert launcher._options_row(plain, 0.5) == dataclasses.asdict(plain)
    assert launcher._options_row(plain, 0.5, "stop") == dataclasses.asdict(plain)
    # The file switches either recording off.
    for key in ("timing", "model_trace"):
        off = agent_config.load_settings(
            settings_file(tmp_path, experiment={"stack": "full", key: False})
        )
        assert dataclasses.asdict(parsed(off))[key] is False
        assert parsed(off) != parsed(full)
    # The minimal stack takes nothing, and the approval the program has is "task".
    minimal = agent_config.load_settings(settings_file(tmp_path))
    assert parsed(minimal) == parsed(minimal, "--motion-approval", "task")
    assert parsed(minimal) == RunOptions() and map_hz(minimal) == 2.0
    assert at_limit(minimal) == "drop-oldest"
    assert parsed(full, "--motion-approval", "task") == parsed(full)


def test_the_command_line_of_the_experiments_launcher_is_taken(tmp_path: Path) -> None:
    full = agent_config.load_settings(
        settings_file(tmp_path, experiment={"stack": "full"})
    )
    old = (
        "--operate --interactive --record-debug --depth-backend ffs "
        "--motion-approval task --batch-milestones"
    ).split()
    # --operate and --depth-backend ffs change nothing of what a run records.
    assert parsed(full, *old) == parsed(
        full, "--interactive", "--record-debug", "--batch-milestones"
    )
    # Three of its arguments go to APAI's configuration: the stereo matcher by
    # any name (the experiment's own list is asked when it is built), its two
    # speed profiles and a mounting record.
    parser = launcher._parser()
    args = parser.parse_args(["--config", "x", *old])
    assert launcher._config_arguments(args) == {
        "speed_profile": None,
        "depth_backend": "ffs",
        "mounting": None,
    }
    more = ["--speed-profile", "conservative", "--mounting", "/m/latest.json"]
    args = parser.parse_args(["--config", "x", "--depth-backend", "sgm", *more])
    assert launcher._config_arguments(args) == {
        "speed_profile": "conservative",
        "depth_backend": "sgm",
        "mounting": "/m/latest.json",
    }
    with pytest.raises(SystemExit) as refused:
        parsed(full, "--speed-profile", "fast")
    assert refused.value.code == 2
    # The speed profile is the full stack's: the minimal one has its own key.
    minimal = agent_config.load_settings(settings_file(tmp_path))
    with pytest.raises(agent_config.ConfigError, match="--speed-profile needs"):
        agent_config.joint_speed(minimal, "conservative")
    assert agent_config.joint_speed(minimal) == minimal.task.max_joint_speed


def test_the_shipped_files_name_the_two_recordings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import yaml

    for name in agent_config.PATH_VARIABLES:
        monkeypatch.setenv(name, "/absolute")
    configs = test_launcher.EXAMPLE / "configs"
    for name in ("agent.yaml", "agent_rebot.yaml"):
        written = yaml.safe_load((configs / name).read_text())["experiment"]
        assert written["timing"] is True and written["model_trace"] is True, name
        settings = agent_config.load_settings(configs / name)
        assert parsed(settings) == RunOptions(timing=True, model_trace=True), name
    # The mock file is of the minimal stack: it names neither and takes none.
    mock_file = yaml.safe_load((configs / "agent_mock.yaml").read_text())
    assert not {"timing", "model_trace"} & set(mock_file["experiment"])
    assert parsed(agent_config.load_settings(configs / "agent_mock.yaml")) == (
        RunOptions()
    )


@pytest.mark.parametrize(
    ("arguments", "block", "message"),
    [
        (
            ["--batch-milestones"],
            {},
            "--batch-milestones needs experiment.stack full (it is minimal)",
        ),
        (["--record-debug"], {}, "--record-debug needs experiment.stack full"),
        (["--debug-max-gib", "4"], {}, "--debug-max-gib needs experiment.stack full"),
        (["--no-timing"], {}, "--no-timing needs experiment.stack full"),
        (
            ["--record-debug", "--batch-milestones", "--no-timing"],
            {},
            "--batch-milestones, --no-timing, --record-debug need experiment.stack",
        ),
        (
            ["--record-debug", "--no-timing"],
            {"stack": "full"},
            "--record-debug needs the timing; omit --no-timing",
        ),
        (
            ["--record-debug"],
            {"stack": "full", "timing": False},
            "--record-debug needs the timing; set experiment.timing to true",
        ),
        (
            ["--record-debug", "--debug-max-gib", "0"],
            {"stack": "full"},
            "--debug-max-gib must be a finite number above 0, not 0.0",
        ),
        (
            ["--debug-max-gib", "-3"],
            {"stack": "full"},
            "--debug-max-gib must be a finite number above 0",
        ),
        (
            ["--debug-max-gib", "nan"],
            {"stack": "full"},
            "--debug-max-gib must be a finite number above 0",
        ),
        (
            ["--debug-max-gib", "inf"],
            {"stack": "full"},
            "--debug-max-gib must be a finite number above 0",
        ),
        (
            ["--debug-map-hz", "1"],
            {},
            "--debug-map-hz needs experiment.stack full (it is minimal)",
        ),
        (
            ["--record-debug", "--debug-map-hz", "-1"],
            {"stack": "full"},
            "--debug-map-hz must be a finite number, 0 or above, not -1.0",
        ),
        (
            ["--debug-map-hz", "nan"],
            {"stack": "full"},
            "--debug-map-hz must be a finite number, 0 or above",
        ),
        (
            ["--debug-map-hz", "inf"],
            {"stack": "full"},
            "--debug-map-hz must be a finite number, 0 or above",
        ),
        (
            ["--debug-at-limit", "stop"],
            {},
            "--debug-at-limit needs experiment.stack full (it is minimal)",
        ),
        (
            ["--debug-at-limit", "drop-oldest"],
            {},
            "--debug-at-limit needs experiment.stack full (it is minimal)",
        ),
        (["--motion-approval", "tool"], {}, "--motion-approval tool is not built"),
        (
            ["--speed-profile", "conservative"],
            {},
            "--speed-profile needs experiment.stack full (it is minimal)",
        ),
        (
            ["--motion-approval", "tool"],
            {"stack": "full"},
            "one typed yes per task (--motion-approval task)",
        ),
    ],
)
def test_an_option_the_run_cannot_take_is_refused_with_one_line_and_exit_code_2(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    block: dict[str, Any],
    message: str,
) -> None:
    path = settings_file(tmp_path, experiment={"stack": "minimal", **block})
    code = launcher.main(
        ["--config", str(path), "--task", QUICK, *arguments], attach=never_attached
    )
    assert code == launcher.EXIT_NOT_RUN
    error = capsys.readouterr().err
    assert message in error
    assert error.startswith("agent program: ") and error.count("\n") == 1
    assert not (tmp_path / "state").exists()  # nothing was made or started


# A run without an option


@apai
def test_without_an_option_a_run_loads_nothing_of_the_experiment_and_leaves_no_file(
    stage: Stage, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def never(name: str) -> Any:
        raise AssertionError(f"the experiment's {name} was loaded")

    monkeypatch.setattr(launcher, "experiment_module", never)
    execute, stream_to = task_gate.TaskGate.execute, RobotLinkArm.stream_to
    assert stage.run("--task", QUICK) == launcher.EXIT_OK
    # As before the options were there: one typed yes, the arm given back;
    # then the question at the end of a session that used the arm.
    assert stage.order == ["agent:completed", "nobody:completed", "close:nobody"]
    assert len(stage.asked) == 2 and stage.asked[1] == task_gate.END_QUESTION
    for name in LEFT:
        assert not (stage.run_dir / name).exists(), name
    assert not [line for line in stage.lines if line.startswith(SAID)]
    assert task_gate.TaskGate.execute is execute
    assert RobotLinkArm.stream_to is stream_to
    # --check names what a run would take: here nothing.
    capsys.readouterr()
    assert stage.run("--check") == launcher.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["run_options"] == dataclasses.asdict(RunOptions())
    # With the debug recording it also names the map pictures a second and
    # what the recording does at its size.
    recorded = RunOptions(timing=True, record_debug=True, debug_max_gib=2.0)
    monkeypatch.setattr(launcher, "_run_options", lambda args, settings: recorded)
    assert stage.run("--check") == launcher.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["run_options"] == {
        **dataclasses.asdict(recorded),
        "debug_map_hz": 2.0,
        "debug_at_limit": "drop-oldest",
    }


# Every option, on the mock scene, with stand-ins for the experiment's modules


@apai
def test_each_option_installs_its_hook_once_in_the_order_of_the_experiments_launcher(
    stage: Stage, stand_ins: Callable[..., Experiment]
) -> None:
    made = stand_ins(stage.order.append)
    execute, stream_to = task_gate.TaskGate.execute, RobotLinkArm.stream_to
    stage.typed = [QUICK]
    code = stage.run("--interactive", attach=noting_attach(stage))
    assert code == launcher.EXIT_OK
    # Timing, model transcript, debug recording: begun before anything is
    # attached and ended, in reverse, after the link is closed.
    assert stage.order == [
        "begin:timing",
        "begin:model_trace",
        "begin:recording",
        "attach",
        "batch",
        "agent:completed",
        "nobody:completed",
        "close:nobody",
        "end:recording",
        "end:model_trace",
        "end:timing",
    ]
    # What each got: the run folder; the model transcript the timing's recorder;
    # the debug recording its size in bytes.
    for name in ("timing", "model_trace", "recording"):
        assert made.began[name]["directory"] == stage.run_dir, name
    assert made.began["model_trace"]["timing"] is made.recorder
    assert made.began["recording"]["max_bytes"] == 2 * GIB
    # The timing took the task gate for the experiment's tool admission and the
    # program's question for its confirmation: both ran through it.
    assert made.began["timing"]["module"].Admission is task_gate.TaskGate
    # The task's typed yes and the question at the session's end.
    assert made.confirmations == 2 == len(stage.asked)
    assert stage.asked[1] == task_gate.END_QUESTION
    assert made.tools.count("open_gripper") == 1
    # The batching was put on the one agent before the task gate wrapped its
    # run_task, as the experiment's launcher does.
    assert made.batched == [False]
    # The launcher's own spans, and the wait for a task line.
    assert made.recorder.wrapped == [
        *[(name, category, options) for _, _, name, category, options in LINK_SPANS],
        ("operator.terminal_input", "Operator", {}),
    ]
    # What ran through them: the one typed task, whose jaw command took the
    # arm and gave it back, then the end of the input. No move was made.
    assert made.recorder.calls == [
        "operator.terminal_input",
        "task.take_control",
        "task.give_back_control",
        "operator.terminal_input",
    ]
    assert [hook.rsplit(".", 2)[1:] for hook in made.recorder.hooks] == [
        [owner, method] for owner, method, _, _, _ in LINK_SPANS
    ]
    # Nothing stays wrapped after the run.
    assert task_gate.TaskGate.execute is execute
    assert RobotLinkArm.stream_to is stream_to
    assert not hasattr(RobotLinkArm._permit_loop, "__wrapped__")
    assert not hasattr(checkahead.MoveCheck.step, "__wrapped__")
    assert not hasattr(task_gate.TaskGate._start, "__wrapped__")
    # Said at the start, before the robot connection: what is recorded where,
    # and what the workstation cannot record.
    up = stage.lines.index(stage.said("robot connection up")[0])
    for start in SAID:
        (line,) = (line for line in stage.lines if line.startswith(start))
        assert stage.lines.index(line) < up, start
    (timing,) = stage.said("timing: ")
    assert str(stage.run_dir / "timing.jsonl") in timing
    assert "Not timed here: the motor reads and sends" in timing
    (recording,) = stage.said("debug recording: ")
    assert f"{stage.run_dir / 'debug'}, at most 2 GiB" in recording
    # What happens at that size, and how long it holds at the rate of the
    # map pictures.
    assert (
        "at most 2 GiB: when that is reached the oldest map pictures are dropped "
        "and the viewer keeps following. That size holds about 4 min at 2 map "
        "pictures a second, which --debug-map-hz sets; with --debug-at-limit "
        "stop the recording ends there instead; ./run_viewer.sh"
    ) in recording
    assert "./run_viewer.sh shows it in a browser" in recording
    assert "Not recorded here: the motor commands and the single motor reads" in (
        recording
    )
    assert "robot.feedback, is what this program read from ROBOT_STATE" in recording
    written = json.loads((stage.run_dir / "run-options.json").read_text())
    assert written == {
        **dataclasses.asdict(ALL),
        "debug_map_hz": 2.0,
        "debug_at_limit": "drop-oldest",
    }


@apai
def test_a_stop_and_the_close_of_the_link_come_before_any_recording_ends(
    stage: Stage, stand_ins: Callable[..., Experiment]
) -> None:
    stand_ins(stage.order.append)
    stage.on_ask = lambda: signal.raise_signal(signal.SIGINT)
    before = stage.h.app.snapshot().stop_number
    code = stage.run("--task", QUICK, attach=noting_attach(stage))
    assert stage.order == [
        "begin:timing",
        "begin:model_trace",
        "begin:recording",
        "attach",
        "batch",
        f"stop:{AGENT_TERMINAL}",
        "close:nobody",
        "end:recording",
        "end:model_trace",
        "end:timing",
    ]
    assert stage.h.app.snapshot().stop_number == before + 1
    assert stage.motion == [] and code == launcher.EXIT_OK


@apai
def test_a_recording_that_cannot_begin_ends_the_program_before_it_attaches(
    stage: Stage,
    stand_ins: Callable[..., Experiment],
    capsys: pytest.CaptureFixture[str],
) -> None:
    made = stand_ins(stage.order.append)
    made.fails = "recording"
    execute = task_gate.TaskGate.execute
    code = stage.run("--task", QUICK, attach=never_attached)
    assert code == launcher.EXIT_NOT_RUN
    # The two that had begun ended; nothing was attached, asked or sent.
    assert stage.order == [
        "begin:timing",
        "begin:model_trace",
        "end:model_trace",
        "end:timing",
    ]
    assert stage.asked == [] and stage.h.heard("owner") == []
    error = capsys.readouterr().err
    assert "the experiment's recordings cannot start (FileExistsError: " in error
    assert "nothing was attached" in error
    assert task_gate.TaskGate.execute is execute


@apai
def test_a_recording_that_cannot_end_keeps_the_runs_exit_code(
    stage: Stage,
    stand_ins: Callable[..., Experiment],
    capsys: pytest.CaptureFixture[str],
) -> None:
    made = stand_ins(stage.order.append)
    made.fails_at_end = "model_trace"
    assert stage.run("--task", QUICK) == launcher.EXIT_OK
    assert stage.order[-4:] == [
        "close:nobody",
        "end:recording",
        "end:model_trace",
        "end:timing",
    ]
    error = capsys.readouterr().err
    assert "a recording of this run did not end as it should: RuntimeError" in error


@apai
def test_without_the_timing_nothing_is_timed_and_the_question_is_the_programs_own(
    stage: Stage, stand_ins: Callable[..., Experiment]
) -> None:
    made = stand_ins(stage.order.append, RunOptions(model_trace=True))
    assert stage.run("--task", QUICK) == launcher.EXIT_OK
    assert stage.order == [
        "begin:model_trace",
        "agent:completed",
        "nobody:completed",
        "close:nobody",
        "end:model_trace",
    ]
    # As the experiment's launcher without its timing: no recorder for the
    # model transcript, no span, and no batching that was not asked for.
    assert made.began["model_trace"]["timing"] is None
    assert made.recorder.wrapped == [] and made.batched == []
    # Both questions, the task's and the session's end, are the program's own.
    assert made.confirmations == 0 and len(stage.asked) == 2
    assert [line for line in stage.lines if line.startswith(SAID)] == stage.said(
        "model transcript: "
    )


def test_a_timing_without_a_recorder_adds_no_span_and_keeps_the_question() -> None:
    noted: list[str] = []
    made = Experiment(noted.append)
    made.timing = None  # the experiment's timing could not make its recorder

    def ask(prompt: str) -> str:
        return "yes"

    options = RunOptions(timing=True, model_trace=True)
    sessions = launcher.experiment_sessions(options, Path("run"), ask)
    with mock.patch.object(launcher, "experiment_module", made.module):
        with sessions as (asking, timing, recording):
            assert timing is None and asking("Type yes: ") == "yes"
            assert recording is None  # no debug recording was asked for
            assert noted == ["begin:timing", "begin:model_trace"]
    assert noted[2:] == ["end:model_trace", "end:timing"]
    assert made.began["model_trace"]["timing"] is None
    assert made.recorder.wrapped == []


# What the experiment's viewer needs of a run. The viewer (thor_viewer.py) is a
# web page for a browser on a run's debug recording. It lists a run only with
# APAI's configuration in the run folder, and it draws the measured joints
# from the recording's robot.feedback events.


@apai
def test_a_run_writes_apais_configuration_before_it_attaches(stage: Stage) -> None:
    import yaml

    file = stage.run_dir / agent_config.RESOLVED_CONFIG
    there: list[bool] = []

    def attach(socket_path: str | None = None, *, timeout_s: float) -> AgentLink:
        there.append(file.is_file())
        return stage.attach(socket_path, timeout_s=timeout_s)

    assert stage.run("--task", QUICK, attach=attach) == launcher.EXIT_OK
    # Written at the start: a run that goes on can be listed already.
    assert file.name == "resolved-config.yaml" and there == [True]
    written = yaml.safe_load(file.read_text(encoding="utf-8"))
    # APAI's configuration as the run built it, with what the viewer reads.
    assert written["arm"]["type"] == "robot_link"
    assert written["arm"]["model"] and written["arm"]["ee_frame"]
    assert written["safety"]["max_joint_vel"] == test_launcher.SPEED
    assert "table_z" in written["safety"]


@apai
def test_a_configuration_that_cannot_be_written_costs_the_run_nothing(
    stage: Stage,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    # A value YAML cannot hold is a ConfigError, and no file is left.
    odd = SimpleNamespace(as_dict=lambda: {"arm": object()})
    with pytest.raises(agent_config.ConfigError, match="cannot be written as YAML"):
        agent_config.write_resolved_config(tmp_path, odd)
    assert not (tmp_path / agent_config.RESOLVED_CONFIG).exists()

    def full(run_dir: Path, cfg: Any) -> None:
        raise OSError("no space left on device")

    monkeypatch.setattr(agent_config, "write_resolved_config", full)
    with caplog.at_level(logging.WARNING):
        assert stage.run("--task", QUICK) == launcher.EXIT_OK
    assert stage.order == ["agent:completed", "nobody:completed", "close:nobody"]
    assert not (stage.run_dir / agent_config.RESOLVED_CONFIG).exists()
    assert (
        "resolved-config.yaml was not written, so the experiment's viewer will "
        "not list this run: no space left on device"
    ) in caplog.text


@apai
def test_a_run_with_the_debug_recording_records_the_poses_of_robot_state(
    stage: Stage, stand_ins: Callable[..., Experiment]
) -> None:
    made = stand_ins(stage.order.append)
    made.evidence = Evidence()
    get_state = RobotLinkArm.get_state
    began = time.monotonic()
    assert stage.run("--task", QUICK) == launcher.EXIT_OK
    ended = time.monotonic()
    events = made.evidence.events
    # The measured joints, robot.feedback, and what else each ROBOT_STATE
    # reports of the motors, robot.state (test_agent_measures.py).
    assert {kind for kind, _ in events} == {"robot.feedback", launcher.STATE_EVENT}
    poses = [data for kind, data in events if kind == "robot.feedback"]
    for data in poses:
        # The pose the robot app published: the arm did not move in this task.
        assert data["state"]["q"] == pytest.approx([0.0, 0.1, 0.1, 0.0, 0.0, 0.0])
        assert data["read_failures"] == 0
        assert data["source"] == launcher.FEEDBACK_SOURCE
        assert data["clock"] == launcher.FEEDBACK_CLOCK
        # Its time is one of this machine's monotonic clock, during the run:
        # the ROBOT_STATE's receipt less the age of the joint readings.
        assert began - 1.0 < data["state"]["t"] < ended
    assert "receipt on the workstation's monotonic clock" in launcher.FEEDBACK_CLOCK
    assert "no time of the motors" in launcher.FEEDBACK_CLOCK
    # A steady feed from the ROBOT_STATEs themselves, also while APAI does
    # not read the arm: at most 10 a second, and no ROBOT_STATE twice.
    seen = [data["seq"] for data in poses]
    assert len(seen) >= 3 and len(set(seen)) == len(seen)
    assert len(poses) <= 10 * (ended - began) + 1
    # The arm's own read is not wrapped: nothing is recorded on the skill thread.
    assert RobotLinkArm.get_state is get_state
    assert [hook.rsplit(".", 2)[1:] for hook in made.evidence.hooks] == [
        ["Measures", "on_state"]
    ]


# Three faults the first runs on the real arm showed: the detector's model
# client, the tools the model is told about, and an empty task line.


@contextmanager
def full_stack_hooks(
    folder: Path, monkeypatch: pytest.MonkeyPatch, cloud: Any, codex_bin: str = ""
) -> Iterator[Any]:
    """launcher.agent_hooks of the full stack, on stand-ins for the experiment.

    The experiment's behaviour hooks do nothing; `cloud` is in place of its
    thor_cloud module, whose detector builds a model client of its own.
    Yields the settings of the run.
    """

    @contextmanager
    def nothing(*args: Any, **kwargs: Any) -> Iterator[None]:
        yield

    parts = {
        "register_detector": lambda: None,
        "motion_adapters": nothing,
        "perception_adapters": nothing,
        "placement_adapters": nothing,
        "target_map_adapters": nothing,
        "colors": SimpleNamespace(mask_color=None),
        "mask_color": None,
        "guidance": "\nTarget naming: the experiment's own guidance.",
    }

    def module(name: str) -> Any:
        assert name == "thor_cloud", f"the experiment's {name} was loaded"
        return cloud

    block = {"stack": "full", **({"codex_bin": codex_bin} if codex_bin else {})}
    settings = agent_config.load_settings(settings_file(folder, experiment=block))
    monkeypatch.setattr(launcher, "_experiment_parts", lambda: parts)
    monkeypatch.setattr(launcher, "experiment_module", module)
    gate = mock.Mock()
    gate.bind_run_task = lambda run_task: run_task
    with launcher.agent_hooks({}, settings, mock.Mock(), gate, folder):
        yield settings


@apai
def test_the_detectors_model_client_gets_the_program_the_file_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[dict[str, Any]] = []

    def make_client(config: dict[str, Any]) -> str:
        built.append(config)
        return "the detector's client"

    cloud = SimpleNamespace(make_client=make_client)
    # The experiment's own file for its detector: a program inside its folder,
    # which on the workstation is the Thor's copy.
    theirs = {
        "llm": {"type": "codex_exec", "codex_bin": ".tools/codex", "model": "m"},
        "segmenter": {"model": "sam.pt", "device": "cuda:0"},
    }
    with full_stack_hooks(tmp_path, monkeypatch, cloud, "codex") as settings:
        named = settings.experiment.codex_bin
        assert named == str(tmp_path / "codex")
        assert cloud.make_client(theirs) == "the detector's client"
    assert cloud.make_client is make_client  # the experiment's own again
    # The detector's client was built with the program this run names, and
    # with everything else the experiment's file says.
    (given,) = built
    assert given["llm"] == {"type": "codex_exec", "codex_bin": named, "model": "m"}
    assert given["segmenter"] == theirs["segmenter"]
    assert theirs["llm"]["codex_bin"] == ".tools/codex"  # their file is not changed
    # Without the setting: the experiment's own program, as its file names it.
    built.clear()
    with full_stack_hooks(tmp_path, monkeypatch, cloud):
        cloud.make_client(theirs)
    assert built == [theirs] and built[0] is theirs
    assert cloud.make_client is make_client


@apai
def test_the_model_is_told_which_tools_apai_names_and_the_program_does_not_offer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib
    import re

    demo = importlib.import_module("apai.apps.demo")
    orchestrator = importlib.import_module("apai.agent.orchestrator")
    prompts = importlib.import_module("apai.agent.prompts")
    apai_tools = {spec["name"] for spec in orchestrator.tool_specs_for_config(None)}
    cloud = SimpleNamespace(make_client=lambda config: None)
    with full_stack_hooks(tmp_path, monkeypatch, cloud):
        # APAI builds its agent through the name the hooks replaced.
        agent = demo.AgentOrchestrator(
            mock.Mock(supports_vision=False), SimpleNamespace(cfg=None)
        )
    instructions = agent.system_prompt
    offered = {spec["name"] for spec in agent.tool_specs}
    # The text is in the agent's instructions, behind APAI's own and the
    # experiment's.
    assert instructions.startswith(prompts.SYSTEM_PROMPT)
    assert instructions.endswith("\n" + launcher.TOOLS_GUIDANCE)
    assert "the experiment's own guidance." in instructions
    assert launcher.TOOLS_GUIDANCE.startswith(
        "Tools on this robot: the tool list you were given is complete. "
    )
    # What APAI tells the model: its instructions and the descriptions of the
    # tools that are offered. They name one tool that is not offered, and the
    # text names exactly that one as not offered. APAI's rest tool is offered,
    # as in the experiment's launcher.
    told = prompts.SYSTEM_PROMPT + " ".join(
        str(spec.get("description", "")) for spec in agent.tool_specs
    )
    missing = {
        name
        for name in apai_tools - offered
        if re.search(rf"\b{re.escape(name)}\b", told)
    }
    assert missing == {"preview_grasp"} and "move_to_rest" in offered
    not_offered = re.findall(r"(\w+) is not offered", launcher.TOOLS_GUIDANCE)
    assert sorted(not_offered) == sorted(missing)
    # Every other tool it names is offered.
    for name in apai_tools & set(re.findall(r"\w+", launcher.TOOLS_GUIDANCE)):
        assert name in offered or name in missing, name
    assert {"grasp_object", "task_done"} <= offered
    # The minimal stack keeps APAI's instructions as they are.
    monkeypatch.undo()
    minimal = agent_config.load_settings(settings_file(tmp_path))
    gate = mock.Mock()
    gate.bind_run_task = lambda run_task: run_task
    with launcher.agent_hooks({}, minimal, mock.Mock(), gate, tmp_path):
        plain = demo.AgentOrchestrator(
            mock.Mock(supports_vision=False), SimpleNamespace(cfg=None)
        )
    assert plain.system_prompt == prompts.SYSTEM_PROMPT


@apai
def test_an_empty_task_line_does_not_end_the_program(stage: Stage) -> None:
    # APAI's own prompt ends at an empty line; an Enter typed during a task
    # is such a line. Here it is read again, and only the end of the input ends.
    stage.typed = ["", "   ", QUICK, "\t", ""]
    code = stage.run("--interactive")
    assert code == launcher.EXIT_OK, stage.lines
    assert stage.typed == []  # every line was read, up to the end of the input
    assert stage.order == ["agent:completed", "nobody:completed", "close:nobody"]
    # One task was asked for; the second question is the session's end.
    assert len(stage.asked) == 2 and stage.motion == ["gripper"]
    (report,) = stage.reports()
    assert report["task"] == QUICK and report["success"] is True


@apai
def test_the_end_of_the_input_at_the_first_prompt_ends_the_program(
    stage: Stage,
) -> None:
    prompts: list[str] = []

    def next_line(prompt: str = "") -> str:
        prompts.append(prompt)
        if len(prompts) < 3:
            return ""
        raise EOFError  # Ctrl+D

    stage.next_line = next_line
    assert stage.run("--interactive") == launcher.EXIT_OK
    # The prompt came again after each empty line, with APAI's own text.
    assert prompts == ["task> "] * 3
    assert stage.asked == [] and stage.order == ["close:nobody"]
    assert not (stage.run_dir / "task-reports.jsonl").exists()


# The experiment's own modules


@apai
@experiment
def test_the_experiments_own_modules_record_a_run_on_the_mock_scene(
    stage: Stage, experiment_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No GPU, no model call and no camera: APAI's mock scene on the clear grid."""
    for name in ("thor_timing", "thor_model_trace", "thor_recording"):
        import_experiment(name)
    import_experiment("thor_milestones")
    monkeypatch.setenv(EXPERIMENT_DIR_ENV, str(experiment_dir))
    monkeypatch.setattr(launcher, "_run_options", lambda args, settings: ALL)
    execute, popen = task_gate.TaskGate.execute, subprocess.Popen
    stage.typed = [QUICK]
    # Not Stage.run: the debug recording hooks APAI's own grid client, which
    # the clear grid replaces only while the runtime is built.
    path = settings_file(stage.folder, experiment={"grid": "clear"})
    arguments = ["--config", str(path), "--camera", "mock", "--interactive"]
    code = launcher.main(
        [*arguments, "--run-dir", str(stage.run_dir)],
        attach=stage.attach,
        ask=stage.ask,
        say=stage.lines.append,
        task_input=stage.next_line,
    )
    assert code == launcher.EXIT_OK, stage.lines
    assert stage.order == ["agent:completed", "nobody:completed", "close:nobody"]
    run = stage.run_dir
    # The timing: every hook of the experiment was found, and the launcher's
    # own are among them.
    status = json.loads((run / "timing-status.json").read_text())
    # An earlier run of this process may have left APAI's shared segmenter,
    # which the timing then names as a lock it came too late for.
    missing = [hook for hook in status["missing_hooks"] if "sam.lock" not in hook]
    assert missing == [] and status["writer_error"] is None
    ours = [hook for hook in status["hooks"] if "agentic_remote" in hook]
    assert [hook.rsplit(".", 2)[1:] for hook in ours] == [
        [owner, method] for owner, method, _, _, _ in LINK_SPANS
    ]
    spans = {
        json.loads(line)["name"]
        for line in (run / "timing.jsonl").read_text().splitlines()
    }
    assert {
        "task",
        "tool:open_gripper",
        "operator.confirmation",
        "operator.terminal_input",
        "task.take_control",
        "task.give_back_control",
    } <= spans
    assert (run / "timing-report.txt").is_file() and (run / "timeline.html").is_file()
    # The model transcript: installed whole; the mock model made no request.
    trace = json.loads((run / "model-trace" / "status.json").read_text())
    assert trace["missing_hooks"] == [] and "CodexExecClient.chat" in trace["hooks"]
    assert trace["calls_started"] == 0
    # The debug recording: begun and ended, with the size the run was given.
    recording = json.loads((run / "debug" / "recording-status.json").read_text())
    assert recording["finalized"] is True and recording["writer_error"] is None
    assert recording["max_bytes"] == 2 * GIB
    assert {"session.start", "session.end"} <= set(recording["counts"])
    # The experiment's recorder took the launcher's two kinds of event, and
    # dropped none of them: the feed ended before the recording did.
    counts = recording["counts"]
    assert counts[launcher.STATE_EVENT] >= counts["robot.feedback"] >= 1
    assert recording["dropped_events"] == 0 and recording["snapshot_errors"] == 0
    assert (run / "debug" / "recording-report.txt").is_file()
    # Nothing of the experiment stays installed.
    assert task_gate.TaskGate.execute is execute and subprocess.Popen is popen

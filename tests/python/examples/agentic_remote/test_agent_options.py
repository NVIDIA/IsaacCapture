# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent program's options of the experiment's own launcher.

--batch-milestones, --record-debug, --debug-max-gib, --no-timing and
--motion-approval. The experiment is the folder of the rig's earlier, single
machine program; its launcher is thor_runtime.py. Stand-ins take the place of
the four modules the options load (thor_timing, thor_model_trace,
thor_recording, thor_milestones), and the runs put the launcher on APAI's mock
scene as test_launcher.py does. The last test runs the experiment's real
modules, when its folder is there.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import signal
import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
import test_launcher
from isaaccapture_examples.agentic_remote.agent import config as agent_config
from isaaccapture_examples.agentic_remote.agent import launcher, task_gate
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
# The spans (timed stretches) the launcher adds on the robot link arm and on the
# link's stereo camera, in the order it adds them.
LINK_SPANS = [
    ("motor.stream_to", "Execution"),
    ("gripper.close_two_stage", "Execution"),
    ("camera.pair_wait_read_depth", "Look"),
]
# What the program says at its start, one line per option.
SAID = ("timing: ", "model transcript: ", "debug recording: ", "milestone batching: ")
# What an option leaves in the run folder.
LEFT = ("run-options.json", "timing.jsonl", "model-trace", "debug")


class Recorder:
    """In place of the experiment's timing recorder: what it wrapped and timed."""

    def __init__(self) -> None:
        self.hooks: list[str] = []
        self.wrapped: list[tuple[str, str]] = []  # (span name, category)
        self.calls: list[str] = []  # the span name of each timed call

    def wrap(
        self, function: Callable[..., Any], name: str, category: str
    ) -> Callable[..., Any]:
        self.wrapped.append((name, category))

        @functools.wraps(function)
        def measured(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(name)
            return function(*args, **kwargs)

        return measured


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

    def module(self, name: str) -> SimpleNamespace:
        """In place of launcher.experiment_module."""
        return {
            "thor_timing": SimpleNamespace(timing_session=self.timing_session),
            "thor_model_trace": SimpleNamespace(
                model_trace_session=self.model_trace_session
            ),
            "thor_recording": SimpleNamespace(recording_session=self.recording_session),
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
    def recording_session(self, run_dir: Any, *, max_bytes: int) -> Iterator[None]:
        with self._session("recording", directory=run_dir, max_bytes=max_bytes):
            yield

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
    assert parsed(minimal) == RunOptions()
    assert parsed(full, "--motion-approval", "task") == parsed(full)


def test_the_command_line_of_the_experiments_launcher_is_taken(tmp_path: Path) -> None:
    full = agent_config.load_settings(
        settings_file(tmp_path, experiment={"stack": "full"})
    )
    old = (
        "--operate --interactive --record-debug --depth-backend ffs "
        "--motion-approval task --batch-milestones"
    ).split()
    # --operate and --depth-backend ffs name what the program does anyway.
    assert parsed(full, *old) == parsed(
        full, "--interactive", "--record-debug", "--batch-milestones"
    )
    with pytest.raises(SystemExit) as refused:
        parsed(full, "--depth-backend", "sgm")
    assert refused.value.code == 2


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
        (["--motion-approval", "tool"], {}, "--motion-approval tool is not built"),
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
    # As before the options were there: one typed yes, the arm given back.
    assert stage.order == ["agent:completed", "nobody:completed", "close:nobody"]
    assert len(stage.asked) == 1
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
    assert made.confirmations == 1 == len(stage.asked)
    assert made.tools.count("open_gripper") == 1
    # The batching was put on the one agent before the task gate wrapped its
    # run_task, as the experiment's launcher does.
    assert made.batched == [False]
    # The spans of the robot link arm and of the link's stereo camera, and the
    # wait for a task line: the one typed task, then the end of the input.
    assert made.recorder.wrapped == [
        *LINK_SPANS,
        ("operator.terminal_input", "Operator"),
    ]
    assert made.recorder.calls == ["operator.terminal_input"] * 2
    assert [hook.rsplit(".", 2)[1:] for hook in made.recorder.hooks] == [
        ["RobotLinkArm", "stream_to"],
        ["RobotLinkArm", "close_gripper_two_stage"],
        ["LinkStereoCamera", "_grab"],
    ]
    # Nothing stays wrapped after the run.
    assert task_gate.TaskGate.execute is execute
    assert RobotLinkArm.stream_to is stream_to
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
    assert "Not recorded here: the motor feedback and the motor commands" in recording
    assert "robot.feedback" in recording
    written = json.loads((stage.run_dir / "run-options.json").read_text())
    assert written == dataclasses.asdict(ALL)


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
    assert made.confirmations == 0 and len(stage.asked) == 1
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
        with sessions as (asking, timing):
            assert timing is None and asking("Type yes: ") == "yes"
            assert noted == ["begin:timing", "begin:model_trace"]
    assert noted[2:] == ["end:model_trace", "end:timing"]
    assert made.began["model_trace"]["timing"] is None
    assert made.recorder.wrapped == []


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
    assert [hook.rsplit(".", 1)[1] for hook in ours] == [
        "stream_to",
        "close_gripper_two_stage",
        "_grab",
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
    assert (run / "debug" / "recording-report.txt").is_file()
    # Nothing of the experiment stays installed.
    assert task_gate.TaskGate.execute is execute and subprocess.Popen is popen

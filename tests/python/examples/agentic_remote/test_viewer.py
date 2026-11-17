# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The experiment's viewer on the runs the agent program records.

The experiment is the folder of the rig's earlier, single machine program. Its
viewer (thor_viewer.py) is a web page for a browser on the recordings of a
run: the obstacle map, the measured joints, the camera frame the map took and
the timing. run_viewer.sh starts it on the workstation. The runs of these
tests are recorded by the agent program on APAI's mock scene, as
test_launcher.py runs it, with the experiment's own recorder. The viewer is
asked over HTTP on 127.0.0.1, on a port the system picks; no browser is
opened. A grid bridge is the program that keeps the obstacle grid; a stand-in
answers for it here.
"""

from __future__ import annotations

import importlib
import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar
from unittest import mock

import pytest
import test_launcher
import yaml
from isaaccapture_examples.agentic_remote.agent import config as agent_config
from isaaccapture_examples.agentic_remote.agent import launcher
from isaaccapture_examples.agentic_remote.agent.config import RunOptions
from test_launcher import EXAMPLE, PICK, QUICK, SHIPPED, Stage, settings_file

from conftest import APAI_SRC_ENV, FakeGridClient, import_experiment

apai = pytest.mark.apai
experiment = pytest.mark.experiment
# The fixtures of test_launcher.py, for the tests of this file.
make_stage = test_launcher.make_stage
stage = test_launcher.stage

SCRIPT = EXAMPLE / "run_viewer.sh"
# Real seconds a test waits for the viewer's answer to change, or for a line
# of its process.
WAIT_S = 30.0
# A run with --record-debug: the timing, the model transcript, the debug recording.
RECORDED = RunOptions(
    timing=True, model_trace=True, record_debug=True, debug_max_gib=1.0
)
# The same run without --record-debug.
TIMED = RunOptions(timing=True, model_trace=True)
# The occupied cells the stand-in for the grid bridge reports, in metres.
CELLS = [[0.25, 0.125, 0.0625], [0.375, -0.125, 0.125]]
# The pose the mock scene's arm starts a run at, in rad (test_apai_app.RAISED).
START_Q = [0.0, 0.1, 0.1, 0.0, 0.0, 0.0]
# What the viewer says of a run it does not list.
UNKNOWN_RUN = "Unknown recorded run"
# The launcher's event with what else a ROBOT_STATE reports of the motors.
STATE_EVENT = launcher.STATE_EVENT


class Bridge(FakeGridClient):
    """FakeGridClient as the experiment's recorder hooks APAI's grid client.

    The recorder wraps the private _request and reads _host and _port. Every
    query reports CELLS as occupied; the distances stay 1 m everywhere until a
    test replaces .grid. .made are the clients built so far.
    """

    _host, _port = "127.0.0.1", 0
    made: ClassVar[list[Bridge]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        Bridge.made.append(self)

    def request(self, payload: dict, timeout_ms: int | None = None) -> dict:
        return self._request(payload, timeout_ms)

    def _request(self, payload: dict, timeout_ms: int | None = None) -> dict:
        import numpy as np

        answer = FakeGridClient.request(self, payload, timeout_ms)
        if payload["action"] == "query":
            answer["points"] = np.array(CELLS, dtype=np.float32)
        return answer


def record(
    stage: Stage,
    run: Path,
    options: RunOptions,
    *tasks: str,
    between: Callable[[], None] | None = None,
) -> int:
    """One run of the agent program on the mock scene, into the folder `run`.

    The run records what `options` say. The tasks are typed one after
    another, each with its typed yes; between() is called every time the
    program waits for a task line. Returns the program's exit code.
    """
    occupancy = importlib.import_module("apai.perception.occupancy")
    stage.run_dir = run  # where the stage reads the task reports from
    stage.typed = list(tasks)
    stage.answers = ["yes"] * len(tasks)

    def next_line(prompt: str = "") -> str:
        if between is not None:
            between()
        return stage.next_line(prompt)

    arguments = ["--config", str(settings_file(stage.folder)), "--camera", "mock"]
    arguments += ["--interactive", "--run-dir", str(run)]
    with (
        mock.patch.object(launcher, "_run_options", lambda args, settings: options),
        mock.patch.object(occupancy, "OccupancyClient", Bridge),
    ):
        return launcher.main(
            arguments,
            attach=stage.attach,
            ask=stage.ask,
            say=stage.lines.append,
            task_input=next_line,
        )


class Page:
    """What the viewer's page asks of the viewer, over HTTP on 127.0.0.1."""

    def __init__(self, port: int, token: str = "") -> None:
        self.port = port
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}

    def get(self, path: str) -> tuple[int, str, bytes]:
        """(status code, content type, body) of one GET."""
        connection = HTTPConnection("127.0.0.1", self.port, timeout=WAIT_S)
        try:
            connection.request("GET", path, headers=self._headers)
            response = connection.getresponse()
            kind = response.getheader("Content-Type", "")
            return response.status, kind, response.read()
        finally:
            connection.close()

    def json(self, path: str) -> Any:
        status, kind, body = self.get(path)
        assert (status, kind) == (200, "application/json"), (path, status, body[:300])
        return json.loads(body)

    def state(self, run: str = "latest", at: float | None = None) -> dict[str, Any]:
        """/api/state: what the page draws of a run, live or at `at` seconds."""
        moment = "" if at is None else f"&at={at}"
        return self.json(f"/api/state?run={run}{moment}")

    def state_when(
        self, holds: Callable[[dict[str, Any]], Any], what: str
    ) -> dict[str, Any]:
        """Ask for the newest run's state until holds(state); at most WAIT_S."""
        end = time.monotonic() + WAIT_S
        while True:
            state = self.state()
            if holds(state):
                return state
            assert time.monotonic() < end, f"{what} did not show: {state}"
            time.sleep(0.1)

    def span_names(self, run: str) -> set[str]:
        """The names of the timed stretches /api/timing gives for a run."""
        timing = self.json(f"/api/timing?run={run}")
        assert timing["viewer"]["run"] == run
        return {span["name"] for span in timing["spans"]}


@contextmanager
def served(viewer: Any, runs: Path) -> Iterator[Page]:
    """The experiment's viewer on a runs folder, on a port the system picks."""
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), viewer.handler(viewer.Viewer(runs), None)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield Page(server.server_address[1])
    finally:
        server.shutdown()
        thread.join(WAIT_S)
        server.server_close()


@pytest.fixture
def viewer() -> Any:
    """The experiment's thor_viewer module, with the recorders a run loads.

    The test is skipped, with the reason, without the experiment folder or
    when one of the modules needs a package this interpreter lacks.
    """
    for name in ("thor_timing", "thor_model_trace", "thor_recording"):
        import_experiment(name)
    return import_experiment("thor_viewer")


def near(q: Any, expected: Any, tol: float = 0.02) -> bool:
    """Whether two joint poses agree within tol rad on every joint."""
    return len(q) == len(expected) and all(
        abs(a - b) <= tol for a, b in zip(q, expected, strict=True)
    )


# The viewer on runs the agent program records


@apai
@experiment
def test_the_viewer_follows_a_run_while_the_agent_program_records_it(
    stage: Stage, viewer: Any
) -> None:
    runs = stage.folder / "runs"
    run = runs / "20261116T120000Z-1"
    seen: list[dict[str, Any]] = []
    spans: set[str] = set()

    def arm_q() -> list[float]:
        return [float(value) for value in stage.h.spy.backend._q]

    with served(viewer, runs) as page:

        def look() -> None:
            """The program waits for a task line: the run goes on."""
            assert page.json("/api/runs") == [run.name]
            events = seen[-1]["events"] if seen else 0
            state = page.state_when(
                lambda s: (
                    s["events"] > events
                    and s["map"]
                    and s["robot"]
                    and near(s["robot"]["q"], arm_q())
                ),
                "more events, a map and the arm's pose",
            )
            assert state["recording"]["finalized"] is False
            seen.append(state)
            if len(seen) == 3:
                spans.update(page.span_names(run.name))

        code = record(stage, run, RECORDED, QUICK, PICK, between=look)
        ended = page.state()
        replayed = page.state(run.name, at=seen[0]["seconds"])
    assert code in (launcher.EXIT_OK, launcher.EXIT_FAILED), stage.lines
    assert stage.order[-1] == "close:nobody"
    assert len(stage.h.spy.targets) > 100  # the pick and place moved the arm
    first, opened, moved = seen
    # Before the first task the run is listed already and is followed live:
    # the map the stand-in bridge answered and the pose ROBOT_STATE reported.
    assert first["run"] == run.name and first["live"] is True
    assert near(first["robot"]["q"], START_Q, 1e-6)
    assert first["robot"]["source"] == launcher.FEEDBACK_SOURCE
    assert first["map"]["points"] == CELLS and first["map"]["query_present"] is True
    assert first["map"]["voxel"] == pytest.approx(0.05)
    # Each later look holds more: the run's events, its length and the map's
    # revisions grew, and the measured joints went where the arm went.
    assert first["events"] < opened["events"] < moved["events"] < ended["events"]
    assert first["duration"] < opened["duration"] < moved["duration"]
    assert first["map"]["revision"] < moved["map"]["revision"]
    assert near(opened["robot"]["q"], START_Q)
    assert not near(moved["robot"]["q"], START_Q, 0.5)
    assert moved["robot"]["source"] == launcher.FEEDBACK_SOURCE
    # The timing while the program still ran: the first task's spans.
    assert {"task", "tool:open_gripper", "operator.confirmation"} <= spans
    # After the program ended the recording is closed, and an earlier moment
    # of the run can be shown again.
    assert ended["recording"]["finalized"] is True
    assert ended["recording"]["writer_error"] is None
    assert near(replayed["robot"]["q"], START_Q, 1e-6)
    assert replayed["seconds"] == pytest.approx(first["seconds"])


@apai
@experiment
def test_the_viewer_answers_the_page_with_a_finished_runs_data_and_lists_no_other(
    stage: Stage, viewer: Any
) -> None:
    import cv2
    import numpy as np

    runs = stage.folder / "runs"
    recorded = runs / "20261116T120000Z-1"
    timed = runs / "20261116T130000Z-2"  # the newer one, without --record-debug
    assert record(stage, recorded, RECORDED, QUICK) == launcher.EXIT_OK, stage.lines
    assert record(stage, timed, TIMED, QUICK) == launcher.EXIT_OK, stage.lines
    # A run folder as the agent program left it before it wrote APAI's
    # configuration: a debug recording and no resolved-config.yaml.
    earlier = runs / "20261115T000000Z-3"
    (earlier / "debug").mkdir(parents=True)
    (earlier / "debug" / "events.jsonl").write_text("")
    # What the viewer lists a run by, and what each of the two others lacks.
    for name in ("debug/events.jsonl", agent_config.RESOLVED_CONFIG):
        assert (recorded / name).is_file(), name
    assert (timed / agent_config.RESOLVED_CONFIG).is_file()
    assert (timed / "timing.jsonl").is_file() and not (timed / "debug").exists()
    with served(viewer, runs) as page:
        assert page.json("/api/runs") == [recorded.name]
        assert page.json("/api/bookmarks") == {}
        state = page.state()
        assert state["run"] == recorded.name and state["failures"] == []
        assert state["map"]["points"] == CELLS and state["map"]["revision"] >= 1
        assert near(state["robot"]["q"], START_Q, 1e-6)
        assert state["robot"]["source"] == launcher.FEEDBACK_SOURCE
        assert state["recording"]["finalized"] is True
        assert state["recording"]["counts"]["robot.feedback"] >= 1
        # The camera frame the map took, as the page's picture.
        status, kind, body = page.get(f"/api/image?run={recorded.name}")
        assert (status, kind) == (200, "image/jpeg")
        picture = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        assert picture is not None and picture.shape[2] == 3
        # The timing: as data, and as the page the viewer's page embeds.
        assert {"task", "tool:open_gripper"} <= page.span_names(recorded.name)
        status, kind, body = page.get(f"/timing?run={recorded.name}")
        assert (status, kind.split(";")[0]) == (200, "text/html")
        assert b"Run timing" in body
        # The page itself, with the run list and the newest state inside it.
        status, kind, body = page.get("/")
        assert (status, kind.split(";")[0]) == (200, "text/html")
        start = b'<script id="initial" type="application/json">'
        inside = body[body.index(start) + len(start) :]
        initial = json.loads(inside[: inside.index(b"</script>")])
        assert initial["runs"] == [recorded.name]
        assert initial["state"]["run"] == recorded.name
        # The two other runs are left out, also when they are asked for by
        # name: the viewer shows no timing without a debug recording.
        for name in (timed.name, earlier.name):
            for path in ("/api/state", "/api/image", "/api/timing", "/timing"):
                status, _, body = page.get(f"{path}?run={name}")
                assert status == 400 and UNKNOWN_RUN.encode() in body, (name, path)


@apai
@experiment
def test_the_viewer_shows_the_pose_the_workstation_checker_refused(
    stage: Stage, viewer: Any
) -> None:
    import numpy as np

    runs = stage.folder / "runs"
    run = runs / "20261116T120000Z-1"
    Bridge.made.clear()

    def obstacle(number: int) -> None:
        if number == 5:  # the arm is under way: the grid's next answer is near
            Bridge.made[-1].grid = np.zeros_like(Bridge.made[-1].grid)

    stage.h.spy.after_target.append(obstacle)
    assert record(stage, run, RECORDED, PICK) == launcher.EXIT_FAILED, stage.lines
    (result,) = stage.results("pick_and_place")
    # The skill failed on the refusal; as with APAI's own arm the result does
    # not end the task by itself.
    assert "motion_blocked" not in result and "clearance" in result["error"]
    stopped_at = [float(value) for value in stage.h.spy.backend._q]
    with served(viewer, runs) as page:
        listed = page.state()["failures"]
        assert listed, "the recording holds no refusal"
        first = listed[0]
        assert "clearance" in first["reason"] and "occupancy map" in first["reason"]
        # The moment of the refusal: the refused pose, the point that was too
        # near and the map revision it was checked against.
        state = page.state(run.name, at=first["seconds"])
        refusal = state["failure"]
        assert refusal["id"] == first["id"] and refusal["gate"] == "tcp"
        pose = refusal["candidate"]
        assert near(pose["q"], stopped_at, 0.05)  # just ahead of where it held
        assert len(pose["joints"]) == 8
        sample = refusal["report"]["sample"]
        assert sample == pytest.approx(pose["joints"][-1], abs=1e-6)  # the tip
        assert refusal["report"]["clearance_m"] < refusal["report"]["margin_m"]
        assert refusal["map_matches"] is True
        assert refusal["report"]["gate_refreshes"] == state["map"]["revision"]
        assert refusal["obstacles"]["nearest"] in CELLS
        # The page's link to that refusal, and its camera frame.
        status, _, body = page.get(f"/?run={run.name}&failure=0")
        assert status == 200 and first["reason"][:40].encode() in body
        status, kind, _ = page.get(f"/api/image?run={run.name}&at={first['seconds']}")
        assert (status, kind) == (200, "image/jpeg")


@apai
@experiment
def test_the_viewers_arm_follows_the_steady_feed_of_robot_state(
    apai_scratch: Path, experiment_dir: Path, viewer: Any
) -> None:
    """Nothing reads the arm here: no agent, no arm adapter, no camera.

    The poses the viewer draws are those of the ROBOT_STATEs alone, which the
    launcher's Measures gives the experiment's own recorder.
    """
    import dataclasses

    from isaaccapture_examples.agentic_remote.agent.link_client import AgentLink
    from isaaccapture_examples.agentic_remote.protocol.messages import JointSample
    from link_bench import FakeThor, robot_state, until

    recording = import_experiment("thor_recording")
    runs = apai_scratch / "runs"
    run = runs / "20261116T120000Z-1"
    settings = agent_config.load_settings(settings_file(apai_scratch))
    cfg = agent_config.build_apai_config(settings, run, camera="mock")
    agent_config.write_resolved_config(run, cfg)
    thor = FakeThor("test")
    link = AgentLink(thor.service_end)
    count = 24  # ROBOT_STATEs, 20 a second of the Thor's clock

    def pose(k: int) -> list[float]:
        """The arm as ROBOT_STATE k measured it: the first joint turns on."""
        return [0.02 * k, *START_Q[1:]]

    def state(k: int) -> Any:
        mono = 1000.0 + 0.05 * k
        # One joint is read every 4 ms: the six readings span 20 ms.
        joints = [
            JointSample(j, value, mono - 0.004 * (6 - j), mono - 0.004 * (5 - j), True)
            for j, value in enumerate(pose(k))
        ]
        return dataclasses.replace(
            robot_state(seq=k + 1),
            clock_mono_s=mono,
            clock_wall_s=mono + 1.7e9,
            joints=joints,
            health={"reading_age_s": 0.0},
        )

    try:
        with recording.recording_session(run, max_bytes=2**30) as recorder:
            measures = launcher.Measures(run, recorder)
            measures.attach(link)
            with served(viewer, runs) as page:
                began = time.monotonic()
                for k in range(count):
                    thor.send(state(k))
                    time.sleep(0.05)
                    if k == count // 2:
                        # While the run goes on the page follows the arm.
                        live = page.state_when(
                            lambda s: s["robot"] and s["robot"]["q"][0] > 0.1,
                            "the arm under way",
                        )
                        assert live["live"] is True
                until(lambda: measures.states == count, "the ROBOT_STATEs")
                took = time.monotonic() - began
                measures.close()
                events = run / "debug" / "events.jsonl"

                def written() -> list[dict[str, Any]]:
                    text = events.read_text()
                    whole = text[: text.rfind("\n") + 1]  # no line half written
                    return [json.loads(line) for line in whole.splitlines()]

                def last_state() -> int:
                    kinds = [row for row in written() if row["kind"] == STATE_EVENT]
                    return kinds[-1]["data"]["seq"] if kinds else 0

                # The recorder writes on a thread of its own: wait for the last.
                until(lambda: last_state() == count, "the last ROBOT_STATE on disk")
                rows = written()
                start = rows[0]["mono_ns"]
                poses = [row for row in rows if row["kind"] == "robot.feedback"]
                # Every second ROBOT_STATE is a pose: ten a second, made on
                # the measuring thread, each at most one period of the feed
                # after the one before.
                assert [row["data"]["seq"] for row in poses] == list(range(1, count, 2))
                assert {row["thread"] for row in poses} == {launcher.MEASURE_THREAD}
                assert len(poses) <= 10 * took + 1
                # The page at the moment of each pose: the joints of that
                # ROBOT_STATE, where they came from and how far apart they
                # were read.
                drawn = []
                for row in poses:
                    at = (row["mono_ns"] - start) / 1e9 + 0.001
                    robot = page.state(run.name, at=at)["robot"]
                    k = row["data"]["seq"] - 1
                    assert near(robot["q"], pose(k), 1e-9), k
                    assert robot["source"] == launcher.FEEDBACK_SOURCE
                    assert robot["sample_span_s"] == pytest.approx(0.02)
                    assert 0.0 <= robot["age_s"] < 0.5
                    assert len(robot["joints"]) == 8
                    drawn.append(robot["q"][0])
                assert len(set(drawn)) == len(poses) == 12  # twelve different poses
    finally:
        link.close()
        thor.carrier.close()
    status = json.loads((run / "debug" / "recording-status.json").read_text())
    assert status["finalized"] is True and status["dropped_events"] == 0
    assert status["counts"]["robot.feedback"] == 12
    assert status["counts"][STATE_EVENT] == count
    assert any(hook.endswith("Measures.on_state") for hook in status["hooks"])


@apai
@experiment
def test_the_full_stacks_configuration_gives_the_viewer_the_arm_and_its_volumes(
    apai_scratch: Path, experiment_dir: Path, viewer: Any
) -> None:
    settings = agent_config.load_settings(SHIPPED)
    run = apai_scratch / "runs" / "rig"
    cfg = agent_config.build_apai_config(settings, run, camera="mock")
    agent_config.write_resolved_config(run, cfg)
    # The file is APAI's configuration as the experiment's launcher writes it.
    written = yaml.safe_load((run / agent_config.RESOLVED_CONFIG).read_text())
    assert written == cfg.as_dict()
    assert Path(written["arm"]["model"]).is_file()
    assert "table_z" in written["safety"]
    (run / "debug").mkdir()
    (run / "debug" / "events.jsonl").write_text("")
    assert viewer.Viewer(run.parent).names() == [run.name]
    # The viewer builds the arm from it: the links of a pose and the gripper's
    # collision volumes, which the mock scene's arm profile does not have.
    drawn = viewer.RunIndex(run).geometry(written["arm"]["home_q"])
    assert len(drawn["joints"]) == written["arm"]["n_joints"] + 2
    assert [box["name"] for box in drawn["boxes"]] == ["housing", "fingers", "jaws"]
    assert all(len(box["corners"]) == 8 for box in drawn["boxes"])


# The start script


def script_run(
    script: Path, *arguments: str, **env: str
) -> subprocess.CompletedProcess[str]:
    """bash <script> <arguments>, with these variables and PATH and no other."""
    return subprocess.run(
        ["bash", str(script), *arguments],
        env={"PATH": os.environ["PATH"], **env},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@pytest.fixture
def stand_in(tmp_path: Path) -> dict[str, Any]:
    """A copy of run_viewer.sh in an example folder of its own, and stand-ins.

    "script" is the copy, so its state folder lies under tmp_path. "env" are
    the three variables: a Python that prints one line per variable it was
    given and per argument, an experiment folder with an empty thor_viewer.py,
    and an APAI src folder.
    """
    example = tmp_path / "example"
    example.mkdir()
    script = example / "run_viewer.sh"
    script.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    python = tmp_path / "python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"PYTHONPATH=$PYTHONPATH\" "
        '"BYTECODE=$PYTHONDONTWRITEBYTECODE" "LD=${LD_LIBRARY_PATH:-unset}" '
        '"CUDA=$CUDA_DEVICE_ORDER:$CUDA_VISIBLE_DEVICES" "$@"\n'
    )
    python.chmod(0o755)
    folder = tmp_path / "experiment"
    folder.mkdir()
    (folder / "thor_viewer.py").write_text("")
    src = tmp_path / "apai-src"
    src.mkdir()
    env = {
        "AGENTIC_REMOTE_AGENT_PYTHON": str(python),
        "AGENTIC_REMOTE_EXPERIMENT_DIR": str(folder),
        "AGENTIC_REMOTE_APAI_SRC": str(src),
    }
    return {"script": script, "env": env, "example": example}


def test_the_start_script_gives_the_viewer_its_runs_its_token_and_this_machine(
    stand_in: dict[str, Any], tmp_path: Path
) -> None:
    script, env, example = stand_in["script"], stand_in["env"], stand_in["example"]
    text = script.read_text(encoding="utf-8")
    assert "[ " not in text.replace("[[ ", "")  # [[ ]] only
    parsed = subprocess.run(
        ["bash", "-n", str(script)], capture_output=True, check=False
    )
    assert parsed.returncode == 0, parsed.stderr
    state = example / ".state"
    viewer_file = Path(env["AGENTIC_REMOTE_EXPERIMENT_DIR"]) / "thor_viewer.py"
    environment = [
        f"PYTHONPATH={env['AGENTIC_REMOTE_APAI_SRC']}",
        "BYTECODE=1",
        "LD=unset",
        "CUDA=PCI_BUS_ID:0",
    ]
    # Without an argument: the runs and the token in the example's own state
    # folder, never in the experiment folder, and 127.0.0.1.
    done = script_run(script, **env, LD_LIBRARY_PATH="/somewhere")
    assert (done.returncode, done.stderr) == (0, "")
    said, token_line, *given = done.stdout.splitlines()
    assert said.startswith(f"run_viewer.sh: the runs in {state / 'runs'}; ")
    assert f"the text of {state / 'viewer.token'} behind it" in token_line
    assert given == [
        *environment,
        str(viewer_file),
        "--host",
        "127.0.0.1",
        "--runs",
        str(state / "runs"),
        "--token-file",
        str(state / "viewer.token"),
    ]
    # The state folder was made as the agent program makes it.
    assert (state / "runs").is_dir()
    assert (state / ".gitignore").read_text() == "*\n"
    # With the token there, the script gives what the browser needs once.
    (state / "viewer.token").write_text("t" * 43 + "\n")
    done = script_run(script, **env)
    assert done.stdout.splitlines()[1].endswith("?token=" + "t" * 43)
    # The caller's arguments go to the viewer as they are, behind the
    # script's own, so they win; what the caller names gets no default.
    elsewhere = tmp_path / "other runs"
    elsewhere.mkdir()
    for arguments in (
        ["--runs", str(elsewhere), "--port", "0", "--host", "192.0.2.9", "--no-auth"],
        [f"--runs={elsewhere}", "--token-file", str(tmp_path / "token")],
        [f"--token-file={tmp_path / 'token'}", "--runs", str(elsewhere)],
    ):
        done = script_run(script, *arguments, **env)
        assert (done.returncode, done.stderr) == (0, ""), arguments
        lines = done.stdout.splitlines()
        assert lines[0].startswith(f"run_viewer.sh: the runs in {elsewhere}; ")
        at = lines.index(str(viewer_file))
        assert lines[at + 1 :] == ["--host", "127.0.0.1", *arguments], arguments
        # Without a token no line about one.
        assert ("?token=" in done.stdout) is ("--no-auth" not in arguments)


def test_the_start_script_makes_no_state_folder_when_the_caller_names_both(
    stand_in: dict[str, Any], tmp_path: Path
) -> None:
    script, env, example = stand_in["script"], stand_in["env"], stand_in["example"]
    done = script_run(script, "--runs", str(tmp_path), "--no-auth", **env)
    assert done.returncode == 0 and not (example / ".state").exists()
    done = script_run(script, "--help", **env)
    assert done.returncode == 0 and not (example / ".state").exists()
    assert done.stdout.startswith("Start the experiment's viewer on the workstation")
    assert done.stdout.rstrip().endswith("that cannot be made.")


@pytest.mark.parametrize(
    ("change", "arguments", "message"),
    [
        (
            {"AGENTIC_REMOTE_AGENT_PYTHON": "/no/python"},
            [],
            "AGENTIC_REMOTE_AGENT_PYTHON must name a Python",
        ),
        (
            {"AGENTIC_REMOTE_AGENT_PYTHON": ""},
            [],
            "AGENTIC_REMOTE_AGENT_PYTHON must name a Python",
        ),
        (
            {"AGENTIC_REMOTE_EXPERIMENT_DIR": ""},
            [],
            "AGENTIC_REMOTE_EXPERIMENT_DIR must name the experiment folder",
        ),
        (
            {"AGENTIC_REMOTE_EXPERIMENT_DIR": "/no/experiment"},
            [],
            "AGENTIC_REMOTE_EXPERIMENT_DIR must name the experiment folder",
        ),
        (
            {"AGENTIC_REMOTE_EXPERIMENT_DIR": "{src}"},
            [],
            "AGENTIC_REMOTE_EXPERIMENT_DIR names a folder without thor_viewer.py",
        ),
        (
            {"AGENTIC_REMOTE_APAI_SRC": "/no/apai"},
            [],
            "AGENTIC_REMOTE_APAI_SRC must name the src folder",
        ),
        ({}, ["--runs"], "--runs needs a value"),
        ({}, ["--no-auth", "--token-file"], "--token-file needs a value"),
        ({}, ["--runs", "/no/runs"], "--runs names no folder: '/no/runs'"),
        ({}, ["--runs="], "--runs names no folder: ''"),
    ],
)
def test_the_start_script_refuses_what_it_cannot_use_with_one_line_and_exit_code_2(
    stand_in: dict[str, Any],
    change: dict[str, str],
    arguments: list[str],
    message: str,
) -> None:
    env = dict(stand_in["env"])
    src = env["AGENTIC_REMOTE_APAI_SRC"]  # a folder without thor_viewer.py
    env.update({name: value.format(src=src) for name, value in change.items()})
    done = script_run(stand_in["script"], *arguments, **env)
    # One line, and nothing started or made: the stand-in Python printed nothing.
    assert (done.returncode, done.stdout) == (2, "")
    assert done.stderr.startswith("run_viewer.sh: ") and done.stderr.count("\n") == 1
    assert message in done.stderr
    assert not (stand_in["example"] / ".state").exists()


def test_the_start_script_refuses_a_state_folder_it_cannot_make(
    stand_in: dict[str, Any],
) -> None:
    example = stand_in["example"]
    (example / ".state").write_text("a file where the folder would be")
    done = script_run(stand_in["script"], **stand_in["env"])
    assert (done.returncode, done.stdout) == (2, "")
    assert done.stderr.count("\n") == 1
    assert f"the state folder cannot be made: {example / '.state'}" in done.stderr
    assert "--runs" in done.stderr and "--no-auth" in done.stderr


class ViewerProcess:
    """run_viewer.sh as a process: bash run_viewer.sh <arguments>."""

    def __init__(self, *arguments: str, env: dict[str, str]) -> None:
        self.text = ""  # what the process wrote, both streams
        self.process = subprocess.Popen(
            ["bash", str(SCRIPT), *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={"PATH": os.environ["PATH"], **env},
        )

    def line(self, piece: str) -> str:
        """The first whole line that holds `piece`; at most WAIT_S."""
        assert self.process.stdout is not None
        fd = self.process.stdout.fileno()
        end = time.monotonic() + WAIT_S
        while True:
            for line in self.text.split("\n")[:-1]:
                if piece in line:
                    return line
            left = end - time.monotonic()
            assert left > 0, f"no line with {piece!r} in: {self.text}"
            if select.select([fd], [], [], left)[0]:
                data = os.read(fd, 65536)
                assert data, f"the viewer ended without {piece!r}: {self.text}"
                self.text += data.decode("utf-8", "replace")

    def end(self) -> int:
        """Ctrl+C; the exit code."""
        self.process.send_signal(signal.SIGINT)
        return self.process.wait(WAIT_S)

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGKILL)
            self.process.wait(WAIT_S)
        if self.process.stdout is not None:
            self.process.stdout.close()


@apai
@experiment
def test_the_start_script_starts_the_experiments_viewer_on_the_agents_runs(
    stage: Stage, viewer: Any, experiment_dir: Path, tmp_path: Path
) -> None:
    runs = stage.folder / "runs"
    run = runs / "20261116T120000Z-1"
    assert record(stage, run, RECORDED, QUICK) == launcher.EXIT_OK, stage.lines
    token_file = tmp_path / "private" / "viewer.token"
    env = {
        "AGENTIC_REMOTE_AGENT_PYTHON": sys.executable,
        "AGENTIC_REMOTE_EXPERIMENT_DIR": str(experiment_dir),
        "AGENTIC_REMOTE_APAI_SRC": os.environ[APAI_SRC_ENV],
    }
    arguments = ["--runs", str(runs), "--port", "0", "--token-file", str(token_file)]
    program = ViewerProcess(*arguments, env=env)
    try:
        # The viewer's own first line: the address, on this machine only and
        # on the port the system picked.
        address = program.line("Thor read-only map viewer: http://")
        host, port = address.split("http://")[1].rstrip("/").split(":")
        assert host == "127.0.0.1" and int(port) > 0
        program.line("Access token file: ")
        # The viewer made its token where the script told it, for the user only.
        token = token_file.read_text().strip()
        assert token_file.stat().st_mode & 0o777 == 0o600
        # Without the token nothing is answered; with it, the run's data.
        assert Page(int(port)).get("/api/runs")[0] == 401
        page = Page(int(port), token)
        assert page.json("/api/runs") == [run.name]
        state = page.state()
        assert state["map"]["points"] == CELLS
        assert near(state["robot"]["q"], START_Q, 1e-6)
        assert "tool:open_gripper" in page.span_names(run.name)
        # The link the script names opens the page: the token becomes a cookie.
        status, _, _ = page.get(f"/?token={token}")
        assert status == 303
        assert program.end() == 0, program.text
    finally:
        program.close()
    # A second start names the token in the script's own line.
    again = ViewerProcess(*arguments, env=env)
    try:
        assert again.line("once per browser").endswith(f"?token={token}")
        address = again.line("Thor read-only map viewer: http://127.0.0.1:")
        port = address.rstrip("/").rsplit(":", 1)[1]
        # An answer says that the viewer serves: Ctrl+C then ends it cleanly.
        assert Page(int(port), token).json("/api/runs") == [run.name]
        assert again.end() == 0, again.text
    finally:
        again.close()


@experiment
def test_the_script_and_the_readme_name_the_viewers_own_port_and_files(
    experiment_dir: Path,
) -> None:
    source = (experiment_dir / "thor_viewer.py").read_text(encoding="utf-8")
    # What the script's head says of the viewer, read from the viewer itself.
    assert 'parser.add_argument("--port", type=int, default=8766)' in source
    assert 'parser.add_argument("--host", default="127.0.0.1")' in source
    script = SCRIPT.read_text(encoding="utf-8")
    assert "#   --port NUMBER      default: the viewer's own, 8766" in script
    # The viewer's own places for the runs and the token are in its folder,
    # which is why the script names both.
    assert 'default=ROOT / ".state/thor/runtime"' in source
    assert 'default=ROOT / ".state/thor/viewer.token"' in source
    # The two files the viewer lists a run by.
    assert '(p / "debug/events.jsonl").is_file()' in source
    assert f'(p / "{agent_config.RESOLVED_CONFIG}").is_file()' in source

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where a record lands, for each source, process shape and condition.

The chart in AGENT_B_REPORT.md is this file. Each test fixes one row of it:
a source (Python logger, in-process C++, out-of-process C++, a raw descriptor
write, a subprocess's stdio), a process shape (leader, forwarding child,
process that dropped the address), and a destination.
"""

from __future__ import annotations

import json
import re
import stat

import pytest
from conftest import (
    LINE_RE,
    capture_logs,
    clean_env,
    cpp_logs,
    emitter_path,
    read,
    read_all,
    requires_emitter,
    run_emitter,
    run_python,
    session_logs,
)

_PROBE = """
import json, logging, os, sys
import isaaccapture  # noqa: F401
log = logging.getLogger("isaaccapture.test.routing")
log.debug("python debug")
log.info("python info")
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(
        {
            "socket": os.environ.get("ISAACCAPTURE_LOG_SOCKET"),
            "capture": os.environ.get("ISAACCAPTURE_NATIVE_CAPTURE_FILE"),
            "pid": os.getpid(),
        },
        handle,
    )
"""


def run_probe(tmp_path, log_dir, **overrides):
    report_path = tmp_path / "probe.json"
    result = run_python(_PROBE, clean_env(log_dir, **overrides), str(report_path))
    assert result.returncode == 0, result.stderr
    return result, json.loads(report_path.read_text(encoding="utf-8"))


def line_with(text: str, needle: str) -> str:
    matches = [line for line in text.splitlines() if needle in line]
    assert matches, f"{needle!r} is not in:\n{text}"
    return matches[0]


class TestPythonLeader:
    def test_the_console_shows_info_and_the_file_keeps_everything(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        result, _ = run_probe(tmp_path, log_dir)

        assert "python info" in result.stderr
        assert "python debug" not in result.stderr

        logged = read_all(session_logs(log_dir))
        assert "python info" in logged
        assert "python debug" in logged
        assert LINE_RE.fullmatch(line_with(logged, "python info")) is not None
        # The console defaults to the short columns: no pid, last name segment.
        assert re.fullmatch(
            r"\[\d{2}:\d{2}:\d{2}\.\d{3}\] \[INFO \] \[routing\] python info",
            line_with(result.stderr, "python info"),
        )


@requires_emitter
class TestStandaloneCpp:
    """A process with no interpreter: a plugin executable run on its own."""

    def test_writes_a_line_in_the_python_format_to_its_own_file(self, tmp_path):
        name = "isaaccapture.plugins.probe.Standalone"
        result = run_emitter(
            ["emit", name, "info", "standalone speaking"], clean_env(tmp_path)
        )
        assert result.returncode == 0

        files = cpp_logs(tmp_path)
        assert len(files) == 1
        assert stat.S_IMODE(files[0].stat().st_mode) == 0o600

        match = LINE_RE.fullmatch(line_with(read(files[0]), "standalone speaking"))
        assert match is not None
        assert match["name"] == name
        assert match["level"] == "INFO "
        assert match["message"] == "standalone speaking"
        assert match["pid"].isdigit()

        # Diagnostics go to stderr, so `tool > data.txt` keeps both intact.
        assert LINE_RE.fullmatch(line_with(result.stderr, "standalone speaking"))
        assert result.stdout == ""

    def test_the_console_threshold_comes_from_the_environment(self, tmp_path):
        name = "isaaccapture.plugins.probe.Threshold"
        result = run_emitter(
            ["emit", name, "info", "below the bar", "warning", "above the bar"],
            clean_env(tmp_path, ISAACCAPTURE_LOG_LEVEL="warning"),
        )
        assert result.returncode == 0
        assert "above the bar" in result.stderr
        assert "below the bar" not in result.stderr

        # The file is not user-configurable; it always captures everything.
        logged = read_all(cpp_logs(tmp_path))
        assert "below the bar" in logged
        assert "above the bar" in logged

    def test_a_dead_address_sends_it_back_to_its_own_sinks(self, tmp_path):
        result = run_emitter(
            ["emit", "isaaccapture.plugins.probe.Dead", "info", "fell back"],
            clean_env(tmp_path, ISAACCAPTURE_LOG_SOCKET=str(tmp_path / "nothing.sock")),
        )
        assert result.returncode == 0
        # A standalone executable has no Python half to unset the variable, so
        # socket_sink.cpp verifies the address for itself.
        assert "fell back" in read_all(cpp_logs(tmp_path))
        assert "fell back" in result.stderr

    def test_the_two_halves_never_want_the_same_file_name(self, tmp_path):
        # A rotating file tolerates exactly one writer, and on Windows -- or
        # wherever ensure_receiver() cannot bind -- both halves want one here.
        shared = tmp_path / "shared"
        shared.mkdir()
        run_probe(tmp_path, shared)
        run_emitter(
            ["emit", "isaaccapture.plugins.probe.Shared", "info", "cpp line"],
            clean_env(shared),
        )

        assert len(session_logs(shared)) == 1
        assert len(cpp_logs(shared)) == 1
        assert session_logs(shared)[0] != cpp_logs(shared)[0]
        assert "python info" in read(session_logs(shared)[0])
        assert "cpp line" in read(cpp_logs(shared)[0])


@requires_emitter
def test_a_standalone_cpp_process_forwards_into_the_leaders_file(leader, tmp_path):
    name = "isaaccapture.plugins.probe.Forwarded"
    child_logs = tmp_path / "child-logs"
    child_logs.mkdir()
    result = run_emitter(
        ["emit", name, "warning", "shipped to the leader"],
        clean_env(child_logs, ISAACCAPTURE_LOG_SOCKET=leader.socket),
    )
    assert result.returncode == 0

    match = LINE_RE.fullmatch(
        line_with(leader.wait_for("shipped to the leader"), "shipped to the leader")
    )
    assert match is not None
    assert match["name"] == name
    assert match["level"] == "WARNING"
    # The sender's pid, not the receiver's: the leader re-emits, it does not
    # re-author.
    assert int(match["pid"]) != leader.pid
    assert "shipped to the leader" in leader.console()

    # Only the leader touches disk or a terminal for isaaccapture records.
    assert cpp_logs(child_logs) == []
    assert result.stderr == ""


def test_an_in_process_cpp_logger_loops_back_through_the_socket(tmp_path):
    # Every pybind extension carries its own copy of local_sinks(), which picks
    # the forwarding sink because the leader published its own receiver, so the
    # record comes back into this same process's Python tree.
    plugins = tmp_path / "plugins" / "broken"
    plugins.mkdir(parents=True)
    (plugins / "plugin.yaml").write_text("name: [unclosed\n", encoding="utf-8")

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    report_path = tmp_path / "report.json"
    result = run_python(
        """
import glob, json, os, sys, time
import isaaccapture  # noqa: F401
from isaaccapture.plugin_manager import PluginManager

PluginManager([sys.argv[2]])


def logged():
    text = ""
    for name in glob.glob(os.path.join(os.environ["ISAACCAPTURE_LOG_DIR"], "*.log")):
        with open(name, encoding="utf-8", errors="replace") as handle:
            text += handle.read()
    return text


# The receiver is a thread of this process: it has to be given the chance to
# read the frame before the interpreter goes away under it.
deadline = time.monotonic() + 30
while time.monotonic() < deadline and "Error parsing metadata" not in logged():
    time.sleep(0.05)
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump({"socket": os.environ.get("ISAACCAPTURE_LOG_SOCKET")}, handle)
""",
        clean_env(log_dir),
        str(report_path),
        str(tmp_path / "plugins"),
    )
    assert result.returncode == 0, result.stderr
    if not json.loads(report_path.read_text(encoding="utf-8"))["socket"]:
        pytest.skip("this interpreter could not start a receiver")

    logged = read_all(session_logs(log_dir))
    match = LINE_RE.fullmatch(line_with(logged, "Error parsing metadata"))
    assert match is not None
    assert match["name"] == "isaaccapture.core.PluginManager"
    assert match["level"] == "ERROR"
    assert "Error parsing metadata" in result.stderr


class TestProcessShape:
    def test_a_forwarding_child_keeps_no_console_and_no_file(self, leader, tmp_path):
        child_logs = tmp_path / "child-logs"
        child_logs.mkdir()
        result, report = run_probe(
            tmp_path, child_logs, ISAACCAPTURE_LOG_SOCKET=leader.socket
        )

        assert session_logs(child_logs) == []
        assert "python info" not in result.stderr

        logged = leader.wait_for("python debug")
        assert "python info" in logged
        # The child forwards below its console threshold too; the leader's file
        # is what decides.
        assert "python debug" in logged

        # It still opens a capture file, for processes it launches in turn.
        assert report["capture"] is not None

    def test_a_process_that_drops_the_address_becomes_its_own_leader(
        self, leader, tmp_path
    ):
        # What cloudxr/background.py does for a service meant to outlive its
        # launcher: the publisher would otherwise stop answering under it.
        detached_logs = tmp_path / "detached-logs"
        detached_logs.mkdir()
        result, report = run_probe(tmp_path, detached_logs)

        assert len(session_logs(detached_logs)) == 1
        assert "python info" in read_all(session_logs(detached_logs))
        assert "python info" in result.stderr
        assert report["socket"] != leader.socket
        assert "python info" not in leader.logged()


@requires_emitter
class TestPluginLaunchWindow:
    """The fork()/execvp() window in plugin_manager/cpp/plugin.cpp."""

    @staticmethod
    def _search_dir(tmp_path, hold_ms: int = 20000):
        plugin = tmp_path / "plugins" / "capture_probe"
        plugin.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text(
            "name: capture_probe\n"
            f"command: {emitter_path()} raw plugin_says_hello {hold_ms}\n",
            encoding="utf-8",
        )
        return tmp_path / "plugins"

    _RUN = """
import json, os, sys, time
import isaaccapture  # noqa: F401
from isaaccapture.logging_config import _native_api
from isaaccapture.plugin_manager import PluginManager

report_path, search, override, budget = sys.argv[1:5]
if override:
    os.environ["ISAACCAPTURE_NATIVE_CAPTURE_FILE"] = override

manager = PluginManager([search])
plugin = manager.start("capture_probe", "probe-root")
watched = override or str(_native_api.native_capture_path())
deadline = time.monotonic() + float(budget)
text = ""
while time.monotonic() < deadline:
    try:
        with open(watched, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        text = ""
    if "plugin_says_hello" in text:
        break
    time.sleep(0.05)
try:
    plugin.stop()
except Exception:
    pass
with open(report_path, "w", encoding="utf-8") as handle:
    json.dump({"watched": watched, "text": text}, handle)
"""

    def _launch(self, tmp_path, override: str = "", budget: str = "30"):
        log_dir = tmp_path / "logs"
        log_dir.mkdir(exist_ok=True)
        report_path = tmp_path / "plugin.json"
        result = run_python(
            self._RUN,
            clean_env(log_dir),
            str(report_path),
            str(self._search_dir(tmp_path)),
            override,
            budget,
        )
        assert result.returncode == 0, result.stderr
        return result, log_dir, json.loads(report_path.read_text(encoding="utf-8"))

    def test_a_plugins_own_output_lands_in_the_capture_file(self, tmp_path):
        # The child's descriptors are ours to set; the host's are not. This is
        # what keeps a plugin's non-logger output off the operator's terminal.
        result, log_dir, report = self._launch(tmp_path)

        assert "plugin_says_hello" in report["text"]
        assert "plugin_says_hello" in read_all(capture_logs(log_dir))
        assert "plugin_says_hello" not in result.stdout
        assert "plugin_says_hello" not in result.stderr

    def test_a_capture_file_that_is_not_ours_is_refused(self, tmp_path):
        # O_NOFOLLOW refuses a symlink but not a plain file substituted at the
        # same path, which would collect the whole plugin's output.
        planted = tmp_path / "planted.log"
        planted.write_text("", encoding="utf-8")
        planted.chmod(0o666)

        # The plugin writes immediately after exec, so a short budget is enough
        # to show the bytes did not go there.
        result, _, _ = self._launch(tmp_path, override=str(planted), budget="2")

        assert planted.read_text(encoding="utf-8") == ""
        # Refused, not redirected somewhere else: the plugin keeps the
        # descriptors it inherited.
        assert "plugin_says_hello" in result.stdout

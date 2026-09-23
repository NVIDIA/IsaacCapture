# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `python -m isaaccapture.cloudxr.service` CLI."""

import os
from unittest.mock import MagicMock, patch

import pytest

from isaaccapture.cloudxr.service import __main__ as cli


@pytest.fixture(autouse=True)
def stub_runtime_version():
    """Keep the summary from dlopening libcloudxr, which needs the GPU driver."""
    with patch(
        "isaaccapture.cloudxr.runtime.runtime_version", return_value="0.0.0-test"
    ):
        yield


def _run_args(**overrides):
    """Parse a `run`/`install` argument set, applying *overrides* after."""
    parser = cli._build_parser()
    args = parser.parse_args(["run"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestRunFlags:
    """Tests for re-serialising run flags into ExecStart."""

    def test_defaults_produce_no_flags(self):
        """A default install renders a bare `service run`."""
        assert cli._run_flags(_run_args()) == []

    def test_only_non_default_values_are_emitted(self):
        args = _run_args(setup_oob=True, host_client=True)
        assert cli._run_flags(args) == [
            "--setup-oob",
            "--host-client",
        ]

    def test_install_dir_emitted_only_when_changed(self):
        assert cli._run_flags(_run_args(cloudxr_install_dir="/opt/x")) == [
            "--cloudxr-install-dir",
            "/opt/x",
        ]
        default = os.path.expanduser("~/.cloudxr")
        assert cli._run_flags(_run_args(cloudxr_install_dir=default)) == []

    def test_accept_eula_is_never_baked_into_the_unit(self):
        """Acceptance is a marker file written at install time, not a unit flag."""
        assert "--accept-eula" not in cli._run_flags(_run_args(accept_eula=True))


class TestOobConsoleReporter:
    def test_absent_headset_is_visible_but_rate_limited(self):
        now = [0.0]
        reporter = cli._OobConsoleReporter(clock=lambda: now[0])
        waiting = {
            "health": "degraded",
            "state": "WAITING_FOR_ADB",
            "reason": "Waiting for selected headset",
            "selectedSerial": None,
            "adbReady": False,
        }
        assert reporter.observe({**waiting, "health": "starting"}) == []
        assert reporter.observe(waiting) == [
            "[setup-oob] No USB-attached HMD detected; connect a headset to continue."
        ]
        now[0] = 5
        assert reporter.observe(waiting) == []
        now[0] = 61
        assert reporter.observe(waiting) == [
            "[setup-oob] No USB-attached HMD detected; connect a headset to continue."
        ]

    def test_ready_loss_and_reconnection_are_reported(self):
        reporter = cli._OobConsoleReporter(clock=lambda: 0)
        ready = {
            "health": "degraded",
            "state": "PREPARING_DEVICE",
            "reason": "Preparing selected headset",
            "selectedSerial": "HMD-1",
            "adbReady": True,
        }
        first = reporter.observe(ready)
        assert any("HMD detected (serial HMD-1)" in line for line in first)
        assert reporter.observe(ready) == []
        lost = {**ready, "state": "WAITING_FOR_ADB", "adbReady": False}
        assert any("connection lost" in line for line in reporter.observe(lost))
        assert any(
            "HMD reconnected (serial HMD-1)" in line for line in reporter.observe(ready)
        )

    def test_phase_failure_and_health_transitions(self):
        reporter = cli._OobConsoleReporter(clock=lambda: 0)
        base = {"selectedSerial": "HMD-1", "adbReady": True, "health": "degraded"}
        assert any(
            "USB reverse rules" in line
            for line in reporter.observe({**base, "state": "REBUILDING_USB"})
        )
        assert reporter.observe({**base, "state": "REBUILDING_USB"}) == []
        failure = {**base, "state": "DEGRADED", "reason": "coturn is not listening"}
        assert reporter.observe(failure) == [
            "[setup-oob] OOB recovery delayed: coturn is not listening"
        ]
        assert reporter.observe(failure) == []
        browser = {**base, "health": "browser_ready", "state": "ACTIVE"}
        assert reporter.observe(browser) == [
            "[setup-oob] OOB browser ready; waiting for OpenXR stream."
        ]
        assert reporter.observe({**browser, "health": "active"}) == [
            "[setup-oob] OOB stream active with fresh client metrics."
        ]

    def test_quick_replug_reports_each_reentered_phase(self):
        reporter = cli._OobConsoleReporter(clock=lambda: 0)
        base = {"health": "degraded", "selectedSerial": "HMD-1", "adbReady": True}
        phases = (
            ("PREPARING_DEVICE", "Preparing headset HMD-1"),
            ("REBUILDING_USB", "Configuring USB reverse rules"),
            ("AUTOMATING_BROWSER", "Opening headset browser"),
            ("VERIFYING_BROWSER", "Waiting for headset browser"),
        )
        for state, expected in phases:
            snapshot = {**base, "state": state}
            assert any(expected in line for line in reporter.observe(snapshot))
            assert reporter.observe(snapshot) == []

        lost = {**base, "state": "WAITING_FOR_ADB", "adbReady": False}
        assert any("connection lost" in line for line in reporter.observe(lost))
        for state, expected in phases:
            snapshot = {**base, "state": state}
            assert any(expected in line for line in reporter.observe(snapshot))
            assert reporter.observe(snapshot) == []

        same_stage_new_generation = {
            **base,
            "state": "VERIFYING_BROWSER",
            "generation": 2,
        }
        assert any(
            "Waiting for headset browser" in line
            for line in reporter.observe(same_stage_new_generation)
        )

    def test_foreground_run_prints_queued_lifecycle_after_banner(
        self, tmp_path, capsys
    ):
        service = MagicMock()
        service.__enter__.return_value = service
        service.drain_oob_updates.side_effect = [
            [
                {"health": "starting", "state": "WAITING_FOR_ADB"},
                {
                    "health": "degraded",
                    "state": "WAITING_FOR_ADB",
                    "selectedSerial": None,
                    "adbReady": False,
                },
            ],
            [],
        ]
        args = _run_args(
            setup_oob=True, usb_local=True, cloudxr_install_dir=str(tmp_path)
        )
        with (
            patch.object(cli, "_oob_preflight", return_value=None),
            patch.object(cli, "CloudXRService", return_value=service),
            patch.object(cli, "_print_service_summary") as summary,
            patch.object(cli.time, "sleep", side_effect=KeyboardInterrupt),
        ):
            with pytest.raises(KeyboardInterrupt):
                cli._cmd_run(args)
        summary.assert_called_once()
        assert capsys.readouterr().out.count("No USB-attached HMD detected") == 1


class TestStartEula:
    """The EULA gate on start."""

    def _start_args(self, tmp_path, accept: bool):
        parser = cli._build_parser()
        return parser.parse_args(
            ["start", "--cloudxr-install-dir", str(tmp_path)]
            + (["--accept-eula"] if accept else [])
        )

    def test_refuses_when_not_accepted(self, tmp_path, capsys):
        """A detached service has no terminal, so it cannot be prompted later."""
        with patch("isaaccapture.cloudxr.runtime.is_runtime_live", return_value=False):
            with pytest.raises(SystemExit) as exc:
                cli._cmd_start(self._start_args(tmp_path, accept=False))

        assert exc.value.code == 1
        assert "EULA has not been accepted" in capsys.readouterr().err

    def test_accept_writes_the_marker(self, tmp_path):
        with (
            # Probed three times: the pre-check, the re-check under the start
            # lock, then the wait loop that sees the runtime come up.
            patch(
                "isaaccapture.cloudxr.runtime.is_runtime_live",
                side_effect=[False, False, True],
            ),
            patch("isaaccapture.cloudxr.background.spawn") as m_spawn,
        ):
            m_spawn.return_value = (4242, tmp_path / "logs" / "service.log")
            rc = cli._cmd_start(self._start_args(tmp_path, accept=True))

        assert rc == 0
        assert (tmp_path / "run" / "eula_accepted").is_file()
        m_spawn.assert_called_once()


class TestStart:
    """Tests for `service start`."""

    def _args(self, tmp_path):
        return cli._build_parser().parse_args(
            ["start", "--cloudxr-install-dir", str(tmp_path), "--accept-eula"]
        )

    def test_refuses_when_one_is_already_running(self, tmp_path, capsys):
        with patch("isaaccapture.cloudxr.runtime.is_runtime_live", return_value=True):
            with pytest.raises(SystemExit):
                cli._cmd_start(self._args(tmp_path))
        assert "already serving" in capsys.readouterr().err

    def test_reports_when_the_child_dies_during_startup(self, tmp_path, capsys):
        """A crash on startup must not look like a slow start."""
        with (
            patch("isaaccapture.cloudxr.runtime.is_runtime_live", return_value=False),
            patch("isaaccapture.cloudxr.background.spawn") as m_spawn,
            patch("isaaccapture.cloudxr.background.read_pid", return_value=None),
        ):
            m_spawn.return_value = (4242, tmp_path / "service.log")
            (tmp_path / "run").mkdir(parents=True)
            (tmp_path / "run" / "eula_accepted").write_text("accepted\n")
            with pytest.raises(SystemExit):
                cli._cmd_start(self._args(tmp_path))

        assert "exited during startup" in capsys.readouterr().err


class TestStopAndStatus:
    """Tests for `service stop` and `service status`."""

    def _args(self, command, tmp_path):
        return cli._build_parser().parse_args(
            [command, "--cloudxr-install-dir", str(tmp_path)]
        )

    def test_stop_is_a_noop_when_nothing_runs(self, tmp_path, capsys):
        with patch("isaaccapture.cloudxr.background.read_pid", return_value=None):
            assert cli._cmd_stop(self._args("stop", tmp_path)) == 0
        assert "No detached CloudXR service" in capsys.readouterr().out

    def test_stop_reports_a_wedged_service(self, tmp_path, capsys):
        with (
            patch("isaaccapture.cloudxr.background.read_pid", return_value=42),
            patch("isaaccapture.cloudxr.background.terminate", return_value=False),
        ):
            with pytest.raises(SystemExit):
                cli._cmd_stop(self._args("stop", tmp_path))
        assert "would orphan the runtime" in capsys.readouterr().err

    def test_status_exit_code_tracks_liveness(self, tmp_path):
        with (
            patch("isaaccapture.cloudxr.runtime.is_runtime_live", return_value=True),
            patch("isaaccapture.cloudxr.background.read_pid", return_value=7),
        ):
            assert cli._cmd_status(self._args("status", tmp_path)) == 0
        with (
            patch("isaaccapture.cloudxr.runtime.is_runtime_live", return_value=False),
            patch("isaaccapture.cloudxr.background.read_pid", return_value=None),
        ):
            assert cli._cmd_status(self._args("status", tmp_path)) == 1

    def test_status_distinguishes_foreground_from_detached(self, tmp_path, capsys):
        """A runtime with no pid file was started by hand or another supervisor."""
        with (
            patch("isaaccapture.cloudxr.runtime.is_runtime_live", return_value=True),
            patch("isaaccapture.cloudxr.background.read_pid", return_value=None),
        ):
            cli._cmd_status(self._args("status", tmp_path))
        assert "foreground" in capsys.readouterr().out

    def test_status_reports_the_running_session_not_our_defaults(
        self, tmp_path, capsys
    ):
        """Flags come from the running service's own command line."""
        with (
            patch("isaaccapture.cloudxr.runtime.is_runtime_live", return_value=True),
            patch("isaaccapture.cloudxr.background.read_pid", return_value=7),
            patch(
                "isaaccapture.cloudxr.background.read_run_flags",
                return_value=["--host-client"],
            ),
        ):
            cli._cmd_status(self._args("status", tmp_path))
        # --host-client means the client is served off the WSS proxy, not Pages.
        assert "/client/" in capsys.readouterr().out


class TestRunValidation:
    """Flag combinations rejected before anything starts."""

    def test_usb_local_requires_setup_oob(self, capsys):
        with pytest.raises(SystemExit):
            cli._cmd_run(_run_args(usb_local=True, setup_oob=False))
        assert "--usb-local requires --setup-oob" in capsys.readouterr().err

    def test_usb_local_rejects_hub_only(self, monkeypatch, capsys):
        monkeypatch.setenv("TELEOP_OOB_HUB_ONLY", "1")
        with pytest.raises(SystemExit):
            cli._cmd_run(_run_args(usb_local=True, setup_oob=True))
        assert "not compatible with --usb-local" in capsys.readouterr().err


class TestParser:
    """Dispatch-level behaviour."""

    def test_bare_invocation_prints_help(self, capsys):
        assert cli.main([]) == 0
        assert "COMMAND" in capsys.readouterr().out

    def test_commands_are_registered(self):
        parser = cli._build_parser()
        for command in ("run", "start", "stop", "status", "logs"):
            assert parser.parse_args([command]).func is not None


class TestOutputChannels:
    """`run` writes to a terminal or to service.log; they want different text."""

    def test_terminal_instructions_stay_out_of_the_log(self, capsys):
        """A detached service has no terminal and nobody to press Ctrl+C."""
        with patch("sys.stdout.isatty", return_value=False):
            cli._out_interactive("Keep this terminal open, Ctrl+C to terminate.")
        assert capsys.readouterr().out == ""

    def test_terminal_instructions_are_printed_on_a_terminal(self, capsys):
        with patch("sys.stdout.isatty", return_value=True):
            cli._out_interactive("Keep this terminal open, Ctrl+C to terminate.")
        assert "Ctrl+C" in capsys.readouterr().out

    def test_colour_is_dropped_when_output_is_redirected(self, capsys):
        with patch("sys.stdout.isatty", return_value=False):
            cli._out("\033[33mplain\033[0m")
        assert capsys.readouterr().out == "plain\n"

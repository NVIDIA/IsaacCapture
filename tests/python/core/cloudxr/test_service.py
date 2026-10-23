# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for isaaccapture.cloudxr.service — CloudXRService lifecycle."""

import asyncio
import contextlib
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conftest import live_ipc_socket, mock_service_deps
from isaaccapture.cloudxr.service import CloudXRService
from isaaccapture.logging_config._native_api import (
    native_capture_fd,
    native_capture_path,
)

_posix_only = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Process-group APIs (os.getpgid/os.killpg) are POSIX-only",
)

_windows_skip = pytest.mark.skipif(
    sys.platform == "win32",
    reason="CloudXR runtime process termination is not supported on Windows",
)


# ============================================================================
# TestServiceConstruction
# ============================================================================


class TestServiceConstruction:
    """Tests for CloudXRService construction (which starts the runtime)."""

    def test_embedded_oob_preflight_fails_before_runtime_spawn(self, tmp_path):
        """Embedded startup validates host ADB without requiring a headset."""
        with (
            mock_service_deps(tmp_path, ready=True) as mocks,
            patch(
                "isaaccapture.cloudxr.oob_teleop_adb.require_adb_on_path",
                side_effect=RuntimeError("adb missing"),
            ),
        ):
            with pytest.raises(RuntimeError, match="adb missing"):
                CloudXRService(install_dir=str(tmp_path), setup_oob=True)
            mocks["popen"].assert_not_called()

    def test_embedded_usb_preflight_checks_coturn_before_runtime_spawn(self, tmp_path):
        """A missing TURN binary is a host setup failure, not a retry episode."""
        with (
            mock_service_deps(tmp_path, ready=True) as mocks,
            patch("isaaccapture.cloudxr.oob_teleop_env.require_web_client_static_dir"),
            patch("isaaccapture.cloudxr.oob_teleop_adb.require_adb_on_path"),
            patch(
                "isaaccapture.cloudxr.oob_teleop_adb.require_coturn_available",
                side_effect=RuntimeError("coturn missing"),
            ),
        ):
            with pytest.raises(RuntimeError, match="coturn missing"):
                CloudXRService(
                    install_dir=str(tmp_path), setup_oob=True, usb_local=True
                )
            mocks["popen"].assert_not_called()

    def test_embedded_usb_port_conflict_fails_before_runtime_spawn(self, tmp_path):
        with (
            mock_service_deps(tmp_path, ready=True) as mocks,
            patch("isaaccapture.cloudxr.oob_teleop_env.require_web_client_static_dir"),
            patch("isaaccapture.cloudxr.oob_teleop_adb.require_adb_on_path"),
            patch("isaaccapture.cloudxr.oob_teleop_adb.require_coturn_available"),
            patch("isaaccapture.cloudxr.oob_teleop_adb.require_turn_port_free"),
            patch(
                "isaaccapture.cloudxr.oob_teleop_env.print_host_preflight_warnings",
                side_effect=RuntimeError("required ports occupied"),
            ) as ports,
        ):
            with pytest.raises(RuntimeError, match="required ports occupied"):
                CloudXRService(
                    install_dir=str(tmp_path), setup_oob=True, usb_local=True
                )
            ports.assert_called_once_with(usb_local=True, emit_console=False)
            mocks["popen"].assert_not_called()

    def test_oob_rejects_incompatible_webxr_bundle_before_runtime_spawn(self, tmp_path):
        with (
            mock_service_deps(tmp_path, ready=True) as mocks,
            patch(
                "isaaccapture.cloudxr.oob_teleop_env.require_web_client_static_dir",
                side_effect=RuntimeError("lacks the OOB healthProbe"),
            ) as require_static,
        ):
            with pytest.raises(RuntimeError, match="lacks the OOB healthProbe"):
                CloudXRService(
                    install_dir=str(tmp_path), setup_oob=True, usb_local=True
                )
            require_static.assert_called_once_with(
                require_health_probe=True, require_fragment_token=True
            )
            mocks["popen"].assert_not_called()

    def test_construction_stores_parameters(self, tmp_path):
        """Constructor stores install_dir, env_config, device_profile, and accept_eula."""
        with mock_service_deps(tmp_path, ready=True):
            service = CloudXRService(
                install_dir="/opt/cloudxr",
                env_config="/etc/cloudxr.env",
                device_profile="AppleVisionPro",
                accept_eula=True,
            )
        assert service._install_dir == "/opt/cloudxr"
        assert service._env_config == "/etc/cloudxr.env"
        assert service._device_profile == "AppleVisionPro"
        assert service._accept_eula is True

    def test_construction_passes_device_profile_to_env_config(self, tmp_path):
        """Constructor forwards device_profile to EnvConfig.from_args."""
        with mock_service_deps(tmp_path, ready=True) as mocks:
            CloudXRService(device_profile="auto-native")

            mocks["from_args"].assert_called_once_with(
                "~/.cloudxr",
                None,
                launcher_defaults={"NV_DEVICE_PROFILE": "auto-native"},
            )

    def test_construction_launches_runtime_and_wss(self, tmp_path):
        """Successful construction calls Popen and WSS proxy."""
        with mock_service_deps(tmp_path, ready=True) as mocks:
            CloudXRService()

            mocks["popen"].assert_called_once()
            mocks["wss"].assert_called_once()
            mocks["check_eula"].assert_called_once()
            mocks["cleanup"].assert_called_once()

    @_windows_skip
    def test_construction_raises_on_runtime_failure(self, tmp_path):
        """RuntimeError when the runtime fails to become ready."""
        with mock_service_deps(tmp_path, ready=False) as mocks:
            mocks["proc"].poll.return_value = 1

            with pytest.raises(RuntimeError, match="failed to start"):
                CloudXRService()

    @_windows_skip
    def test_runtime_stderr_goes_to_a_file_not_a_pipe(self, tmp_path):
        """Nothing drains a pipe while the runtime runs, so it would fill and block."""
        with mock_service_deps(tmp_path, ready=True) as mocks:
            CloudXRService()

        stderr = mocks["popen"].call_args.kwargs["stderr"]
        assert stderr is not subprocess.PIPE
        assert Path(stderr.name) == tmp_path / "logs" / "runtime_worker_stderr.log"

    @_windows_skip
    def test_startup_failure_reports_the_worker_stderr(self, tmp_path):
        """A file the caller is never told about is no better than a full pipe."""

        def _write_what_the_worker_would_have(*_args, **kwargs):
            Path(kwargs["stderr"].name).write_text("ImportError: no runtime\n")
            return mocks["proc"]

        with mock_service_deps(tmp_path, ready=False) as mocks:
            mocks["proc"].poll.return_value = 1
            mocks["popen"].side_effect = _write_what_the_worker_would_have

            with pytest.raises(RuntimeError) as exc:
                CloudXRService()

        detail = str(exc.value)
        assert "runtime_worker_stderr.log" in detail
        assert "ImportError: no runtime" in detail

    @_windows_skip
    def test_the_runtime_workers_stdout_goes_to_the_session_capture_file(
        self, tmp_path
    ):
        """The runtime's banner is isaaccapture's output, not the host terminal's.

        A process this library launched is not the host, so pointing its
        descriptors at the capture file is exactly what the design relies on.
        """
        capture_fd = native_capture_fd()
        if capture_fd is None:
            pytest.skip("no capture file in this environment")

        with mock_service_deps(tmp_path, ready=True) as mocks:
            CloudXRService()

        assert mocks["popen"].call_args.kwargs["stdout"] == capture_fd

    @_windows_skip
    def test_startup_failure_reports_the_native_capture_file(self, tmp_path):
        """The worker's own fd 1 lands there, not under logs_dir."""
        capture_fd = native_capture_fd()
        capture = native_capture_path()
        if capture_fd is None or capture is None:
            pytest.skip("no capture file in this environment")
        os.write(capture_fd, b"cxr_server: libcloudxr.so not found\n")

        with mock_service_deps(tmp_path, ready=False) as mocks:
            mocks["proc"].poll.return_value = 1

            with pytest.raises(RuntimeError) as exc:
                CloudXRService()

        detail = str(exc.value)
        assert capture.name in detail
        assert "libcloudxr.so not found" in detail

    def test_wss_log_path_set_after_construction(self, tmp_path):
        """wss_log_path is a Path after successful construction."""
        with mock_service_deps(tmp_path, ready=True):
            service = CloudXRService()

            assert service.wss_log_path is not None
            assert isinstance(service.wss_log_path, Path)
            assert "wss." in str(service.wss_log_path)


# ============================================================================
# TestServiceStop
# ============================================================================


@_windows_skip
class TestServiceStop:
    """Tests for CloudXRService.stop()."""

    @_posix_only
    def test_stop_terminates_runtime(self, tmp_path):
        """stop() sends SIGTERM to the runtime process group."""
        with mock_service_deps(tmp_path, ready=True) as mocks:
            service = CloudXRService()

            proc = mocks["proc"]
            poll_seq = [None, 0]
            proc.poll = MagicMock(
                side_effect=lambda: poll_seq.pop(0) if poll_seq else 0
            )
            proc.wait = MagicMock()

            with (
                patch(
                    "isaaccapture.cloudxr.service._service.os.getpgid", return_value=99
                ) as m_getpgid,
                patch("isaaccapture.cloudxr.service._service.os.killpg") as m_killpg,
            ):
                service.stop()

                m_getpgid.assert_called_once_with(proc.pid)
                m_killpg.assert_called_once_with(99, signal.SIGTERM)

    def test_stop_idempotent(self, tmp_path):
        """Calling stop() twice does not raise."""
        with mock_service_deps(tmp_path, ready=True) as mocks:
            service = CloudXRService()

            mocks["proc"].poll.return_value = 0

            service.stop()
            service.stop()

    @_posix_only
    def test_stop_escalates_to_sigkill(self, tmp_path):
        """stop() sends SIGKILL when SIGTERM doesn't work."""
        with mock_service_deps(tmp_path, ready=True) as mocks:
            service = CloudXRService()

            proc = mocks["proc"]
            poll_seq = [None, None, 0]
            proc.poll = MagicMock(
                side_effect=lambda: poll_seq.pop(0) if poll_seq else 0
            )
            proc.wait = MagicMock(side_effect=subprocess.TimeoutExpired("cmd", 10))

            with (
                patch(
                    "isaaccapture.cloudxr.service._service.os.getpgid", return_value=99
                ),
                patch("isaaccapture.cloudxr.service._service.os.killpg") as m_killpg,
            ):
                service.stop()

                calls = m_killpg.call_args_list
                assert len(calls) == 2
                assert calls[0].args == (99, signal.SIGTERM)
                assert calls[1].args == (99, signal.SIGKILL)

    def test_stop_on_windows_raises_unsupported(self, tmp_path) -> None:
        """Simulated win32 platform raises instead of calling POSIX APIs."""
        with mock_service_deps(tmp_path, ready=True) as mocks:
            service = CloudXRService()
            mocks["proc"].poll.return_value = None

            with patch("isaaccapture.cloudxr.service._service.sys.platform", "win32"):
                with pytest.raises(RuntimeError, match="not supported on Windows"):
                    service.stop()


# ============================================================================
# TestServiceContextManager
# ============================================================================


@_windows_skip
class TestServiceContextManager:
    """Tests for CloudXRService used as a context manager."""

    def test_context_manager_stops_on_exit(self, tmp_path):
        """__exit__ calls stop(), cleaning up the runtime."""
        with mock_service_deps(tmp_path, ready=True) as mocks:
            with CloudXRService() as service:
                mocks["popen"].assert_called_once()
                mocks["proc"].poll.return_value = 0

            assert service._runtime_proc is None


# ============================================================================
# TestCleanupStaleRuntime
# ============================================================================


@_posix_only
class TestCleanupStaleRuntime:
    """Tests for CloudXRService._cleanup_stale_runtime."""

    @staticmethod
    def _stale_run_dir(tmp_path) -> tuple[str, list[str]]:
        """Create a run dir holding every sentinel file; return it and their paths."""
        run_dir = str(tmp_path / "run")
        os.makedirs(run_dir)
        paths = [
            os.path.join(run_dir, name)
            for name in ("ipc_cloudxr", "runtime_started", "cloudxr.pid")
        ]
        # A stale IPC endpoint is a closed Unix socket, not a regular file.
        cwd = os.getcwd()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            os.chdir(run_dir)
            sock.bind("ipc_cloudxr")
        finally:
            os.chdir(cwd)
            sock.close()
        for path in paths[1:]:
            Path(path).touch()
        return run_dir, paths

    def test_removes_stale_sentinel_files(self, tmp_path, caplog):
        """A socket file nobody is serving is stale: removed, at WARNING."""
        run_dir, paths = self._stale_run_dir(tmp_path)

        with caplog.at_level(
            logging.WARNING, logger="isaaccapture.cloudxr.service._service"
        ):
            CloudXRService._cleanup_stale_runtime(run_dir)

        assert not any(os.path.exists(p) for p in paths)
        assert len(caplog.records) == len(paths)

    def test_noop_when_no_stale_files(self, tmp_path):
        """No errors when the run directory has no stale files."""
        run_dir = str(tmp_path / "run")
        os.makedirs(run_dir)

        CloudXRService._cleanup_stale_runtime(run_dir)

    def test_refuses_when_runtime_is_live(self, tmp_path):
        """A served socket is a live runtime: refuse, and keep its files."""
        run_dir, paths = self._stale_run_dir(tmp_path)

        with live_ipc_socket(run_dir):
            with pytest.raises(RuntimeError, match="already serving") as exc_info:
                CloudXRService._cleanup_stale_runtime(run_dir)
            assert all(os.path.exists(p) for p in paths)

        message = str(exc_info.value)
        assert os.path.join(run_dir, "cloudxr.env") in message
        assert "isaaccapture.cloudxr.service stop" in message


class TestRefusalLeavesTheLiveRuntimeAlone:
    """A refused start must not disturb the runtime it refused to replace."""

    def test_the_live_runtimes_env_file_is_not_rewritten(self, tmp_path):
        """EnvConfig resolution truncates cloudxr.env, so it must run after."""
        from isaaccapture.cloudxr.service import CloudXRService

        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True)
        env_file = run_dir / "cloudxr.env"
        env_file.write_text("export NV_DEVICE_PROFILE=Quest3\n", encoding="utf-8")

        with live_ipc_socket(str(run_dir)):
            with pytest.raises(RuntimeError, match="already serving"):
                CloudXRService(install_dir=str(tmp_path))

        assert env_file.read_text() == "export NV_DEVICE_PROFILE=Quest3\n"


# ============================================================================
# TestWssProxyStartup
# ============================================================================


@contextlib.contextmanager
def _stub_wss(run):
    """Stand in for isaaccapture.cloudxr.wss so no real proxy is started."""
    name = "isaaccapture.cloudxr.wss"
    module = types.ModuleType(name)
    module.run = run
    with patch.dict(sys.modules, {name: module}):
        yield


async def _never_listens(*, stop_future, **_kwargs):
    """A proxy that runs forever without ever binding."""
    await stop_future


async def _fails_to_bind(**_kwargs):
    """A proxy whose port is taken."""
    raise OSError("[Errno 98] Address already in use")


class TestWssProxyStartup:
    """Construction reports a proxy that is listening, not one that was launched."""

    def test_construction_waits_for_the_proxy_to_listen(self, tmp_path):
        """A proxy still binding is not one the caller can be handed."""
        bound = []

        async def _listens_after_a_beat(*, stop_future, on_listening, **_kwargs):
            await asyncio.sleep(0.2)
            bound.append(True)
            on_listening()
            await stop_future

        with mock_service_deps(tmp_path, wss=False), _stub_wss(_listens_after_a_beat):
            service = CloudXRService(install_dir=str(tmp_path))

            assert bound == [True]
            assert service._wss_thread.is_alive()
            service._stop_wss_proxy()

    @_posix_only
    def test_a_proxy_that_cannot_bind_fails_the_service(self, tmp_path):
        """Reporting success here would defer the failure to a later health_check."""
        with (
            mock_service_deps(tmp_path, wss=False) as mocks,
            _stub_wss(_fails_to_bind),
            patch("isaaccapture.cloudxr.service._service.os.getpgid", return_value=99),
            patch("isaaccapture.cloudxr.service._service.os.killpg") as m_killpg,
        ):
            poll_seq = [None, 0]
            mocks["proc"].poll = MagicMock(
                side_effect=lambda: poll_seq.pop(0) if poll_seq else 0
            )

            with pytest.raises(RuntimeError, match="Address already in use"):
                CloudXRService(install_dir=str(tmp_path))

            # The runtime came up before the proxy, so a failed start must not
            # leave it holding the run directory.
            m_killpg.assert_called_once_with(99, signal.SIGTERM)

    def test_a_proxy_that_never_listens_times_out(self, tmp_path):
        with mock_service_deps(tmp_path):
            service = CloudXRService(install_dir=str(tmp_path))

        with _stub_wss(_never_listens):
            with pytest.raises(RuntimeError, match="did not start listening"):
                service._start_wss_proxy_thread(tmp_path / "wss.log", timeout_sec=0.2)

        # The thread outlives the timeout; stop() is what reaps it.
        service._stop_wss_proxy()

    @pytest.mark.parametrize("defer_oob_connect", [False, True])
    def test_headset_connection_is_explicit_only_when_deferred(
        self, tmp_path, defer_oob_connect
    ):
        from cloudxr_py_test_ns.wss import _wait_for_connect

        connected = threading.Event()

        async def proxy(*, stop_future, on_listening, connect_requested, **kwargs):
            on_listening()
            if await _wait_for_connect(connect_requested, stop_future):
                connected.set()
            await stop_future

        with (
            mock_service_deps(tmp_path, wss=False),
            _stub_wss(proxy),
            patch("isaaccapture.cloudxr.oob_teleop_adb.require_adb_on_path"),
        ):
            service = CloudXRService(
                setup_oob=True, defer_oob_connect=defer_oob_connect
            )
            try:
                if defer_oob_connect:
                    assert not connected.wait(0.05)
                    service.connect_headset()
                    service.connect_headset()  # Releasing the same gate is harmless.
                assert connected.wait(1)
            finally:
                service._stop_wss_proxy()

    @pytest.mark.parametrize("configured", ["operator-chosen", None])
    def test_control_token_is_the_one_the_hub_enforces(
        self, tmp_path, monkeypatch, configured
    ):
        from isaaccapture.cloudxr.oob_teleop_hub import OOBControlHub

        hubs = []

        async def proxy(*, stop_future, on_listening, control_token, **kwargs):
            hubs.append(OOBControlHub(control_token=control_token))
            on_listening()
            await stop_future

        if configured:
            monkeypatch.setenv("CONTROL_TOKEN", configured)
        else:
            monkeypatch.delenv("CONTROL_TOKEN", raising=False)
        with (
            mock_service_deps(tmp_path, wss=False),
            _stub_wss(proxy),
            patch("isaaccapture.cloudxr.oob_teleop_adb.require_adb_on_path"),
        ):
            service = CloudXRService(setup_oob=True)
            try:
                token = service.control_token
                assert token and hubs[0].check_token(token)
                if configured:
                    assert token == configured
                # The hub keeps its start-up token whatever the environment does.
                monkeypatch.setenv("CONTROL_TOKEN", "changed-later")
                assert service.control_token == token
                monkeypatch.delenv("CONTROL_TOKEN")
                assert service.control_token == token
                assert hubs[0].check_token(token)
            finally:
                service._stop_wss_proxy()

    def test_control_token_is_none_without_oob(self, tmp_path):
        with mock_service_deps(tmp_path):
            service = CloudXRService()
            assert service.control_token is None

    def test_stopping_before_connect_does_not_launch_headset(self, tmp_path):
        from cloudxr_py_test_ns.wss import _wait_for_connect

        connected = []

        async def proxy(*, stop_future, on_listening, connect_requested, **kwargs):
            on_listening()
            if await _wait_for_connect(connect_requested, stop_future):
                connected.append(True)

        with (
            mock_service_deps(tmp_path, wss=False),
            _stub_wss(proxy),
            patch("isaaccapture.cloudxr.oob_teleop_adb.require_adb_on_path"),
        ):
            service = CloudXRService(setup_oob=True, defer_oob_connect=True)
            thread = service._wss_thread
            service._stop_wss_proxy()
            assert not thread.is_alive()
            assert connected == []
            with pytest.raises(RuntimeError, match="not running"):
                service.connect_headset()


# ============================================================================
# TestSignalHandlers
# ============================================================================


class TestSignalHandlers:
    """Signal handlers must propagate first so with-block teardown runs in order.

    The key invariant: the handler must NOT call stop() directly.  Instead it
    propagates to the previous handler (e.g. raising KeyboardInterrupt for
    SIGINT), so that inner context managers (like an OpenXR session) can clean
    up before the outer CloudXRService terminates the runtime process.
    stop() is reached via __exit__ and atexit.
    """

    def test_sigint_handler_raises_keyboard_interrupt_not_stop(self, tmp_path):
        """SIGINT handler raises KeyboardInterrupt via prev; does not call stop()."""
        with mock_service_deps(tmp_path, ready=True):
            service = CloudXRService()

        stop_called = []
        service._original_stop = service.stop
        service.stop = lambda: stop_called.append(True)

        handler = signal.getsignal(signal.SIGINT)
        try:
            with pytest.raises(KeyboardInterrupt):
                handler(signal.SIGINT, None)
        finally:
            service.stop = service._original_stop
            service.stop()

        assert not stop_called, "signal handler must not call stop() directly"

    def test_sigterm_with_sig_dfl_prev_raises_system_exit(self, tmp_path):
        """SIGTERM handler raises SystemExit when prev was SIG_DFL."""
        orig_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        try:
            with mock_service_deps(tmp_path, ready=True):
                service = CloudXRService()

            stop_called = []
            service._original_stop = service.stop
            service.stop = lambda: stop_called.append(True)

            handler = signal.getsignal(signal.SIGTERM)
            try:
                with pytest.raises(SystemExit) as exc_info:
                    handler(signal.SIGTERM, None)
                assert exc_info.value.code == 0
            finally:
                service.stop = service._original_stop
                service.stop()
        finally:
            signal.signal(signal.SIGTERM, orig_sigterm)

        assert not stop_called, "signal handler must not call stop() directly"

    def test_sigint_handler_with_callable_prev_calls_prev(self, tmp_path):
        """SIGINT handler with a custom callable prev calls it instead of stop()."""
        prev_called = []

        def custom_prev(signum, frame):
            prev_called.append(signum)

        orig_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, custom_prev)
        try:
            with mock_service_deps(tmp_path, ready=True):
                service = CloudXRService()

            stop_called = []
            service._original_stop = service.stop
            service.stop = lambda: stop_called.append(True)

            handler = signal.getsignal(signal.SIGINT)
            try:
                handler(signal.SIGINT, None)
            finally:
                service.stop = service._original_stop
                service.stop()
        finally:
            signal.signal(signal.SIGINT, orig_sigint)

        assert prev_called == [signal.SIGINT]
        assert not stop_called, "signal handler must not call stop() directly"

    def test_sigint_handler_with_sig_ign_prev_is_noop(self, tmp_path):
        """SIGINT handler is a no-op when prev was SIG_IGN."""
        orig_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            with mock_service_deps(tmp_path, ready=True):
                service = CloudXRService()

            stop_called = []
            service._original_stop = service.stop
            service.stop = lambda: stop_called.append(True)

            handler = signal.getsignal(signal.SIGINT)
            try:
                handler(signal.SIGINT, None)  # must not raise
            finally:
                service.stop = service._original_stop
                service.stop()
        finally:
            signal.signal(signal.SIGINT, orig_sigint)

        assert not stop_called, "signal handler must not call stop() directly"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", ["connect", "shutdown", "disconnect", "disconnect_and_connect", "cancel"]
)
async def test_early_websocket_waits_for_app_or_disconnect(monkeypatch, outcome):
    from unittest.mock import AsyncMock
    from cloudxr_py_test_ns import wss

    requested = asyncio.Event()
    listening = asyncio.Event()
    closed = asyncio.Event()
    watching_close = asyncio.Event()
    close_wait_finished = asyncio.Event()
    stop = asyncio.get_running_loop().create_future()
    handlers = []

    async def wait_closed():
        watching_close.set()
        try:
            await closed.wait()
        finally:
            close_wait_finished.set()

    @contextlib.asynccontextmanager
    async def serve(handler, **kwargs):
        handlers.append(handler)
        yield

    proxy = AsyncMock()
    # Keep this signaling test independent of an installed CloudXR environment.
    monkeypatch.setattr(wss, "default_cert_paths", lambda: None)
    monkeypatch.setattr(wss, "ensure_certificate", lambda *args: None)
    monkeypatch.setattr(wss, "build_ssl_context", lambda *args: None)
    monkeypatch.setattr(wss, "ws_serve", serve)
    monkeypatch.setattr(wss, "proxy_handler", proxy)
    task = asyncio.create_task(
        wss.run(
            None,
            stop,
            on_listening=listening.set,
            connect_requested=requested,
        )
    )
    client = None
    try:
        await asyncio.wait_for(listening.wait(), timeout=1)
        client = asyncio.create_task(
            handlers[0](types.SimpleNamespace(wait_closed=wait_closed))
        )
        await asyncio.wait_for(watching_close.wait(), timeout=1)
        proxy.assert_not_called()
        if outcome in ("disconnect", "disconnect_and_connect"):
            closed.set()
        if outcome in ("connect", "disconnect_and_connect"):
            requested.set()
        if outcome == "shutdown":
            stop.set_result(None)
        if outcome == "cancel":
            client.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(client, timeout=1)
        else:
            await asyncio.wait_for(client, timeout=1)
        assert proxy.call_count == int(outcome == "connect")
        assert close_wait_finished.is_set()
        assert stop.done() == (outcome == "shutdown")
    finally:
        if client is not None:
            client.cancel()
            await asyncio.gather(client, return_exceptions=True)
        if not stop.done():
            stop.set_result(None)
        await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_disconnected_websockets_do_not_forward_stale_offers(monkeypatch):
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve
    from cloudxr_py_test_ns import wss

    requested = asyncio.Event()
    listening = asyncio.Event()
    stop = asyncio.get_running_loop().create_future()
    finished = asyncio.Queue()
    connections = []
    offers = []
    proxy_port = None

    async def backend_handler(ws):
        connections.append(ws.request.path)
        async for offer in ws:
            offers.append(offer)
            await ws.send(offer)

    @contextlib.asynccontextmanager
    async def serve_proxy(handler, **kwargs):
        nonlocal proxy_port

        async def tracked_handler(ws):
            try:
                await handler(ws)
            finally:
                finished.put_nowait(ws.request.path)

        kwargs.update(host="127.0.0.1", port=0)
        async with serve(tracked_handler, **kwargs) as server:
            proxy_port = server.sockets[0].getsockname()[1]
            yield

    monkeypatch.setattr(wss, "default_cert_paths", lambda: None)
    monkeypatch.setattr(wss, "ensure_certificate", lambda *args: None)
    monkeypatch.setattr(wss, "build_ssl_context", lambda *args: None)
    monkeypatch.setattr(wss, "ws_serve", serve_proxy)

    async with serve(backend_handler, "127.0.0.1", 0) as backend:
        task = asyncio.create_task(
            wss.run(
                None,
                stop,
                backend_host="127.0.0.1",
                backend_port=backend.sockets[0].getsockname()[1],
                proxy_port=0,
                on_listening=listening.set,
                connect_requested=requested,
            )
        )
        try:
            await asyncio.wait_for(listening.wait(), timeout=2)
            uri = f"ws://127.0.0.1:{proxy_port}"
            for index in range(3):
                async with connect(f"{uri}/stale-{index}") as client:
                    await client.send(f"stale-offer-{index}")

            # Disconnected handlers must exit while the startup gate is still held.
            exited = {
                await asyncio.wait_for(finished.get(), timeout=2) for _ in range(3)
            }
            assert exited == {f"/stale-{index}" for index in range(3)}
            assert not connections
            assert not offers

            async with connect(f"{uri}/live") as client:
                await client.send("fresh-offer")
                requested.set()
                assert await asyncio.wait_for(client.recv(), timeout=2) == "fresh-offer"
            assert await asyncio.wait_for(finished.get(), timeout=2) == "/live"
        finally:
            if not stop.done():
                stop.set_result(None)
            await asyncio.wait_for(task, timeout=2)

    assert connections == ["/live"]
    assert offers == ["fresh-offer"]

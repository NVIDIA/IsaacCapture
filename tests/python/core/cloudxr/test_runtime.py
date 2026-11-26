# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for isaaccapture.cloudxr.runtime — wait_for_runtime_ready_sync and
terminate_or_kill_runtime."""

import importlib.util
import os
import socket
import threading
import time
from types import SimpleNamespace

import pytest
from unittest.mock import MagicMock, patch

from isaaccapture.cloudxr import runtime
from isaaccapture.cloudxr.runtime import (
    _is_exp_available,
    _should_join_main,
    _should_use_exp,
    get_sdk_path,
    is_runtime_live,
    resolve_cloudxr_runtime_module,
    terminate_or_kill_runtime,
    wait_for_runtime_ready_sync,
)


# ============================================================================
# Helpers
# ============================================================================


class _FakeEnvConfig:
    """Minimal stand-in for EnvConfig that redirects openxr_run_dir to a tmp path."""

    def __init__(self, run_dir: str) -> None:
        self._run_dir = run_dir

    def openxr_run_dir(self) -> str:
        return self._run_dir


def _patch_find_spec(monkeypatch, roots_by_name: dict[str, list[str]]) -> list[str]:
    """Serve fake ``__path__`` roots from ``find_spec``; record the names asked for.

    Patching ``find_spec`` mutates the process-wide ``importlib.util`` module,
    so unmapped names delegate to the real implementation rather than failing:
    any import machinery that runs inside the patch window must keep working.
    The returned list lets a test assert that only the bare *package* name was
    probed -- a dotted submodule would import the package as a side effect.
    """
    real_find_spec = importlib.util.find_spec
    calls: list[str] = []

    def _fake_find_spec(name: str, package: str | None = None):
        calls.append(name)
        if name in roots_by_name:
            return SimpleNamespace(submodule_search_locations=list(roots_by_name[name]))
        return real_find_spec(name, package)

    monkeypatch.setattr(
        "isaaccapture.cloudxr.runtime.importlib.util.find_spec", _fake_find_spec
    )
    return calls


def _make_package_roots(tmp_path, artifact_index: int | None) -> tuple[list[str], str]:
    """Build a two-entry ``__path__``, at most one root holding the runtime.

    Both roots get a ``native/`` directory; only ``roots[artifact_index]`` gets
    ``libcloudxr.so`` in it. An empty ``native/`` is exactly what a build leaves
    behind when the SDK tarball is missing, and the two-root shape is the only
    one that catches a regression to single-root (``__path__[0]``) resolution.

    Returns the roots and the expected ``native/`` path ("" when no root holds
    the artifact).
    """
    roots = []
    for index in range(2):
        native = tmp_path / f"root{index}" / "native"
        native.mkdir(parents=True)
        roots.append(str(native.parent))
    if artifact_index is None:
        return roots, ""
    native_dir = os.path.join(roots[artifact_index], "native")
    (tmp_path / f"root{artifact_index}" / "native" / "libcloudxr.so").write_bytes(b"")
    return roots, native_dir


# ============================================================================
# TestShouldUseExp / TestShouldJoinMain / resolve / get_sdk_path
# ============================================================================


class TestIsTegraT234:
    """Tests for Orin detection from sysfs and the device tree."""

    @pytest.fixture
    def paths(self, tmp_path, monkeypatch):
        """Point every signal at an empty tmp dir, i.e. a non-Jetson host."""
        soc_devices = tmp_path / "soc_devices"
        soc_devices.mkdir()
        monkeypatch.setattr(runtime, "_SOC_DEVICES", soc_devices)
        monkeypatch.setattr(runtime, "_DEVICE_TREE_COMPATIBLE", tmp_path / "compatible")
        return tmp_path

    @staticmethod
    def _add_soc(paths, name: str, family: str, soc_id: str) -> None:
        soc = paths / "soc_devices" / name
        soc.mkdir()
        (soc / "family").write_text(f"{family}\n")
        (soc / "soc_id").write_text(f"{soc_id}\n")

    def test_nothing_readable(self, paths):
        assert runtime._is_tegra_t234() is False

    def test_tegra_soc_device(self, paths):
        self._add_soc(paths, "soc0", "Tegra", "35")
        assert runtime._is_tegra_t234() is True

    def test_tegra_soc_device_not_first(self, paths):
        self._add_soc(paths, "soc0", "jep106:0001", "jep106:0001:0001")
        self._add_soc(paths, "soc1", "Tegra", "35")
        assert runtime._is_tegra_t234() is True

    def test_smccc_soc_id(self, paths):
        self._add_soc(paths, "soc1", "jep106:036b", "jep106:036b:0234")
        assert runtime._is_tegra_t234() is True

    def test_xavier_chip_id(self, paths):
        self._add_soc(paths, "soc0", "Tegra", "25")
        assert runtime._is_tegra_t234() is False

    def test_chip_id_from_other_family(self, paths):
        self._add_soc(paths, "soc0", "Other", "35")
        assert runtime._is_tegra_t234() is False

    def test_non_numeric_soc_id(self, paths):
        self._add_soc(paths, "soc0", "Tegra", "abc")
        assert runtime._is_tegra_t234() is False

    def test_device_tree_compatible(self, paths):
        (paths / "compatible").write_bytes(b"nvidia,p3737-0000\x00nvidia,tegra234\x00")
        assert runtime._is_tegra_t234() is True


class TestShouldUseExp:
    """Tests for ISAAC_TELEOP_CLOUDXR_EXP selection."""

    def test_default_is_false_off_tegra(self, monkeypatch):
        monkeypatch.delenv("ISAAC_TELEOP_CLOUDXR_EXP", raising=False)
        monkeypatch.setattr(
            "isaaccapture.cloudxr.runtime._is_tegra_t234", lambda: False
        )
        assert _should_use_exp() is False

    def test_auto_true_on_t234(self, monkeypatch):
        monkeypatch.delenv("ISAAC_TELEOP_CLOUDXR_EXP", raising=False)
        monkeypatch.setattr("isaaccapture.cloudxr.runtime._is_tegra_t234", lambda: True)
        assert _should_use_exp() is True

    def test_accepts_truthy(self, monkeypatch):
        monkeypatch.setenv("ISAAC_TELEOP_CLOUDXR_EXP", "1")
        assert _should_use_exp() is True
        monkeypatch.setenv("ISAAC_TELEOP_CLOUDXR_EXP", "true")
        assert _should_use_exp() is True

    def test_rejects_falsy_even_on_t234(self, monkeypatch):
        monkeypatch.setattr("isaaccapture.cloudxr.runtime._is_tegra_t234", lambda: True)
        monkeypatch.setenv("ISAAC_TELEOP_CLOUDXR_EXP", "0")
        assert _should_use_exp() is False
        monkeypatch.setenv("ISAAC_TELEOP_CLOUDXR_EXP", "false")
        assert _should_use_exp() is False


class TestShouldJoinMain:
    """Tests for ISAAC_TELEOP_CLOUDXR_JOIN_MAIN selection."""

    def test_default_is_false_off_tegra(self, monkeypatch):
        monkeypatch.delenv("ISAAC_TELEOP_CLOUDXR_JOIN_MAIN", raising=False)
        monkeypatch.setattr(
            "isaaccapture.cloudxr.runtime._is_tegra_t234", lambda: False
        )
        assert _should_join_main() is False

    def test_auto_true_on_t234(self, monkeypatch):
        monkeypatch.delenv("ISAAC_TELEOP_CLOUDXR_JOIN_MAIN", raising=False)
        monkeypatch.setattr("isaaccapture.cloudxr.runtime._is_tegra_t234", lambda: True)
        assert _should_join_main() is True

    def test_accepts_truthy(self, monkeypatch):
        monkeypatch.setenv("ISAAC_TELEOP_CLOUDXR_JOIN_MAIN", "1")
        assert _should_join_main() is True
        monkeypatch.setenv("ISAAC_TELEOP_CLOUDXR_JOIN_MAIN", "true")
        assert _should_join_main() is True

    def test_rejects_falsy_even_on_t234(self, monkeypatch):
        monkeypatch.setattr("isaaccapture.cloudxr.runtime._is_tegra_t234", lambda: True)
        monkeypatch.setenv("ISAAC_TELEOP_CLOUDXR_JOIN_MAIN", "0")
        assert _should_join_main() is False
        monkeypatch.setenv("ISAAC_TELEOP_CLOUDXR_JOIN_MAIN", "false")
        assert _should_join_main() is False


class TestResolveCloudxrRuntimeModule:
    """Tests for stable vs cloudxr_exp module selection."""

    def test_stable_when_exp_not_wanted(self, monkeypatch):
        monkeypatch.delenv("ISAAC_TELEOP_CLOUDXR_EXP", raising=False)
        monkeypatch.setattr(
            "isaaccapture.cloudxr.runtime._is_tegra_t234", lambda: False
        )
        assert resolve_cloudxr_runtime_module() == "isaaccapture.cloudxr"

    def test_exp_when_available(self, monkeypatch):
        monkeypatch.setenv("ISAAC_TELEOP_CLOUDXR_EXP", "1")
        monkeypatch.setattr(
            "isaaccapture.cloudxr.runtime._is_exp_available", lambda: True
        )
        assert resolve_cloudxr_runtime_module() == "isaaccapture.cloudxr_exp"

    def test_explicit_exp_missing_raises(self, monkeypatch):
        monkeypatch.setenv("ISAAC_TELEOP_CLOUDXR_EXP", "1")
        monkeypatch.setattr(
            "isaaccapture.cloudxr.runtime._is_exp_available", lambda: False
        )
        with pytest.raises(
            RuntimeError, match=r"cloudxr_exp|ENABLE_CLOUDXR_EXP"
        ) as excinfo:
            resolve_cloudxr_runtime_module()
        # The message must blame the missing *artifact*: cloudxr_exp is authored
        # Python that ships in every wheel, so "is not installed" would be false.
        assert "libcloudxr.so" in str(excinfo.value)

    @pytest.mark.parametrize("artifact_index", [0, 1])
    def test_is_exp_available_scans_every_root(
        self, tmp_path, monkeypatch, artifact_index
    ):
        """The bundled runtime is found in whichever root holds it."""
        roots, _ = _make_package_roots(tmp_path, artifact_index)
        calls = _patch_find_spec(monkeypatch, {"isaaccapture.cloudxr_exp": roots})

        assert _is_exp_available() is True
        # Bare package name only -- a dotted submodule would import cloudxr_exp.
        assert calls == ["isaaccapture.cloudxr_exp"]

    def test_is_exp_available_false_without_artifact(self, tmp_path, monkeypatch):
        """An importable cloudxr_exp with an empty native/ is NOT available.

        This is the regression that matters: probing module presence would
        report True here, and every Orin install would then fail in
        get_sdk_path() because the experimental runtime was never bundled.
        """
        roots, _ = _make_package_roots(tmp_path, None)
        calls = _patch_find_spec(monkeypatch, {"isaaccapture.cloudxr_exp": roots})

        assert _is_exp_available() is False
        assert calls == ["isaaccapture.cloudxr_exp"]

    def test_is_exp_available_handles_missing_parent(self, monkeypatch):
        """A package that cannot be located is not available, and does not raise."""

        def _boom(_name: str, _package: str | None = None):
            raise ModuleNotFoundError("isaaccapture.cloudxr_exp")

        monkeypatch.setattr(
            "isaaccapture.cloudxr.runtime.importlib.util.find_spec", _boom
        )

        assert _is_exp_available() is False

    def test_is_exp_available_handles_unspecced_parent(self, monkeypatch):
        """find_spec raises ValueError when a parent has ``__spec__`` set to None."""

        def _boom(_name: str, _package: str | None = None):
            raise ValueError("isaaccapture.__spec__ is None")

        monkeypatch.setattr(
            "isaaccapture.cloudxr.runtime.importlib.util.find_spec", _boom
        )

        assert _is_exp_available() is False

    def test_is_exp_available_handles_absent_package(self, monkeypatch):
        """find_spec returns None for a name the finders cannot resolve."""

        def _none(_name: str, _package: str | None = None):
            return None

        monkeypatch.setattr(
            "isaaccapture.cloudxr.runtime.importlib.util.find_spec", _none
        )

        assert _is_exp_available() is False

    def test_auto_t234_missing_raises(self, monkeypatch):
        monkeypatch.delenv("ISAAC_TELEOP_CLOUDXR_EXP", raising=False)
        monkeypatch.setattr("isaaccapture.cloudxr.runtime._is_tegra_t234", lambda: True)
        monkeypatch.setattr(
            "isaaccapture.cloudxr.runtime._is_exp_available", lambda: False
        )
        with pytest.raises(RuntimeError, match=r"cloudxr_exp|ENABLE_CLOUDXR_EXP"):
            resolve_cloudxr_runtime_module()


class TestGetSdkPath:
    """Tests for selected-package native/ resolution across every __path__ root."""

    @staticmethod
    def _select(monkeypatch, module: str) -> None:
        monkeypatch.setattr(
            "isaaccapture.cloudxr.runtime.resolve_cloudxr_runtime_module",
            lambda: module,
        )

    @pytest.mark.parametrize("artifact_index", [0, 1])
    def test_scans_every_root(self, tmp_path, monkeypatch, artifact_index):
        """Resolution must not depend on root order: __path__ is unordered."""
        roots, expected = _make_package_roots(tmp_path, artifact_index)
        self._select(monkeypatch, "isaaccapture.cloudxr")
        calls = _patch_find_spec(monkeypatch, {"isaaccapture.cloudxr": roots})

        assert get_sdk_path() == expected
        assert calls == ["isaaccapture.cloudxr"]

    def test_missing_raises(self, tmp_path, monkeypatch):
        """No root holds libcloudxr.so -- an empty native/ must not satisfy it."""
        roots, _ = _make_package_roots(tmp_path, None)
        self._select(monkeypatch, "isaaccapture.cloudxr")
        _patch_find_spec(monkeypatch, {"isaaccapture.cloudxr": roots})

        with pytest.raises(RuntimeError, match=r"libcloudxr\.so") as excinfo:
            get_sdk_path()
        for root in roots:
            assert root in str(excinfo.value)

    def test_follows_exp_selection(self, tmp_path, monkeypatch):
        roots, expected = _make_package_roots(tmp_path, 1)
        self._select(monkeypatch, "isaaccapture.cloudxr_exp")
        calls = _patch_find_spec(monkeypatch, {"isaaccapture.cloudxr_exp": roots})

        assert get_sdk_path() == expected
        assert calls == ["isaaccapture.cloudxr_exp"]


# ============================================================================
# TestWaitForRuntimeReadySync
# ============================================================================


class TestWaitForRuntimeReadySync:
    """Tests for the synchronous sentinel-file polling helper."""

    def test_returns_true_when_sentinel_exists(self, tmp_path):
        """Immediately returns True when runtime_started already exists."""
        run_dir = str(tmp_path / "run")
        os.makedirs(run_dir)
        (tmp_path / "run" / "runtime_started").touch()

        fake_cfg = _FakeEnvConfig(run_dir)
        with patch(
            "isaaccapture.cloudxr.runtime.get_env_config", return_value=fake_cfg
        ):
            result = wait_for_runtime_ready_sync(
                is_process_alive=lambda: True,
                timeout_sec=1.0,
                poll_interval_sec=0.05,
            )

        assert result is True

    def test_returns_false_on_timeout(self, tmp_path):
        """Returns False when sentinel never appears within the timeout."""
        run_dir = str(tmp_path / "run")
        os.makedirs(run_dir)

        fake_cfg = _FakeEnvConfig(run_dir)
        with patch(
            "isaaccapture.cloudxr.runtime.get_env_config", return_value=fake_cfg
        ):
            start = time.monotonic()
            result = wait_for_runtime_ready_sync(
                is_process_alive=lambda: True,
                timeout_sec=0.2,
                poll_interval_sec=0.05,
            )
            elapsed = time.monotonic() - start

        assert result is False
        assert elapsed >= 0.2

    def test_returns_false_when_process_dies(self, tmp_path):
        """Returns False immediately when is_process_alive reports dead."""
        run_dir = str(tmp_path / "run")
        os.makedirs(run_dir)

        fake_cfg = _FakeEnvConfig(run_dir)
        with patch(
            "isaaccapture.cloudxr.runtime.get_env_config", return_value=fake_cfg
        ):
            start = time.monotonic()
            result = wait_for_runtime_ready_sync(
                is_process_alive=lambda: False,
                timeout_sec=5.0,
                poll_interval_sec=0.05,
            )
            elapsed = time.monotonic() - start

        assert result is False
        assert elapsed < 1.0

    def test_detects_sentinel_created_mid_wait(self, tmp_path):
        """Returns True when sentinel appears partway through the wait."""
        run_dir = str(tmp_path / "run")
        os.makedirs(run_dir)
        sentinel = tmp_path / "run" / "runtime_started"

        def _create_sentinel_later():
            time.sleep(0.15)
            sentinel.touch()

        threading.Thread(target=_create_sentinel_later, daemon=True).start()

        fake_cfg = _FakeEnvConfig(run_dir)
        with patch(
            "isaaccapture.cloudxr.runtime.get_env_config", return_value=fake_cfg
        ):
            result = wait_for_runtime_ready_sync(
                is_process_alive=lambda: True,
                timeout_sec=2.0,
                poll_interval_sec=0.05,
            )

        assert result is True

    def test_respects_custom_timeout_and_poll_interval(self, tmp_path):
        """Completes quickly with a tiny timeout, honouring custom values."""
        run_dir = str(tmp_path / "run")
        os.makedirs(run_dir)

        fake_cfg = _FakeEnvConfig(run_dir)
        with patch(
            "isaaccapture.cloudxr.runtime.get_env_config", return_value=fake_cfg
        ):
            start = time.monotonic()
            result = wait_for_runtime_ready_sync(
                is_process_alive=lambda: True,
                timeout_sec=0.1,
                poll_interval_sec=0.02,
            )
            elapsed = time.monotonic() - start

        assert result is False
        assert elapsed < 0.5


# ============================================================================
# TestTerminateOrKillRuntime
# ============================================================================


def _make_mock_process(alive_sequence: list[bool]) -> MagicMock:
    """Create a mock multiprocessing.Process whose is_alive() returns values from a sequence.

    Each call to is_alive() pops the next value; once exhausted it always returns False.
    """
    proc = MagicMock()
    seq = list(alive_sequence)

    def _is_alive():
        if seq:
            return seq.pop(0)
        return False

    proc.is_alive = MagicMock(side_effect=_is_alive)
    return proc


class TestTerminateOrKillRuntime:
    """Tests for the multiprocessing.Process termination helper."""

    def test_terminates_cleanly(self):
        """Process exits after terminate() — no kill needed."""
        proc = _make_mock_process([True, False])
        terminate_or_kill_runtime(proc)

        proc.terminate.assert_called_once()
        proc.kill.assert_not_called()

    def test_escalates_to_kill(self):
        """Process survives terminate(), exits after kill()."""
        # is_alive() is called 3 times: before terminate, before kill, final check
        proc = _make_mock_process([True, True, False])
        terminate_or_kill_runtime(proc)

        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()

    def test_raises_if_unkillable(self):
        """RuntimeError when process stays alive after both terminate and kill."""
        proc = _make_mock_process([True, True, True, True, True])
        with pytest.raises(RuntimeError, match="Failed to terminate or kill"):
            terminate_or_kill_runtime(proc)

    def test_noop_if_already_dead(self):
        """No terminate/kill calls when the process is already dead."""
        proc = _make_mock_process([False])
        terminate_or_kill_runtime(proc)

        proc.terminate.assert_not_called()
        proc.kill.assert_not_called()


@pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"), reason="AF_UNIX sockets are POSIX-only"
)
class TestIsRuntimeLive:
    """Tests for the IPC-socket liveness probe."""

    def test_false_when_socket_missing(self, tmp_path):
        """An empty run directory has no runtime."""
        run_dir = str(tmp_path)
        assert is_runtime_live(run_dir) is False

    def test_false_for_leftover_socket_file(self, tmp_path):
        """A socket file nobody listens on is the post-crash case: not live."""
        run_dir = str(tmp_path)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        cwd = os.getcwd()
        try:
            os.chdir(run_dir)
            sock.bind("ipc_cloudxr")
        finally:
            os.chdir(cwd)
            sock.close()

        assert os.path.exists(os.path.join(run_dir, "ipc_cloudxr"))
        assert is_runtime_live(run_dir) is False

    def test_true_while_socket_is_served(self, tmp_path):
        """A listening peer means a runtime owns the run directory."""
        run_dir = str(tmp_path)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        cwd = os.getcwd()
        try:
            os.chdir(run_dir)
            sock.bind("ipc_cloudxr")
            sock.listen(1)
            os.chdir(cwd)
            assert is_runtime_live(run_dir) is True
        finally:
            os.chdir(cwd)
            sock.close()

    def test_ambiguous_errors_count_as_live(self, tmp_path):
        """Errors that do not prove absence must not license a takeover."""
        run_dir = str(tmp_path)
        (tmp_path / "ipc_cloudxr").touch()
        with patch("socket.socket") as mock_socket:
            mock_socket.return_value.connect.side_effect = PermissionError
            assert is_runtime_live(run_dir) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestReadExportedEnvRobustness:
    """The env file describes a live runtime; a bad line must not fail the read."""

    def test_a_malformed_value_is_skipped_not_raised(self, tmp_path):
        from isaaccapture.cloudxr.env_config import read_exported_env

        path = tmp_path / "cloudxr.env"
        path.write_text(
            "export GOOD=fine\nexport BROKEN='unclosed\nexport ALSO_GOOD=yes\n",
            encoding="utf-8",
        )
        env = read_exported_env(path)
        assert env == {"GOOD": "fine", "ALSO_GOOD": "yes"}

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make CloudXR python sources importable without installing ``isaaccapture``.

* Flat ``sys.path`` entry: ``from oob_teleop_hub import …`` (no relative imports).
* Synthetic package ``cloudxr_py_test_ns``: ``from cloudxr_py_test_ns.oob_teleop_env import …``
  so modules that use sibling relative imports load correctly, under a synthetic
  ``isaaccapture`` root so that the ones reaching ``..`` load too.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import socket
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

_tests_python = Path(__file__).resolve().parents[2]
if str(_tests_python) not in sys.path:
    sys.path.insert(0, str(_tests_python))

from repo_paths import repo_root  # noqa: E402

_CLOUDXR_PY = repo_root() / "src" / "python" / "isaaccapture" / "cloudxr"
if _CLOUDXR_PY.is_dir() and str(_CLOUDXR_PY) not in sys.path:
    sys.path.insert(0, str(_CLOUDXR_PY))

CLOUDXR_TEST_PKG = "cloudxr_py_test_ns"

# ``wss`` reaches its own package's parent (``from ..logging_config._core import``), which a
# one-level synthetic package has nothing to resolve. This root stands in for
# ``isaaccapture`` and takes its ``__path__`` from the real source tree, so a sibling
# subpackage loads from source with nothing installed. A module's real name therefore
# has two levels, and every one is aliased back to CLOUDXR_TEST_PKG -- the name the
# tests patch by.
_TEST_ROOT_PKG = "isaaccapture_py_test_ns"


def _ensure_cloudxr_package() -> None:
    if CLOUDXR_TEST_PKG in sys.modules:
        return
    root = types.ModuleType(_TEST_ROOT_PKG)
    root.__path__ = [str(_CLOUDXR_PY.parent)]
    sys.modules[_TEST_ROOT_PKG] = root

    pkg_name = f"{_TEST_ROOT_PKG}.cloudxr"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(_CLOUDXR_PY)]
    sys.modules[pkg_name] = pkg
    sys.modules[CLOUDXR_TEST_PKG] = pkg
    root.cloudxr = pkg

    def load(mod: str) -> None:
        full = f"{pkg_name}.{mod}"
        path = _CLOUDXR_PY / f"{mod}.py"
        spec = importlib.util.spec_from_file_location(full, path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        sys.modules[f"{CLOUDXR_TEST_PKG}.{mod}"] = module
        spec.loader.exec_module(module)
        setattr(pkg, mod, module)

    load("oob_teleop_hub")
    load("oob_teleop_env")
    load("oob_teleop_adb")
    load("webclient")
    # Preloaded rather than left to the package's ``__path__``: an import through
    # CLOUDXR_TEST_PKG would name it one level up, and ``..`` would be out of range.
    load("wss")


_ensure_cloudxr_package()


# ============================================================================
# Shared CloudXRService test doubles (used by test_service.py + test_launcher.py)
# ============================================================================


class FakeEnvConfig:
    """Minimal stand-in for EnvConfig."""

    def __init__(self, run_dir: str, logs_dir: Path) -> None:
        self._run_dir = run_dir
        self._logs_dir = logs_dir

    @classmethod
    def from_args(cls, install_dir, env_file=None):
        raise NotImplementedError("Should be patched")

    def openxr_run_dir(self) -> str:
        return self._run_dir

    def ensure_logs_dir(self) -> Path:
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        return self._logs_dir

    def env_filepath(self) -> str:
        return os.path.join(self._run_dir, "cloudxr.env")


@contextmanager
def live_ipc_socket(run_dir: str):
    """Serve ``run_dir``'s IPC socket for the duration of the block.

    Binds relative from a chdir: AF_UNIX ``sun_path`` caps at 108 bytes,
    which pytest's tmp_path can exceed.
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    cwd = os.getcwd()
    try:
        os.chdir(run_dir)
        with contextlib.suppress(FileNotFoundError):
            os.remove("ipc_cloudxr")
        sock.bind("ipc_cloudxr")
        sock.listen(1)
        yield sock
    finally:
        os.chdir(cwd)
        sock.close()


def make_mock_popen(pid: int = 12345, poll_returns: list | None = None) -> MagicMock:
    """Create a mock subprocess.Popen with configurable poll() behaviour."""
    proc = MagicMock()
    proc.pid = pid
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = MagicMock()

    if poll_returns is not None:
        seq = list(poll_returns)

        def _poll():
            if seq:
                return seq.pop(0)
            return 0

        proc.poll = MagicMock(side_effect=_poll)
    else:
        proc.poll = MagicMock(return_value=None)

    return proc


@contextmanager
def mock_service_deps(tmp_path, ready=True, wss=True):
    """Patch process and network dependencies for isolated service construction.

    Yields a dict of the mock objects for assertion.  Pass ``wss=False`` to
    leave ``_start_wss_proxy_thread`` real, for tests about the proxy's own
    start-up; ``mocks["wss"]`` is then ``None``.
    """
    from isaaccapture.cloudxr.service import CloudXRService  # noqa: PLC0415

    run_dir = str(tmp_path / "run")
    logs_dir = tmp_path / "logs"
    fake_cfg = FakeEnvConfig(run_dir, logs_dir)
    static_dir = tmp_path / "static-client"
    static_dir.mkdir(parents=True, exist_ok=True)
    (static_dir / "index.html").write_text("<!doctype html>", encoding="utf-8")
    (static_dir / "bundle.js").write_text("// test bundle", encoding="utf-8")

    mock_proc = make_mock_popen()
    wss_patch = (
        patch.object(CloudXRService, "_start_wss_proxy_thread")
        if wss
        else contextlib.nullcontext()
    )

    mocks = {}
    with (
        patch(
            "isaaccapture.cloudxr.service._service.EnvConfig.from_args",
            return_value=fake_cfg,
        ) as m_from_args,
        patch(
            "isaaccapture.cloudxr.service._service.check_eula",
        ) as m_eula,
        # The worker is mocked, so lifecycle tests need no installed native SDK.
        patch(
            "isaaccapture.cloudxr.service._service.get_sdk_path",
            return_value=str(tmp_path / "sdk"),
        ),
        patch(
            "isaaccapture.cloudxr.service._service.wait_for_runtime_ready_sync",
            return_value=ready,
        ) as m_wait,
        patch(
            "isaaccapture.cloudxr.oob_teleop_env.require_web_client_static_dir",
            return_value=static_dir,
        ) as m_static_client,
        patch(
            "isaaccapture.cloudxr.service._service.subprocess.Popen",
            return_value=mock_proc,
        ) as m_popen,
        wss_patch as m_wss,
        patch.object(
            CloudXRService,
            "_cleanup_stale_runtime",
        ) as m_cleanup,
        patch(
            "isaaccapture.cloudxr.service._service.atexit",
        ) as m_atexit,
    ):
        mocks["from_args"] = m_from_args
        mocks["check_eula"] = m_eula
        mocks["wait"] = m_wait
        mocks["static_client"] = m_static_client
        mocks["popen"] = m_popen
        mocks["proc"] = mock_proc
        mocks["wss"] = m_wss
        mocks["cleanup"] = m_cleanup
        mocks["atexit"] = m_atexit
        mocks["env_cfg"] = fake_cfg
        yield mocks

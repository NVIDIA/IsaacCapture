# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make CloudXR python sources importable without installing ``isaaccapture``.

* Flat ``sys.path`` entry: ``from oob_teleop_hub import …`` (no relative imports).
* Synthetic package ``cloudxr_py_test_ns``: ``from cloudxr_py_test_ns.oob_teleop_env import …``
  so modules that use sibling relative imports load correctly, under a synthetic
  ``isaaccapture`` root so that the ones reaching ``..`` load too.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
import types
import urllib.error
import urllib.request
from contextlib import asynccontextmanager, contextmanager
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


# ============================================================================
# Fake adb (used by oob_teleop_adb tests: run_oob_connect() and friends)
# ============================================================================


class FakeAdb:
    """Emulates real ``adb`` CLI output for the commands oob_teleop_adb's launch/repair
    call chain invokes (device-state checks, ``/proc/net/unix`` scanning, ``am start``,
    ``adb forward``/``--remove``, ``getprop``), so a test can drive that real chain -
    exercising its actual parsing/command-building logic, not just assuming it works -
    without shelling out to a real adb/device.

    Use via :func:`mock_adb`, which patches ``subprocess.run`` with an instance of this
    and yields it. Every helper in ``oob_teleop_adb`` that shells out ultimately calls
    ``subprocess.run`` (directly, or through its own ``_run_adb`` wrapper), so patching
    at that one boundary covers ``_discover_devtools_socket``, ``run_adb_headset_bookmark``/
    ``open_url_on_headset``, ``_adb_forward_cdp``/``_adb_forward_remove``,
    ``_close_stale_teleop_tabs``, and ``assert_adb_device_online`` alike.

    An unrecognized command raises rather than silently succeeding, so a test surfaces
    any new adb invocation this fake doesn't yet know how to answer instead of passing
    for the wrong reason.
    """

    def __init__(
        self,
        *,
        device_state: str = "device",
        devtools_socket: str = "chrome_devtools_remote",
        am_start_rc: int = 0,
        forward_rc: int = 0,
    ) -> None:
        self.device_state = device_state
        self.devtools_socket = devtools_socket
        self.am_start_rc = am_start_rc
        self.forward_rc = forward_rc
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str], **kwargs) -> subprocess.CompletedProcess:
        self.calls.append(list(args))

        def result(rc: int, stdout: str = "") -> subprocess.CompletedProcess:
            completed = subprocess.CompletedProcess(args, rc, stdout, "")
            if kwargs.get("check", False):
                completed.check_returncode()
            return completed

        if args[:2] == ["adb", "get-state"]:
            return result(0 if self.device_state == "device" else 1, self.device_state)
        if args[:2] == ["adb", "reconnect"]:
            return result(0)
        if args[:3] == ["adb", "shell", "cat"] and args[-1] == "/proc/net/unix":
            # One abstract-socket line matching _DEVTOOLS_SOCKET_RE's `@<name>_devtools_
            # remote` token, mirroring what a real Chromium-based browser's DevTools
            # listener looks like in a genuine `cat /proc/net/unix` dump. An empty
            # devtools_socket produces a line with no matching token, simulating "browser
            # never exposed a socket" without needing a separate empty-output branch.
            line = (
                "0000000000000000: 00000002 00000000 00010000 01 0 0 "
                f"@{self.devtools_socket}"
                if self.devtools_socket
                else "0000000000000000: 00000002 00000000 00010000 01 0 0 @not_a_match"
            )
            return result(0, line)
        if args[:3] == ["adb", "shell", "getprop"]:
            return result(
                0, ""
            )  # unknown vendor - falls back to the generic VIEW intent
        if (
            len(args) == 4
            and args[:3]
            in (["adb", "forward", "--remove"], ["adb", "reverse", "--remove"])
            and args[3].startswith("tcp:")
        ):
            return result(0)
        if args[:2] == ["adb", "forward"]:
            return result(self.forward_rc)
        if args[:2] == ["adb", "shell"] and any("am start" in a for a in args):
            return result(self.am_start_rc)
        raise AssertionError(f"unscripted adb call: {args}")


@contextmanager
def mock_adb(**kwargs):
    """Patches ``subprocess.run`` with a :class:`FakeAdb` (constructed from *kwargs*) for
    the duration of the block; yields the instance so a test can inspect ``.calls``."""
    fake = FakeAdb(**kwargs)
    with patch("subprocess.run", side_effect=fake):
        yield fake


# ============================================================================
# Real-browser OOB integration test-infra (see /oob-real-browser-integration-
# test-plan.md at the repo root): a real Chromium running the real webxr
# client, driven by run_oob_connect() through a RealBrowserAdb whose
# success-path commands have real side effects instead of canned output.
# ============================================================================

_CHROME_EXECUTABLE = os.environ.get("PLAYWRIGHT_CHROME_PATH", "/usr/bin/google-chrome")

# Matches deps/cloudxr/webxr_client/playwright.config.js's CHROMIUM_ARGS:
# SwiftShader software WebGL (no GPU in this sandbox) and --disable-features=WebXR
# so native/real XR is off and IWER (loaded by the page itself) provides navigator.xr.
_CHROME_LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--ignore-gpu-blocklist",
    "--enable-webgl",
    "--use-angle=swiftshader",
    "--use-gl=angle",
    "--disable-features=WebXR",
]


def _free_tcp_port() -> int:
    """Bind an ephemeral port and immediately release it for a subprocess to reuse."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_http(url: str, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last_exc = exc
            time.sleep(0.2)
    raise RuntimeError(f"{url} did not respond within {timeout:.0f}s") from last_exc


class RealChrome:
    """A real, locally-launched Chromium instance reachable over its own CDP port."""

    def __init__(self, cdp_port: int) -> None:
        self.cdp_port = cdp_port

    def open_url(self, url: str) -> dict:
        """Open *url* in a new real tab via CDP's HTTP endpoint (``PUT /json/new``)."""
        req = urllib.request.Request(
            f"http://localhost:{self.cdp_port}/json/new?{url}", method="PUT"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())

    def list_tabs(self) -> list[dict]:
        """Real ``GET /json`` — every open tab, with its ``webSocketDebuggerUrl``."""
        with urllib.request.urlopen(
            f"http://localhost:{self.cdp_port}/json", timeout=5
        ) as resp:
            return json.loads(resp.read())


@contextmanager
def real_chrome(cdp_port: int, *, user_data_dir: Path):
    """Launch a real local Chromium reachable over CDP on *cdp_port*, for the block's duration."""
    args = [
        _CHROME_EXECUTABLE,
        f"--remote-debugging-port={cdp_port}",
        f"--user-data-dir={user_data_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        *_CHROME_LAUNCH_ARGS,
        "about:blank",
    ]
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_for_http(f"http://localhost:{cdp_port}/json/version", timeout=15.0)
        yield RealChrome(cdp_port)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


@contextmanager
def static_webxr_build(port: int, *, npm_build_script: str, build_dir_name: str):
    """Build webxr_client once via a real ``npm run <npm_build_script>`` production
    build (no dev-server, no HMR), then serve the resulting static directory with a
    plain HTTP server for the block's duration.

    Deliberately never a dev-server: webpack's hot-module-replacement re-executes a
    module's top-level code on live-reload, creating a *second* instance of it
    side by side with the one React already mounted against — for
    ``tests/mock/cloudxr-mock-alias.ts`` this means ``window.__mockCloudXRFail()``
    silently binds to a fresh, never-used ``activeSession`` while the real,
    already-running session lives on in the original instance untouched. A static
    build has no live-reload runtime at all, so this class of bug is structurally
    impossible, not just avoided by luck.

    *npm_build_script*/*build_dir_name* pairs (see ``package.json``/the matching
    ``webpack.*.js``). Always use the ``build:app-mock`` pair for a real-browser
    test: the real SDK genuinely tries to stream against whatever's on
    ``backend_port``, which is unpredictable and uncontrollable to assert against.

    * ``"build"`` / ``"build"`` — the real production client against the real
      ``@nvidia/cloudxr`` SDK. Not for tests — kept only as a manual sanity
      build; see the warning above.
    * ``"build:app-mock"`` / ``"build-app-mock"`` — the identical ``App.tsx`` /
      ``CloudXRComponent.tsx`` UI (same ``#startButton``/``#errorMessageBox``
      DOM), but with ``@nvidia/cloudxr`` aliased to ``MockCloudXR``, which exposes
      ``window.__mockCloudXRFail(message?, code?)`` globally so a CDP
      ``Runtime.evaluate`` call can force a deterministic, controllable
      mid-session failure — see :func:`cdp_evaluate`. ``MockCloudXR`` opens no
      socket of its own, so nothing needs to stand in for a CloudXR runtime at
      all.

    Always rebuilds (mirrors ``cloudxr-js``'s ``global-setup.js``: a stale bundle
    from a previous branch must never silently pass) rather than reusing whatever
    happens to be on disk or already listening on *port*.
    """
    webxr_client_dir = repo_root() / "deps" / "cloudxr" / "webxr_client"
    # webpack.common.js turns on persistent filesystem caching
    # (cache: {type: 'filesystem'}), keyed only on the config file
    # (buildDependencies.config), not on every source file it compiles. A stale
    # entry there can silently serve an old compiled module for a source file
    # that *did* change - hit for real while building this fixture: an edit to
    # tests/mock/cloudxr-mock-alias.ts was invisible in the output bundle
    # (`grep` for the new symbol found nothing) until this cache was cleared.
    # Correctness matters far more than incremental-build speed for a test
    # fixture, so always start from a clean cache rather than trust it.
    shutil.rmtree(
        webxr_client_dir / "node_modules" / ".cache" / "webpack", ignore_errors=True
    )
    subprocess.run(
        ["npm", "run", npm_build_script],
        cwd=webxr_client_dir,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    build_dir = webxr_client_dir / build_dir_name
    if not (build_dir / "index.html").is_file():
        raise RuntimeError(
            f"{build_dir} has no index.html after `npm run {npm_build_script}`"
        )

    from functools import partial
    from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
    from threading import Thread

    handler = partial(SimpleHTTPRequestHandler, directory=str(build_dir))
    httpd = ThreadingHTTPServer(("localhost", port), handler)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        _wait_for_http(f"http://localhost:{port}/", timeout=15.0)
        yield
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


async def cdp_evaluate(ws_url: str, expression: str) -> dict:
    """Open a one-shot CDP session to *ws_url* and evaluate *expression*.

    Mirrors the ``send()`` helper duplicated in ``_cdp_session_click_connect``/
    ``_monitor_teleop_error_banner`` (``oob_teleop_adb.py``), for a test that
    needs to poke a real tab from outside those functions — e.g. calling
    ``window.__mockCloudXRFail()`` to force a deterministic mid-session error.
    """
    from websockets.asyncio.client import connect as ws_connect  # noqa: PLC0415

    async with ws_connect(ws_url) as ws:
        await ws.send(
            json.dumps(
                {
                    "id": 1,
                    "method": "Runtime.evaluate",
                    "params": {"expression": expression, "returnByValue": True},
                }
            )
        )
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10.0))
            if msg.get("id") == 1:
                return msg.get("result", {})


@asynccontextmanager
async def real_wss_proxy(install_dir: Path, *, proxy_port: int, backend_port: int):
    """Run the real ``wss.py`` proxy (TLS + OOB hub) for the block's duration.

    ``TELEOP_OOB_HUB_ONLY=1`` makes ``setup_oob=True`` wire up the real
    ``OOBControlHub`` without also having ``wss.py`` itself call
    ``run_oob_connect()`` on our behalf — the test drives that call directly,
    against the real proxy this starts. ``usb_local=False`` here (a ``wss.run()``
    param, independent of the ``usb_local`` a test passes to its own
    ``run_oob_connect()`` call) skips ``wss.py``'s own adb-reverse/coturn setup,
    which has no real device to run against in this test.
    """
    from cloudxr_py_test_ns.wss import run as wss_run  # noqa: PLC0415

    os.environ["CXR_INSTALL_DIR"] = str(install_dir)
    prev_hub_only = os.environ.get("TELEOP_OOB_HUB_ONLY")
    os.environ["TELEOP_OOB_HUB_ONLY"] = "1"

    stop_future: asyncio.Future = asyncio.get_running_loop().create_future()
    listening = asyncio.Event()

    task = asyncio.create_task(
        wss_run(
            None,
            stop_future,
            backend_host="localhost",
            backend_port=backend_port,
            proxy_port=proxy_port,
            setup_oob=True,
            usb_local=False,
            host_client=False,
            on_listening=listening.set,
        )
    )
    try:
        await asyncio.wait_for(listening.wait(), timeout=15.0)
        yield
    finally:
        if not stop_future.done():
            stop_future.set_result(None)
        await asyncio.wait_for(task, timeout=10.0)
        if prev_hub_only is None:
            os.environ.pop("TELEOP_OOB_HUB_ONLY", None)
        else:
            os.environ["TELEOP_OOB_HUB_ONLY"] = prev_hub_only


class RealBrowserAdb(FakeAdb):
    """Like :class:`FakeAdb`, but ``am start`` and ``adb forward`` have real side
    effects against a real :class:`RealChrome` instead of returning canned output.

    Error-injection kwargs (``device_state``, ``am_start_rc``, ``devtools_socket``,
    ``forward_rc``) work exactly as on :class:`FakeAdb`: the fake side of each still
    gates whether the real side effect happens at all, so tests can still
    deliberately drive the same failure paths ``FakeAdb``-based tests do.

    ``adb forward`` is a real no-op (not a real TCP forward) because *chrome*
    is launched with ``--remote-debugging-port`` equal to
    ``oob_teleop_adb._CDP_LOCAL_PORT`` already — the port ``run_oob_connect()``
    forwards to is already the real CDP port, so there is nothing to actually
    forward.
    """

    def __init__(self, chrome: RealChrome, **kwargs) -> None:
        super().__init__(**kwargs)
        self._chrome = chrome

    def __call__(self, args: list[str], **kwargs) -> subprocess.CompletedProcess:
        if args[:2] == ["adb", "shell"] and any("am start" in a for a in args):
            self.calls.append(list(args))
            if self.am_start_rc != 0:
                return subprocess.CompletedProcess(
                    args, self.am_start_rc, "", "am start failed"
                )
            shell_cmd = args[2]
            tokens = shlex.split(shell_cmd)
            url = tokens[tokens.index("-d") + 1]
            self._chrome.open_url(url)
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[:2] == ["adb", "forward"] and "--remove" not in args:
            self.calls.append(list(args))
            return subprocess.CompletedProcess(args, self.forward_rc, "", "")
        return super().__call__(args, **kwargs)


@contextmanager
def real_browser_adb(chrome: RealChrome, **kwargs):
    """Patches ``subprocess.run`` with a :class:`RealBrowserAdb` for the block's
    duration; yields the instance so a test can inspect ``.calls``."""
    fake = RealBrowserAdb(chrome, **kwargs)
    with patch("subprocess.run", side_effect=fake):
        yield fake

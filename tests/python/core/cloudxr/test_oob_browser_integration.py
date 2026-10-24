# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-browser OOB tests: the headset launch path, the hub, and the web client together.

The fragment token travels from the host to the hub, and client timeouts set in the
bookmark reach the page.

``build_teleop_url`` -> ``run_oob_connect`` -> :class:`conftest.FakeAdb` ``am start`` ->
real Chrome opens the exact recorded URL -> the real web client (``npm run
build:app-mock``: real ``App.tsx``/``CloudXRComponent``, MockCloudXR as the SDK) registers
with the real ``wss.run`` + ``OOBControlHub``.

The browser cases need Chrome and an npm build, so they run only with
``RUN_OOB_BROWSER_TESTS=1`` (``-m oob_browser``); nothing runs them automatically.
Chrome here ignores TLS errors: certificate trust is bypassed, not covered.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import socket
import ssl
import subprocess
import threading
import time
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect as ws_connect

from cloudxr_py_test_ns import oob_teleop_adb as adb_module
from cloudxr_py_test_ns import oob_teleop_env as env_module
from conftest import FakeAdb, mock_adb
from repo_paths import repo_root

# Unpatched, for helpers that run while ``subprocess.run`` is faked.
_REAL_RUN = subprocess.run

# A token no URL builder may pass through unencoded.
_AWKWARD_TOKEN = "tok&a=b?c#d+e f'\"g"

_CHROME = os.environ.get("PLAYWRIGHT_CHROME_PATH", "/usr/bin/google-chrome")
_CHROME_ARGS = [
    "--no-sandbox",
    "--ignore-gpu-blocklist",
    "--use-angle=swiftshader",
    "--use-gl=angle",
    "--disable-features=WebXR",
    "--ignore-certificate-errors",
    "--no-first-run",
    "--no-default-browser-check",
]


def _intent_url(shell_cmd: str) -> str:
    """The ``-d`` argument of an ``am start`` command, as the headset shell would pass it."""
    script = (
        'am() { while [ "$#" -gt 0 ]; do '
        '[ "$1" = "-d" ] && { printf "%s" "$2"; return; }; shift; done; }; ' + shell_cmd
    )
    return _REAL_RUN(
        ["/bin/sh", "-c", script], capture_output=True, text=True, check=True
    ).stdout


def test_launch_command_keeps_the_whole_url_through_sh(monkeypatch):
    """``&``, ``?``, ``#`` and quotes in the token must not split the ``-d`` argument."""
    monkeypatch.setenv("TELEOP_CLIENT_ROUTE", "/sim/isaaclab")
    monkeypatch.setenv("TELEOP_WEB_CLIENT_BASE", "http://localhost:8080")
    monkeypatch.setenv("CONTROL_TOKEN", _AWKWARD_TOKEN)
    url = adb_module.build_teleop_url(resolved_port=48322, usb_local=True)

    with mock_adb() as adb:
        assert adb_module.open_url_on_headset(url) == (0, "")
    shell_cmd = next(
        c[2] for c in adb.calls if c[:2] == ["adb", "shell"] and "am start" in c[2]
    )

    assert _intent_url(shell_cmd) == url
    fragment = urlsplit(url).fragment
    assert fragment.startswith("/sim/isaaclab?controlToken=")
    assert parse_qs(fragment.split("?", 1)[1])["controlToken"] == [_AWKWARD_TOKEN]


# ---------------------------------------------------------------------------
# Browser harness
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """An ephemeral loopback port; released immediately for the subprocess to take."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_http(url: str, timeout: float) -> None:
    """Block until *url* answers, or raise."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return
        except OSError:
            if time.monotonic() > deadline:
                raise RuntimeError(f"{url} did not answer within {timeout}s") from None
            time.sleep(0.2)


async def _wait_for(check, what: str, timeout: float = 30.0):
    """Poll the async *check* until it returns something truthy; fail with *what*."""
    deadline = time.monotonic() + timeout
    while True:
        value = await check()
        if value:
            return value
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out after {timeout}s waiting for {what}")
        await asyncio.sleep(0.25)


class _Recorder(SimpleHTTPRequestHandler):
    """Static file handler that records every request target and Referer."""

    def do_GET(self):
        self.server.seen.append((self.path, self.headers.get("Referer") or ""))
        super().do_GET()

    def log_message(self, *args):
        pass


class _Tab:
    """Opens a URL in Chrome over CDP, keeping console output from before navigation.

    ``/json/new?<url>`` would drop a ``#fragment``, so the tab is created blank, console
    capture is enabled, and only then navigated to the exact URL.
    """

    def __init__(self, cdp_port: int) -> None:
        self.cdp_port = cdp_port
        self.console: list[str] = []  # console lines and uncaught exceptions
        self.navigations: list[str] = []  # main-frame URLs, fragment dropped
        self._ws = None
        self._id = 0

    def _call(self, method, params=None, session=None):
        self._id += 1
        msg = {"id": self._id, "method": method, "params": params or {}}
        if session:
            msg["sessionId"] = session
        self._ws.send(json.dumps(msg))
        while True:
            reply = json.loads(self._ws.recv(timeout=10))
            if reply.get("id") == self._id:
                return reply.get("result", {})
            self._note(reply)

    def _note(self, event):
        method, params = event.get("method"), event.get("params", {})
        if method == "Runtime.consoleAPICalled":
            self.console.append(
                " ".join(str(a.get("value", "")) for a in params["args"])
            )
        elif method == "Runtime.exceptionThrown":
            detail = params["exceptionDetails"]
            self.console.append(
                "EXCEPTION "
                + str(detail.get("exception", {}).get("description", detail["text"]))
            )
        elif method == "Page.frameNavigated" and "parentId" not in params["frame"]:
            self.navigations.append(params["frame"]["url"].split("#")[0])

    def _pump(self):
        while True:
            try:
                self._note(json.loads(self._ws.recv(timeout=1)))
            except TimeoutError:
                continue
            except (ConnectionClosed, OSError):
                return

    def open(self, url: str, init_script: str | None = None) -> None:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.cdp_port}/json/version", timeout=5
        ) as resp:
            browser_ws = json.load(resp)["webSocketDebuggerUrl"]
        self._ws = ws_connect(browser_ws, max_size=None)
        target = self._call("Target.createTarget", {"url": "about:blank"})["targetId"]
        session = self._call(
            "Target.attachToTarget", {"targetId": target, "flatten": True}
        )["sessionId"]
        self._call("Runtime.enable", session=session)
        self._call("Page.enable", session=session)
        if init_script:  # runs in every new document, before the page's own scripts
            self._call(
                "Page.addScriptToEvaluateOnNewDocument",
                {"source": init_script},
                session=session,
            )
        self._call("Page.navigate", {"url": url}, session=session)
        self.pump = threading.Thread(target=self._pump, daemon=True)
        self.pump.start()

    def close(self) -> None:
        if self._ws is not None:
            self._ws.close()


class _BrowserAdb(FakeAdb):
    """FakeAdb whose accepted ``am start`` opens its exact ``-d`` URL in real Chrome."""

    def __init__(
        self,
        tab: _Tab,
        *,
        fragment_token: str | None = None,
        init_script: str | None = None,
    ) -> None:
        super().__init__()
        self.tab = tab
        self.init_script = init_script
        self.fragment_token = fragment_token  # fault injection: replaces the real token
        self.launched: list[str] = []
        self.monitor: asyncio.Task | None = None

    async def stop_monitor(self) -> None:
        """Cancel the connect monitor while ``subprocess.run`` is still faked."""
        if self.monitor is not None:
            self.monitor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.monitor

    def __call__(self, args, **kwargs):
        result = super().__call__(args, **kwargs)  # validates and records the command
        if args[:2] == ["adb", "shell"] and args[2].startswith("am start"):
            url = _intent_url(args[2])
            if self.fragment_token is not None:
                head, _, fragment = url.partition("#")
                route = fragment.split("?", 1)[0]
                url = (
                    f"{head}#{route}?{urlencode({'controlToken': self.fragment_token})}"
                )
            self.launched.append(url)
            self.tab.open(url, self.init_script)
        return result


@pytest.fixture(scope="session")
def app_mock_build() -> Path:
    """Build the MockCloudXR web client once, from this checkout, with a clean cache."""
    client = repo_root() / "deps" / "cloudxr" / "webxr_client"
    shutil.rmtree(client / "node_modules" / ".cache" / "webpack", ignore_errors=True)
    subprocess.run(
        ["npm", "run", "build:app-mock"], cwd=client, check=True, capture_output=True
    )
    build = client / "build-app-mock"
    assert (build / "index.html").is_file()
    return build


@pytest.fixture
def static_server(app_mock_build):
    """Serve the build on a free loopback port, recording requests."""
    server = ThreadingHTTPServer(
        ("localhost", 0), partial(_Recorder, directory=str(app_mock_build))
    )
    server.seen = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def chrome(tmp_path):
    """A real Chrome with its own profile and CDP port."""
    if not os.path.exists(_CHROME):
        pytest.fail(f"Chrome not found at {_CHROME}; set PLAYWRIGHT_CHROME_PATH")
    port = _free_port()
    proc = subprocess.Popen(
        [
            _CHROME,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={tmp_path / 'profile'}",
            *_CHROME_ARGS,
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_http(f"http://127.0.0.1:{port}/json/version", 15)
        yield port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def token(request, monkeypatch) -> str:
    """The hub's control token: ``request.param`` as ``CONTROL_TOKEN``, or generated if None.

    Fixed before the hub starts, since the hub resolves it once at start-up.
    """
    if request.param is None:
        monkeypatch.delenv("CONTROL_TOKEN", raising=False)
        monkeypatch.setattr(env_module, "_generated_control_token", None)
    else:
        monkeypatch.setenv("CONTROL_TOKEN", request.param)
    return env_module.resolve_control_token()


@pytest.fixture
async def hub(tmp_path, monkeypatch, token):
    """The real ``wss.run`` with the OOB hub; yields a function reading its HTTP state."""
    from cloudxr_py_test_ns.wss import run as wss_run  # noqa: PLC0415

    monkeypatch.setenv("CXR_INSTALL_DIR", str(tmp_path / "install"))
    # Hub only: this test drives run_oob_connect itself.
    monkeypatch.setenv("TELEOP_OOB_HUB_ONLY", "1")
    port = _free_port()
    stop = asyncio.get_running_loop().create_future()
    listening = asyncio.Event()
    task = asyncio.create_task(
        wss_run(
            None,
            stop,
            backend_port=_free_port(),
            proxy_port=port,
            setup_oob=True,
            on_listening=listening.set,
        )
    )
    ready = asyncio.create_task(listening.wait())

    def fetch() -> list[dict]:
        request = urllib.request.Request(
            f"https://localhost:{port}/api/oob/v1/state",
            headers={"X-Control-Token": token},
        )
        insecure = ssl.create_default_context()
        insecure.check_hostname = False
        insecure.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(request, timeout=5, context=insecure) as resp:
            return json.load(resp)["headsets"]

    async def headsets() -> list[dict]:
        return await asyncio.to_thread(fetch)

    headsets.port = port
    try:
        await asyncio.wait(
            {ready, task}, timeout=30, return_when=asyncio.FIRST_COMPLETED
        )
        if task.done():  # startup failed: surface why, not a bare timeout
            task.result()
            raise AssertionError("wss.run exited before listening")
        assert ready.done(), "wss.run did not start listening within 30s"
        yield headsets
    finally:
        ready.cancel()
        if not task.done():
            stop.set_result(None)
            await asyncio.wait_for(task, timeout=15)


@contextlib.asynccontextmanager
async def _session(
    monkeypatch, chrome, static_server, hub, *, fragment_token=None, init_script=None
):
    """Run the real ``run_oob_connect`` against Chrome; yield the adapter, then tear down."""
    monkeypatch.setenv("TELEOP_CLIENT_ROUTE", "/sim/isaaclab")
    monkeypatch.setenv(
        "TELEOP_WEB_CLIENT_BASE", f"http://localhost:{static_server.server_port}"
    )
    monkeypatch.setattr(adb_module, "_CDP_LOCAL_PORT", chrome)

    tab = _Tab(chrome)
    adb = _BrowserAdb(tab, fragment_token=fragment_token, init_script=init_script)
    try:
        with patch("subprocess.run", side_effect=adb):
            adb.monitor = await adb_module.run_oob_connect(
                resolved_port=hub.port, timeout=30.0, usb_local=True
            )
            try:
                yield adb
            finally:
                # Its cleanup issues `adb forward --remove`, which must hit the fake.
                await adb.stop_monitor()
    finally:
        tab.close()


def _tab_url(cdp_port: int) -> str:
    """URL of the page tab, from Chrome's own ``/json`` list."""
    with urllib.request.urlopen(f"http://127.0.0.1:{cdp_port}/json", timeout=5) as resp:
        pages = [t for t in json.load(resp) if t.get("type") == "page"]
    return next(t["url"] for t in pages if "oobEnable=1" in t["url"])


@pytest.mark.oob_browser
@pytest.mark.skipif(
    os.environ.get("RUN_OOB_BROWSER_TESTS") != "1",
    reason="set RUN_OOB_BROWSER_TESTS=1 to run the real-browser token tests",
)
class TestFragmentTokenInRealBrowser:
    """Launch, authenticate, clean the URL; and reject a wrong token.

    Reload persistence is covered by the jest and Playwright tests, not here.
    """

    @pytest.mark.parametrize(
        "token", [None, _AWKWARD_TOKEN], ids=["generated", "awkward"], indirect=True
    )
    async def test_launch_authenticates_via_fragment(
        self, monkeypatch, chrome, static_server, hub, token
    ):
        async with _session(monkeypatch, chrome, static_server, hub) as adb:
            expected = token

            # The URL handed to `am start`: token only in the fragment, after the route.
            (url,) = adb.launched
            parts = urlsplit(url)
            assert "controlToken" not in parts.query
            route, _, fragment_query = parts.fragment.partition("?")
            assert route == "/sim/isaaclab"
            assert parse_qs(fragment_query)["controlToken"] == [expected]

            # The real hub accepted the real client, so the fragment token was lifted.
            await _wait_for(hub, "an authenticated headset registration")

            # The tab's URL no longer carries the token but keeps route and endpoint.
            page_url = _tab_url(chrome)
            assert expected not in page_url and "controlToken" not in page_url
            assert urlsplit(page_url).fragment == "/sim/isaaclab"
            assert f"port={hub.port}" in urlsplit(page_url).query

            # The mock session really connected, not just registered.
            async def streaming():
                return [h for h in await hub() if h["streaming"]]

            await _wait_for(streaming, "the mock session to report streaming")

            # Stopping the monitor under the fake runs its `adb forward --remove`.
            await adb.stop_monitor()
            assert any(c[1:3] == ["forward", "--remove"] for c in adb.calls)

            # No HTTP request or Referer ever carried the token, raw or encoded.
            seen = " ".join(f"{path} {referer}" for path, referer in static_server.seen)
            assert expected not in seen
            assert urlencode({"t": expected})[2:] not in seen

    @pytest.mark.parametrize("token", [_AWKWARD_TOKEN], indirect=True)
    async def test_wrong_fragment_token_is_rejected(
        self, monkeypatch, chrome, static_server, hub, token
    ):
        async with _session(
            monkeypatch, chrome, static_server, hub, fragment_token="not-the-hub-token"
        ) as adb:
            (url,) = adb.launched
            sent = parse_qs(urlsplit(url).fragment.split("?", 1)[1])["controlToken"]
            assert sent == ["not-the-hub-token"] != [token]

            async def rejected():
                return any("Terminal close (1008" in line for line in adb.tab.console)

            await _wait_for(rejected, "the client's 1008 auth rejection in the console")
            assert await hub() == []

    @pytest.mark.parametrize("token", [None], indirect=True)
    async def test_stream_attach_timeout_from_the_bookmark_reaches_the_page(
        self, monkeypatch, chrome, static_server, hub, token
    ):
        """A timeout set in the environment travels in the bookmark and fires."""
        monkeypatch.setenv("TELEOP_CLIENT_STREAM_ATTACH_TIMEOUT_MS", "3000")
        async with _session(
            monkeypatch,
            chrome,
            static_server,
            hub,
            # Hold the mock session in Connecting so the attach deadline expires.
            init_script="window.__mockCloudXRConnectDelayMs = 3600000;",
        ) as adb:
            (url,) = adb.launched
            assert parse_qs(urlsplit(url).query)["streamAttachTimeoutMs"] == ["3000"]

            async def timed_out():
                return any(
                    "stream did not attach within 3000ms" in line
                    for line in adb.tab.console
                )

            await _wait_for(timed_out, "the client's stream-attach timeout", timeout=20)

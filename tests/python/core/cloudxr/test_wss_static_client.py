# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ``--host-client`` static routes, certificate page, and ``/connect/`` helper in ``wss``.

Run from this directory (after ``pip install pytest``)::

    pytest -q

No CloudXR runtime, TLS, or ``isaaccapture`` install required — ``conftest.py`` adds
``src/core/cloudxr/python`` to ``sys.path``.
"""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path

import pytest

from cloudxr_py_test_ns.wss import _make_http_handler


class FakeRequest:
    """Minimal stand-in for the websockets request passed to the HTTP handler."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.headers: dict[str, str] = {}


@pytest.fixture
def static_dir(tmp_path: Path) -> Path:
    (tmp_path / "index.html").write_text('<script src="bundle.js"></script>')
    (tmp_path / "bundle.js").write_text("// bundle")
    return tmp_path


async def _get(static_dir: Path, path: str):
    handler = _make_http_handler("localhost", 49100, static_dir=static_dir)
    return await handler(None, FakeRequest(path))


@pytest.mark.asyncio
async def test_client_without_trailing_slash_redirects(static_dir: Path) -> None:
    response = await _get(static_dir, "/client")
    assert response.status_code == HTTPStatus.FOUND
    assert response.headers["Location"] == "/client/"


@pytest.mark.asyncio
async def test_client_redirect_preserves_query(static_dir: Path) -> None:
    response = await _get(static_dir, "/client?showVersion=1")
    assert response.status_code == HTTPStatus.FOUND
    assert response.headers["Location"] == "/client/?showVersion=1"


@pytest.mark.asyncio
async def test_client_directory_serves_index(static_dir: Path) -> None:
    response = await _get(static_dir, "/client/")
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/html; charset=utf-8"
    assert b"bundle.js" in response.body


@pytest.mark.asyncio
async def test_bundle_resolves_under_client_prefix(static_dir: Path) -> None:
    response = await _get(static_dir, "/client/bundle.js")
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/javascript; charset=utf-8"
    assert response.headers["Cache-Control"] == "no-store"


@pytest.mark.asyncio
async def test_webxr_controller_profile_assets_are_served(static_dir: Path) -> None:
    asset = (
        static_dir
        / "npm/@webxr-input-profiles/assets@1.0.0/dist/profiles/profilesList.json"
    )
    asset.parent.mkdir(parents=True)
    asset.write_text('{"profiles": []}')

    response = await _get(static_dir, f"/client/{asset.relative_to(static_dir)}")
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/json"
    assert response.body == b'{"profiles": []}'


@pytest.mark.asyncio
async def test_unknown_client_asset_is_404(static_dir: Path) -> None:
    response = await _get(static_dir, "/client/secrets.env")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_certificate_page_offers_fixed_manual_helper(monkeypatch) -> None:
    monkeypatch.setenv("TELEOP_WEB_CLIENT_BASE", "https://[bad")
    monkeypatch.setenv("TELEOP_CLIENT_RECONNECT_ENABLED", "invalid")
    request = FakeRequest("/")
    request.headers["Host"] = "[invalid"
    response = await _make_http_handler("localhost", 49100)(None, request)
    assert response.status_code == 200
    assert b"Certificate Accepted" in response.body
    assert b"close this tab and return to the web client" in response.body
    assert b'href="/connect/"' in response.body
    assert b"<script>" not in response.body


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/connect", "/connect/", "/connect/?ignored=1"])
@pytest.mark.parametrize("host_client", [False, True])
async def test_manual_helper_is_available_with_or_without_local_client(
    static_dir: Path, monkeypatch: pytest.MonkeyPatch, path: str, host_client: bool
) -> None:
    monkeypatch.setenv("TELEOP_WEB_CLIENT_BASE", "https://[bad")
    monkeypatch.setenv("TELEOP_CLIENT_RECONNECT_ENABLED", "invalid")
    handler = _make_http_handler(
        "localhost", 49100, static_dir=static_dir if host_client else None
    )
    response = await handler(None, FakeRequest(path))
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/html; charset=utf-8"
    assert b"Manual VR connection" in response.body
    assert b"Open NVIDIA-hosted client" in response.body
    assert b"window.location.hostname" in response.body
    assert b"window.location.port" in response.body
    assert b"https://nvidia.github.io/IsaacCapture/client/main/" in response.body
    assert b"<noscript>" in response.body
    assert b"launcher-provided client link" in response.body


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/connect/extra", "/connectx"])
async def test_other_paths_keep_certificate_confirmation(path: str) -> None:
    response = await _make_http_handler("localhost", 49100)(None, FakeRequest(path))
    assert b"Certificate Accepted" in response.body
    assert b"<script>" not in response.body


@pytest.mark.asyncio
async def test_helper_path_keeps_websocket_upgrade_behavior() -> None:
    request = FakeRequest("/connect/")
    request.headers["Upgrade"] = "websocket"
    assert await _make_http_handler("localhost", 49100)(None, request) is None

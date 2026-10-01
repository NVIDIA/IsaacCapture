# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ``--host-client`` static routes in ``wss``.

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
@pytest.mark.parametrize(
    ("authority", "host", "port"),
    [
        ("192.168.0.201:48322", "192.168.0.201", "48322"),
        ("robot.local:50000", "robot.local", "50000"),
        ("robot.local", "robot.local", "443"),
        ("[2001:db8::1]:48322", "2001:db8::1", "48322"),
    ],
)
async def test_certificate_link_prefills_browser_address(
    monkeypatch, authority, host, port
) -> None:
    from html import unescape
    from urllib.parse import parse_qs, urlparse

    monkeypatch.delenv("TELEOP_WEB_CLIENT_BASE", raising=False)
    monkeypatch.setattr(
        "cloudxr_py_test_ns.wss.default_web_client_origin",
        lambda: "https://nvidia.github.io/IsaacCapture/client/v1.2.3/",
    )
    request = FakeRequest("/")
    request.headers["Host"] = authority
    handler = _make_http_handler("localhost", 49100)
    response = await handler(None, request)
    html = response.body.decode()
    link = unescape(html.split('href="')[1].split('"')[0])
    parsed = urlparse(link)
    assert parsed.path == "/IsaacCapture/client/v1.2.3/"
    assert parse_qs(parsed.query) == {
        "serverIP": [host],
        "port": [port],
        "serverType": ["manual"],
    }
    assert "Certificate Accepted" in html
    assert "close this tab and return to the web client" in html


@pytest.mark.asyncio
async def test_certificate_link_prefers_hosted_client(static_dir, monkeypatch) -> None:
    monkeypatch.delenv("TELEOP_WEB_CLIENT_BASE", raising=False)
    handler = _make_http_handler("localhost", 49100, static_dir=static_dir)
    request = FakeRequest("/")
    request.headers["Host"] = "robot.local:48322"
    response = await handler(None, request)
    assert (
        b'href="/client/?serverIP=robot.local&amp;port=48322&amp;serverType=manual"'
        in response.body
    )


def test_certificate_link_respects_override_and_preserves_query_and_route(monkeypatch):
    from urllib.parse import parse_qs, urlparse
    from cloudxr_py_test_ns.wss import _cert_client_url

    monkeypatch.setenv(
        "TELEOP_WEB_CLIENT_BASE",
        "https://client.example/?codec=h264&serverIP=old&port=123&serverType=nvcf#/custom",
    )
    link = _cert_client_url("robot.local:48322", host_client=True)
    parsed = urlparse(link)
    assert parsed.netloc == "client.example"
    assert parsed.fragment == "/custom"
    assert parse_qs(parsed.query) == {
        "codec": ["h264"],
        "serverIP": ["robot.local"],
        "port": ["48322"],
        "serverType": ["manual"],
    }


@pytest.mark.parametrize(
    "authority", ["", "robot:invalid", "[broken", "user@robot", "robot/path"]
)
def test_certificate_link_omitted_for_invalid_authority(authority):
    from cloudxr_py_test_ns.wss import _cert_client_url, _cert_html

    assert b"href=" not in _cert_html(_cert_client_url(authority, host_client=False))


def test_certificate_link_escapes_html():
    from cloudxr_py_test_ns.wss import _cert_html

    html = _cert_html('https://client.example/?a="<test>"&b=2')
    assert b"a=&quot;&lt;test&gt;&quot;&amp;b=2" in html

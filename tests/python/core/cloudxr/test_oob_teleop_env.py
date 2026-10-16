# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`oob_teleop_env` (bookmark URLs, env-driven defaults, LAN helpers)."""

from __future__ import annotations

import logging
from urllib.parse import parse_qs, urlparse

import cloudxr_py_test_ns.oob_teleop_env as oob_teleop_env_under_test
import pytest

from cloudxr_py_test_ns.oob_teleop_env import (
    FALLBACK_WEB_CLIENT_ORIGIN,
    TELEOP_WEB_CLIENT_BASE_ENV,
    TELEOP_WEB_CLIENT_STATIC_DIR_ENV,
    USB_BACKEND_DEFAULT_PORT,
    USB_TURN_DEFAULT_PORT,
    WEB_CLIENT_BASE,
    WSS_PROXY_DEFAULT_PORT,
    build_headset_bookmark_url,
    client_ui_fields_from_env,
    default_initial_stream_config,
    default_web_client_origin,
    guess_lan_ipv4,
    versioned_web_client_url,
    print_hosted_client_line,
    print_oob_hub_startup_banner,
    require_web_client_static_dir,
    resolve_lan_host_for_oob,
    usb_backend_port,
    usb_turn_port,
    web_client_base_override_from_env,
    wss_proxy_port,
)
from cloudxr_py_test_ns.oob_teleop_hub import OOB_WS_PATH


@pytest.fixture
def clear_teleop_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all teleop environment variables so tests start from a clean slate."""
    keys = (
        "PROXY_PORT",
        "USB_BACKEND_PORT",
        "USB_TURN_PORT",
        "TELEOP_STREAM_SERVER_IP",
        "TELEOP_STREAM_PORT",
        "TELEOP_CLIENT_CODEC",
        "TELEOP_CLIENT_PANEL_HIDDEN_AT_START",
        "TELEOP_CLIENT_RECONNECT_MAX_ATTEMPTS",
        "TELEOP_CLIENT_RECONNECT_DELAY_MS",
        "TELEOP_CLIENT_STREAM_ATTACH_TIMEOUT_MS",
        "TELEOP_CLIENT_WARMUP_BEGIN_TIMEOUT_MS",
        "TELEOP_CLIENT_WARMUP_END_TIMEOUT_MS",
        "TELEOP_CLIENT_ROUTE",
        "TELEOP_WEB_CLIENT_BASE",
        "TELEOP_PROXY_HOST",
        "CONTROL_TOKEN",
        TELEOP_WEB_CLIENT_STATIC_DIR_ENV,
    )
    for k in keys:
        monkeypatch.delenv(k, raising=False)


def test_wss_proxy_port_default(clear_teleop_env: None) -> None:
    """WSS proxy port returns the compile-time default when PROXY_PORT is unset."""
    assert wss_proxy_port() == WSS_PROXY_DEFAULT_PORT


def test_wss_proxy_port_from_env(
    clear_teleop_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PROXY_PORT env var overrides the default WSS proxy port."""
    monkeypatch.setenv("PROXY_PORT", "50000")
    assert wss_proxy_port() == 50000


def test_usb_backend_port_default(clear_teleop_env: None) -> None:
    """USB backend port returns the compile-time default when USB_BACKEND_PORT is unset."""
    assert usb_backend_port() == USB_BACKEND_DEFAULT_PORT


def test_usb_backend_port_from_env(
    clear_teleop_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """USB_BACKEND_PORT env var overrides the default backend port."""
    monkeypatch.setenv("USB_BACKEND_PORT", "49200")
    assert usb_backend_port() == 49200


def test_usb_backend_port_invalid_env_raises(
    clear_teleop_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-integer USB_BACKEND_PORT raises ValueError mentioning the variable name."""
    monkeypatch.setenv("USB_BACKEND_PORT", "not-a-port")
    with pytest.raises(ValueError, match="USB_BACKEND_PORT"):
        usb_backend_port()


def test_usb_turn_port_default(clear_teleop_env: None) -> None:
    """USB TURN port returns the compile-time default when USB_TURN_PORT is unset."""
    assert usb_turn_port() == USB_TURN_DEFAULT_PORT


def test_usb_turn_port_from_env(
    clear_teleop_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """USB_TURN_PORT env var overrides the default TURN port."""
    monkeypatch.setenv("USB_TURN_PORT", "5349")
    assert usb_turn_port() == 5349


def test_usb_turn_port_invalid_env_raises(
    clear_teleop_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-integer USB_TURN_PORT raises ValueError mentioning the variable name."""
    monkeypatch.setenv("USB_TURN_PORT", "not-a-port")
    with pytest.raises(ValueError, match="USB_TURN_PORT"):
        usb_turn_port()


def test_web_client_base_override_from_env(
    clear_teleop_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TELEOP_WEB_CLIENT_BASE returns None when unset and the stripped URL when set."""
    assert web_client_base_override_from_env() is None
    monkeypatch.setenv(TELEOP_WEB_CLIENT_BASE_ENV, "  https://example.test/app  ")
    assert web_client_base_override_from_env() == "https://example.test/app"


def test_versioned_web_client_url_exact_tag() -> None:
    """Clean MAJOR.MINOR.PATCH release maps to the per-tag GitHub Pages client URL."""
    # A clean MAJOR.MINOR.PATCH release maps to the per-tag client.
    assert (
        versioned_web_client_url("1.2.3")
        == "https://nvidia.github.io/IsaacCapture/client/v1.2.3/"
    )


def test_versioned_web_client_url_prerelease_uses_release_line() -> None:
    """rc/dev pre-release versions map to the release-line URL, not a per-tag URL."""
    # rc / dev builds fall to the release line, not a per-tag client.
    assert (
        versioned_web_client_url("1.3.9rc1")
        == "https://nvidia.github.io/IsaacCapture/client/release-1.3.x/"
    )
    assert (
        versioned_web_client_url("1.3.0.dev5")
        == "https://nvidia.github.io/IsaacCapture/client/release-1.3.x/"
    )


def test_versioned_web_client_url_unparseable_falls_back_to_base() -> None:
    """Unparseable version strings fall back to the generic WEB_CLIENT_BASE URL."""
    # No leading MAJOR.MINOR -> generic client root (site redirects to stable).
    assert versioned_web_client_url("unknown") == WEB_CLIENT_BASE


def test_default_web_client_origin_uses_installed_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """default_web_client_origin returns a versioned URL when the package version is known."""
    import cloudxr_py_test_ns.oob_teleop_env as mod

    monkeypatch.setattr(mod, "version", lambda _name: "1.4.0", raising=False)
    assert (
        default_web_client_origin()
        == "https://nvidia.github.io/IsaacCapture/client/v1.4.0/"
    )


def test_default_web_client_origin_falls_back_when_version_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """default_web_client_origin falls back to FALLBACK_WEB_CLIENT_ORIGIN when the package is not found."""
    import cloudxr_py_test_ns.oob_teleop_env as mod
    from importlib.metadata import PackageNotFoundError

    def _raise(_name: str) -> str:
        """Stub that always raises PackageNotFoundError."""
        raise PackageNotFoundError(_name)

    monkeypatch.setattr(mod, "version", _raise, raising=False)
    assert default_web_client_origin() == FALLBACK_WEB_CLIENT_ORIGIN


def test_build_headset_bookmark_url_minimal() -> None:
    """Minimal stream config produces a valid bookmark URL with oobEnable, serverIP, and port."""
    u = build_headset_bookmark_url(
        web_client_base="https://h.test/",
        stream_config={"serverIP": "10.0.0.1", "port": 48322},
    )
    assert urlparse(u).hostname == "h.test"
    q = parse_qs(urlparse(u).query)
    assert q["oobEnable"] == ["1"]
    assert q["serverIP"] == ["10.0.0.1"]
    assert q["port"] == ["48322"]


def test_build_headset_bookmark_url_with_token() -> None:
    """control_token is included as controlToken in the bookmark URL query string."""
    u = build_headset_bookmark_url(
        web_client_base="https://h.test/",
        stream_config={"serverIP": "10.0.0.1", "port": 48322},
        control_token="secret123",
    )
    q = parse_qs(urlparse(u).query)
    assert q["controlToken"] == ["secret123"]


def test_build_headset_bookmark_url_with_codec() -> None:
    """codec from stream_config is forwarded as a query param in the bookmark URL."""
    u = build_headset_bookmark_url(
        web_client_base="https://h.test/",
        stream_config={"serverIP": "10.0.0.1", "port": 48322, "codec": "av1"},
    )
    q = parse_qs(urlparse(u).query)
    assert q["codec"] == ["av1"]


def test_build_headset_bookmark_url_panel_hidden() -> None:
    """panelHiddenAtStart=True is serialised as the string "true" in the bookmark URL."""
    u = build_headset_bookmark_url(
        web_client_base="https://h.test/",
        stream_config={
            "serverIP": "10.0.0.1",
            "port": 48322,
            "panelHiddenAtStart": True,
        },
    )
    q = parse_qs(urlparse(u).query)
    assert q["panelHiddenAtStart"] == ["true"]


def test_build_headset_bookmark_url_with_reconnect_policy() -> None:
    """The host-owned retry budget reaches the browser on its initial page load."""
    u = build_headset_bookmark_url(
        web_client_base="https://h.test/",
        stream_config={
            "serverIP": "10.0.0.1",
            "port": 48322,
            "reconnectEnabled": False,
            "reconnectMaxAttempts": 5,
            "reconnectDelayMs": 750,
        },
    )
    q = parse_qs(urlparse(u).query)
    assert q["reconnectEnabled"] == ["false"]
    assert q["reconnectMaxAttempts"] == ["5"]
    assert q["reconnectDelayMs"] == ["750"]


def test_build_headset_bookmark_url_requires_server_ip() -> None:
    """stream_config without serverIP raises ValueError."""
    with pytest.raises(ValueError, match="serverIP"):
        build_headset_bookmark_url(
            web_client_base="https://h.test/",
            stream_config={"port": 48322},
        )


def test_build_headset_bookmark_url_default_route(clear_teleop_env: None) -> None:
    """No fragment is added to the bookmark URL when TELEOP_CLIENT_ROUTE is unset."""
    u = build_headset_bookmark_url(
        web_client_base="https://h.test/",
        stream_config={"serverIP": "10.0.0.1", "port": 48322},
    )
    # No fragment by default — the WebXR client picks its own landing route.
    assert urlparse(u).fragment == ""
    assert "#" not in u


def test_build_headset_bookmark_url_route_override(
    clear_teleop_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TELEOP_CLIENT_ROUTE is used as the URL fragment in the bookmark URL."""
    monkeypatch.setenv("TELEOP_CLIENT_ROUTE", "/sim/cube")
    u = build_headset_bookmark_url(
        web_client_base="https://h.test/",
        stream_config={"serverIP": "10.0.0.1", "port": 48322},
    )
    assert urlparse(u).fragment == "/sim/cube"


def test_build_headset_bookmark_url_route_strips_leading_hash(
    clear_teleop_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leading ``#`` in TELEOP_CLIENT_ROUTE is stripped to avoid a double ``##`` fragment."""
    # Operators may write ``#/foo`` out of habit; we should produce ``#/foo``,
    # not ``##/foo``.
    monkeypatch.setenv("TELEOP_CLIENT_ROUTE", "#/foo/bar")
    u = build_headset_bookmark_url(
        web_client_base="https://h.test/",
        stream_config={"serverIP": "10.0.0.1", "port": 48322},
    )
    assert urlparse(u).fragment == "/foo/bar"
    assert "##" not in u


def test_build_headset_bookmark_url_route_empty_suppresses_fragment(
    clear_teleop_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty TELEOP_CLIENT_ROUTE produces no fragment in the bookmark URL."""
    monkeypatch.setenv("TELEOP_CLIENT_ROUTE", "")
    u = build_headset_bookmark_url(
        web_client_base="https://h.test/",
        stream_config={"serverIP": "10.0.0.1", "port": 48322},
    )
    assert urlparse(u).fragment == ""
    assert "#" not in u


def test_client_ui_fields_from_env_empty(clear_teleop_env: None) -> None:
    """Client and host receive the same default bounded-retry settings.

    reconnectEnabled is absent: OOB never sets it, so it is entirely the client's
    own concern (its own default, or a URL param the user supplies directly),
    unlike OOB's own unconditional tab-relaunch retry.
    """
    assert client_ui_fields_from_env() == {
        "reconnectMaxAttempts": 10,
        "reconnectDelayMs": 3000,
    }


def test_client_ui_fields_from_env_reconnect_overrides(
    clear_teleop_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TELEOP_CLIENT_RECONNECT_MAX_ATTEMPTS", "5")
    monkeypatch.setenv("TELEOP_CLIENT_RECONNECT_DELAY_MS", "750")
    assert client_ui_fields_from_env() == {
        "reconnectMaxAttempts": 5,
        "reconnectDelayMs": 750,
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TELEOP_CLIENT_RECONNECT_MAX_ATTEMPTS", "-1"),
        ("TELEOP_CLIENT_RECONNECT_DELAY_MS", "1.5"),
    ],
)
def test_client_ui_fields_reject_invalid_reconnect_values(
    clear_teleop_env: None,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        client_ui_fields_from_env()


def test_client_ui_fields_from_env_codec(
    clear_teleop_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TELEOP_CLIENT_CODEC is surfaced as the codec field."""
    monkeypatch.setenv("TELEOP_CLIENT_CODEC", "h265")
    fields = client_ui_fields_from_env()
    assert fields["codec"] == "h265"


def test_client_ui_fields_from_env_panel_hidden(
    clear_teleop_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TELEOP_CLIENT_PANEL_HIDDEN_AT_START=true sets panelHiddenAtStart to True."""
    monkeypatch.setenv("TELEOP_CLIENT_PANEL_HIDDEN_AT_START", "true")
    fields = client_ui_fields_from_env()
    assert fields["panelHiddenAtStart"] is True


def test_client_ui_fields_from_env_never_sets_reconnect_enabled(
    clear_teleop_env: None,
) -> None:
    """OOB never sets reconnectEnabled - there is no env var for it at all."""
    fields = client_ui_fields_from_env()
    assert "reconnectEnabled" not in fields


def test_client_ui_fields_from_env_reliability_timeouts(
    clear_teleop_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The five reconnect/timeout ms/count env vars are surfaced as integer fields."""
    monkeypatch.setenv("TELEOP_CLIENT_RECONNECT_MAX_ATTEMPTS", "5")
    monkeypatch.setenv("TELEOP_CLIENT_RECONNECT_DELAY_MS", "2500")
    monkeypatch.setenv("TELEOP_CLIENT_STREAM_ATTACH_TIMEOUT_MS", "90000")
    monkeypatch.setenv("TELEOP_CLIENT_WARMUP_BEGIN_TIMEOUT_MS", "8000")
    monkeypatch.setenv("TELEOP_CLIENT_WARMUP_END_TIMEOUT_MS", "20000")
    fields = client_ui_fields_from_env()
    assert fields["reconnectMaxAttempts"] == 5
    assert fields["reconnectDelayMs"] == 2500
    assert fields["streamAttachTimeoutMs"] == 90000
    assert fields["warmupBeginTimeoutMs"] == 8000
    assert fields["warmupEndTimeoutMs"] == 20000


def test_client_ui_fields_from_env_invalid_timeout_ignored(
    clear_teleop_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-integer timeout env var is silently ignored rather than raising."""
    monkeypatch.setenv("TELEOP_CLIENT_STREAM_ATTACH_TIMEOUT_MS", "not-a-number")
    fields = client_ui_fields_from_env()
    assert "streamAttachTimeoutMs" not in fields


def test_build_headset_bookmark_url_reconnect_enabled() -> None:
    """reconnectEnabled=True from stream_config is serialised as "true" in the bookmark URL."""
    u = build_headset_bookmark_url(
        web_client_base="https://h.test/",
        stream_config={
            "serverIP": "10.0.0.1",
            "port": 48322,
            "reconnectEnabled": True,
        },
    )
    q = parse_qs(urlparse(u).query)
    assert q["reconnectEnabled"] == ["true"]


def test_build_headset_bookmark_url_reliability_timeouts() -> None:
    """Reconnect/timeout numeric fields from stream_config are forwarded as query params."""
    u = build_headset_bookmark_url(
        web_client_base="https://h.test/",
        stream_config={
            "serverIP": "10.0.0.1",
            "port": 48322,
            "reconnectMaxAttempts": 5,
            "reconnectDelayMs": 2500,
            "streamAttachTimeoutMs": 90000,
            "warmupBeginTimeoutMs": 8000,
            "warmupEndTimeoutMs": 20000,
        },
    )
    q = parse_qs(urlparse(u).query)
    assert q["reconnectMaxAttempts"] == ["5"]
    assert q["reconnectDelayMs"] == ["2500"]
    assert q["streamAttachTimeoutMs"] == ["90000"]
    assert q["warmupBeginTimeoutMs"] == ["8000"]
    assert q["warmupEndTimeoutMs"] == ["20000"]


def test_default_initial_stream_config_defaults(clear_teleop_env: None) -> None:
    """default_initial_stream_config uses the supplied port and includes a serverIP."""
    cfg = default_initial_stream_config(48322)
    assert cfg["port"] == 48322
    assert "serverIP" in cfg


def test_default_initial_stream_config_env_override(
    clear_teleop_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TELEOP_STREAM_SERVER_IP and TELEOP_STREAM_PORT override automatic defaults."""
    monkeypatch.setenv("TELEOP_STREAM_SERVER_IP", "10.0.0.99")
    monkeypatch.setenv("TELEOP_STREAM_PORT", "50000")
    cfg = default_initial_stream_config(48322)
    assert cfg["serverIP"] == "10.0.0.99"
    assert cfg["port"] == 50000


def test_resolve_lan_host_from_env(
    clear_teleop_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TELEOP_PROXY_HOST is returned by resolve_lan_host_for_oob when set."""
    monkeypatch.setenv("TELEOP_PROXY_HOST", "10.0.0.42")
    assert resolve_lan_host_for_oob() == "10.0.0.42"


def test_guess_lan_ipv4_returns_string_or_none() -> None:
    """guess_lan_ipv4 returns a string IP address or None — never raises."""
    result = guess_lan_ipv4()
    assert result is None or isinstance(result, str)


def test_require_web_client_static_dir_default_downloads(
    clear_teleop_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Sync index.html, bundle.js, and bundle.emulator.js from the published client."""
    # Path.home() consults $HOME on POSIX but %USERPROFILE% on Windows
    # (ntpath.expanduser ignores $HOME when USERPROFILE is set). Patch both
    # so the redirect works on every platform.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    def fake_fetch(url: str, *, timeout: float = 120.0) -> bytes:
        """Return stub asset bytes keyed by URL suffix."""
        if url.endswith("index.html"):
            return b"<!doctype html><title>t</title>"
        if url.endswith("bundle.js"):
            return b"console.log(1);"
        if url.endswith("bundle.emulator.js"):
            return b"// emulator chunk"
        raise AssertionError(url)

    monkeypatch.setattr(
        oob_teleop_env_under_test,
        "_fetch_url_bytes",
        fake_fetch,
    )
    out = require_web_client_static_dir()
    expected = (tmp_path / ".cloudxr" / "static-client").resolve()
    assert out == expected
    assert (out / "index.html").read_bytes().startswith(b"<!doctype")
    assert b"console.log" in (out / "bundle.js").read_bytes()
    assert (out / "bundle.emulator.js").read_bytes() == b"// emulator chunk"


def test_require_web_client_static_dir_skips_optional_emulator(
    clear_teleop_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Older published clients without bundle.emulator.js still sync core assets."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    def fake_fetch(url: str, *, timeout: float = 120.0) -> bytes:
        """Return stub bytes for required assets; raise RuntimeError for the emulator bundle."""
        if url.endswith("index.html"):
            return b"<!doctype html><title>t</title>"
        if url.endswith("bundle.js"):
            return b"console.log(1);"
        if url.endswith("bundle.emulator.js"):
            raise RuntimeError("Could not download: not found")
        raise AssertionError(url)

    monkeypatch.setattr(
        oob_teleop_env_under_test,
        "_fetch_url_bytes",
        fake_fetch,
    )
    out = require_web_client_static_dir()
    assert (out / "index.html").is_file()
    assert (out / "bundle.js").is_file()
    assert not (out / "bundle.emulator.js").exists()


def test_require_web_client_static_dir_ok(
    clear_teleop_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Pre-populated static dir is accepted without any network download."""
    (tmp_path / "index.html").write_text(
        "<!doctype html><title>x</title>", encoding="utf-8"
    )
    (tmp_path / "bundle.js").write_text("// bundle", encoding="utf-8")
    monkeypatch.setenv(TELEOP_WEB_CLIENT_STATIC_DIR_ENV, str(tmp_path))
    assert require_web_client_static_dir() == tmp_path.resolve()


def test_oob_static_dir_rejects_cached_bundle_without_health_protocol(
    clear_teleop_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    (tmp_path / "index.html").write_text("<script src='bundle.js'></script>")
    bundle = tmp_path / "bundle.js"
    bundle.write_bytes(b"old WebXR client")
    monkeypatch.setenv(TELEOP_WEB_CLIENT_STATIC_DIR_ENV, str(tmp_path))

    with pytest.raises(RuntimeError, match="lacks the OOB healthProbe"):
        require_web_client_static_dir(require_health_probe=True)

    bundle.write_bytes(b'"healthProbe" "healthReport"')
    assert (
        require_web_client_static_dir(require_health_probe=True) == tmp_path.resolve()
    )


def test_require_web_client_static_dir_not_a_directory(
    clear_teleop_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """TELEOP_WEB_CLIENT_STATIC_DIR pointing to a file raises RuntimeError."""
    bogus = tmp_path / "file_not_dir"
    bogus.write_text("x", encoding="utf-8")
    monkeypatch.setenv(TELEOP_WEB_CLIENT_STATIC_DIR_ENV, str(bogus))
    with pytest.raises(RuntimeError, match="not a directory"):
        require_web_client_static_dir()


def test_print_oob_hub_startup_banner(
    clear_teleop_env: None, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Startup banner includes the OOB control path and the LAN host IP."""
    monkeypatch.setenv("TELEOP_PROXY_HOST", "10.0.0.1")
    print_oob_hub_startup_banner(lan_host="10.0.0.1")
    out = capsys.readouterr().out
    assert "OOB TELEOP" in out
    assert "10.0.0.1" in out
    assert OOB_WS_PATH in out


def test_startup_banner_hides_control_and_turn_credentials(
    clear_teleop_env: None, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setenv("CONTROL_TOKEN", "control-canary-123")
    monkeypatch.setattr(
        oob_teleop_env_under_test, "USB_TURN_CREDENTIAL", "turn-canary-456"
    )

    print_oob_hub_startup_banner(lan_host="127.0.0.1", usb_local=True)
    out = capsys.readouterr().out
    assert "control-canary-123" not in out
    assert "turn-canary-456" not in out
    assert "controlToken=<REDACTED>" in out
    assert "turnCredential=<REDACTED>" in out


def test_oob_progress_logs_at_info(
    monkeypatch: pytest.MonkeyPatch, capsys, caplog
) -> None:
    """With logging on, progress is an INFO record on the module logger, not a print."""
    monkeypatch.delenv("ISAACCAPTURE_LOGGING", raising=False)
    with caplog.at_level(logging.INFO, logger="isaaccapture.cloudxr.oob_teleop_env"):
        oob_teleop_env_under_test.oob_progress("setup-oob", "teleop tab found")
    [record] = caplog.records
    assert record.name == "isaaccapture.cloudxr.oob_teleop_env"
    assert record.levelno == logging.INFO
    assert record.getMessage() == "[setup-oob] teleop tab found"
    assert capsys.readouterr().err == ""


def test_oob_progress_prints_when_logging_is_off(
    monkeypatch: pytest.MonkeyPatch, capsys, caplog
) -> None:
    """``ISAACCAPTURE_LOGGING=off`` keeps the cyan stderr line and logs nothing."""
    monkeypatch.setenv("ISAACCAPTURE_LOGGING", "off")
    with caplog.at_level(logging.INFO, logger="isaaccapture.cloudxr.oob_teleop_env"):
        oob_teleop_env_under_test.oob_progress("setup-oob", "teleop tab found")
    assert capsys.readouterr().err == "\033[36m[setup-oob]\033[0m teleop tab found\n"
    assert not caplog.records


# Host preflight (H5) ------------------------------------------------------


from unittest.mock import MagicMock, patch  # noqa: E402

from cloudxr_py_test_ns.oob_teleop_env import (  # noqa: E402
    _ufw_unallowed_ports,
    print_host_preflight_warnings,
)


@patch("cloudxr_py_test_ns.oob_teleop_env.subprocess.run")
def test_ufw_unallowed_ports_inactive_returns_none(mock_run: MagicMock) -> None:
    """ufw inactive → _ufw_unallowed_ports returns None (ufw check skipped)."""
    mock_run.return_value = MagicMock(returncode=0, stdout="Status: inactive\n")
    assert _ufw_unallowed_ports([48322]) is None


@patch(
    "cloudxr_py_test_ns.oob_teleop_env.subprocess.run", side_effect=FileNotFoundError()
)
def test_ufw_unallowed_ports_no_ufw_returns_none(mock_run: MagicMock) -> None:
    """ufw binary not on PATH → _ufw_unallowed_ports returns None."""
    assert _ufw_unallowed_ports([48322]) is None


@patch("cloudxr_py_test_ns.oob_teleop_env.subprocess.run")
def test_ufw_unallowed_ports_active_with_allow(mock_run: MagicMock) -> None:
    """ufw active with an ALLOW rule for the port → returns an empty list."""
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=(
            "Status: active\n\n"
            "To                         Action      From\n"
            "--                         ------      ----\n"
            "48322/tcp                  ALLOW       Anywhere\n"
        ),
    )
    assert _ufw_unallowed_ports([48322]) == []


@patch("cloudxr_py_test_ns.oob_teleop_env.subprocess.run")
def test_ufw_unallowed_ports_active_missing_allow(mock_run: MagicMock) -> None:
    """ufw active but no ALLOW rule for the port → returns a list containing that port."""
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=(
            "Status: active\n\n"
            "To                         Action      From\n"
            "22/tcp                     ALLOW       Anywhere\n"
        ),
    )
    assert _ufw_unallowed_ports([48322]) == [48322]


def test_print_host_preflight_warnings_clean(capsys) -> None:
    """No warnings are printed when all ports are free and ufw is not blocking."""
    with (
        patch("cloudxr_py_test_ns.oob_teleop_env._port_in_use", return_value=False),
        patch(
            "cloudxr_py_test_ns.oob_teleop_env._ufw_unallowed_ports", return_value=None
        ),
    ):
        print_host_preflight_warnings(usb_local=False)
    assert capsys.readouterr().out == ""


def test_print_host_preflight_warnings_busy_port(capsys) -> None:
    """A busy proxy port prints a warning mentioning the port and PROXY_PORT."""
    with (
        patch("cloudxr_py_test_ns.oob_teleop_env._port_in_use", return_value=True),
        patch(
            "cloudxr_py_test_ns.oob_teleop_env._ufw_unallowed_ports", return_value=None
        ),
    ):
        print_host_preflight_warnings(usb_local=False)
    out = capsys.readouterr().out
    assert "already in use" in out
    assert "PROXY_PORT" in out


def test_print_host_preflight_warnings_ufw_blocks(capsys) -> None:
    """ufw blocking the proxy port prints a warning with the ufw allow command."""
    with (
        patch("cloudxr_py_test_ns.oob_teleop_env._port_in_use", return_value=False),
        patch(
            "cloudxr_py_test_ns.oob_teleop_env._ufw_unallowed_ports",
            return_value=[48322],
        ),
    ):
        print_host_preflight_warnings(usb_local=False)
    out = capsys.readouterr().out
    assert "ufw" in out
    assert "sudo ufw allow 48322/tcp" in out


def test_print_host_preflight_warnings_skips_ufw_in_usb_local(capsys) -> None:
    """ufw check is skipped entirely in --usb-local mode (loopback-only, no firewall concern)."""
    with (
        patch("cloudxr_py_test_ns.oob_teleop_env._port_in_use", return_value=False),
        patch(
            "cloudxr_py_test_ns.oob_teleop_env._ufw_unallowed_ports",
            return_value=[48322],
        ) as mock_ufw,
    ):
        print_host_preflight_warnings(usb_local=True)
    assert mock_ufw.call_count == 0
    assert capsys.readouterr().out == ""


def test_print_host_preflight_warnings_busy_port_usb_local_raises(capsys) -> None:
    """In --usb-local mode, a busy required port raises RuntimeError (fail-fast)."""
    # In --usb-local, every required loopback port is correctness-critical:
    # a busy port means streaming can't work, so we fail fast instead of
    # warn-and-continue.
    with (
        patch("cloudxr_py_test_ns.oob_teleop_env._port_in_use", return_value=True),
        patch(
            "cloudxr_py_test_ns.oob_teleop_env._ufw_unallowed_ports", return_value=None
        ),
        pytest.raises(RuntimeError, match="USB-local: required port"),
    ):
        print_host_preflight_warnings(usb_local=True)


def test_print_hosted_client_line_prints_url(clear_teleop_env: None) -> None:
    """Hosted client helper prints the prefix and URL."""
    from io import StringIO

    buf = StringIO()
    url = "https://10.0.0.5:48322/client/"
    print_hosted_client_line(url, file=buf)
    assert url in buf.getvalue()
    assert "web client:" in buf.getvalue()


def test_print_hosted_client_line_hides_credentials() -> None:
    from io import StringIO

    buf = StringIO()
    url = (
        "https://host.test/client?controlToken=control-canary-123"
        "&turnCredential=turn-canary-456&port=48322"
    )
    print_hosted_client_line(url, file=buf)
    out = buf.getvalue()
    assert "control-canary-123" not in out
    assert "turn-canary-456" not in out
    assert "controlToken=<REDACTED>" in out
    assert "turnCredential=<REDACTED>" in out
    assert "port=48322" in out

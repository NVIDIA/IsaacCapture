# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`webclient` (URL resolution, redaction, arg parsing, pretty output)."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest

from cloudxr_py_test_ns.oob_teleop_env import redact_control_token
from cloudxr_py_test_ns.webclient import (
    OobAdbError,
    _parse_args,
    main,
    print_summary,
    resolve_client_url,
)

_LAN = "cloudxr_py_test_ns.oob_teleop_adb.resolve_lan_host_for_oob"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep URL building independent of the developer's shell env."""
    for var in (
        "TELEOP_WEB_CLIENT_BASE",
        "TELEOP_CLIENT_ROUTE",
        "TELEOP_STREAM_PORT",
        "CONTROL_TOKEN",
        "PROXY_PORT",
        "NO_COLOR",
    ):
        monkeypatch.delenv(var, raising=False)


# URL resolution -------------------------------------------------------------


@patch(_LAN, return_value="10.0.0.1")
def test_resolve_default_is_not_oob(_mock_lan: MagicMock) -> None:
    """OOB is opt-in: the hub only runs under the launcher's --setup-oob."""
    url, source = resolve_client_url(None)
    assert "oobEnable" not in url
    assert "serverIP=10.0.0.1" in url
    assert "port=48322" in url
    assert "nvidia.github.io" in url
    assert "default" in source.lower()


@patch(_LAN, return_value="10.0.0.1")
def test_control_token_never_in_computed_url(
    _mock_lan: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The token authenticates hub operations, so it is meaningless without OOB."""
    monkeypatch.setenv("CONTROL_TOKEN", "sup3rs3cret")
    assert "controlToken" not in resolve_client_url(None)[0]


@patch(_LAN, return_value="10.0.0.1")
def test_resolve_base_override_appends_params(_mock_lan: MagicMock) -> None:
    url, source = resolve_client_url("https://192.168.1.5:8080")
    assert url.startswith("https://192.168.1.5:8080?")
    assert "serverIP=10.0.0.1" in url
    assert "nvidia.github.io" not in url
    assert "base" in source.lower()


@patch(_LAN, return_value="10.0.0.1")
def test_resolve_base_override_outranks_env(
    _mock_lan: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TELEOP_WEB_CLIENT_BASE", "https://from-env.example")
    url, _ = resolve_client_url("https://from-arg.example")
    assert url.startswith("https://from-arg.example?")
    assert "from-env" not in url


def test_resolve_full_url_is_verbatim() -> None:
    """A URL already carrying OOB params must not get a second set appended."""
    given = "https://example.test/client/?oobEnable=1&serverIP=10.0.0.9&port=48322"
    url, source = resolve_client_url(given)
    assert url == given
    assert url.count("oobEnable=") == 1
    assert url.count("serverIP=") == 1
    assert "verbatim" in source.lower()


# Token redaction ------------------------------------------------------------


def test_redact_token_masks_value() -> None:
    masked = redact_control_token("https://x.test/?controlToken=s3cret&port=48322")
    assert "s3cret" not in masked
    assert "controlToken=<REDACTED>" in masked
    assert "port=48322" in masked


def test_redact_oob_query_token_masks_value() -> None:
    masked = redact_control_token(
        "https://x.test/oob?token=s3cret&mode=1 and token=another-secret"
    )
    assert "s3cret" not in masked
    assert "another-secret" not in masked
    assert masked.count("token=<REDACTED>") == 2
    assert "mode=1" in masked


def test_redact_turn_credential_masks_value() -> None:
    masked = redact_control_token(
        "https://x.test/?turnCredential=turn-canary-456&port=48322"
    )
    assert "turn-canary-456" not in masked
    assert "turnCredential=<REDACTED>" in masked
    assert "port=48322" in masked


def test_redact_token_noop_without_token() -> None:
    url = "https://x.test/?oobEnable=1&port=48322"
    assert redact_control_token(url) == url


def test_summary_hides_token_but_url_keeps_it() -> None:
    """A pasted OOB URL can carry a token; it must not reach the terminal."""
    url, source = resolve_client_url(
        "https://x.test/?oobEnable=1&controlToken=sup3rs3cret"
        "&turnCredential=turn-canary-456&serverIP=10.0.0.9&port=48322"
    )
    assert "sup3rs3cret" in url

    buf = io.StringIO()
    print_summary(
        url=url,
        source=source,
        probe_device=False,
        stream=buf,
    )
    printed = buf.getvalue()
    assert "sup3rs3cret" not in printed
    assert "turn-canary-456" not in printed
    assert "controlToken=<REDACTED>" in printed
    assert "turnCredential=<REDACTED>" in printed


# Pretty output --------------------------------------------------------------


@patch(_LAN, return_value="10.0.0.1")
def test_summary_is_plain_text_when_not_a_tty(_mock_lan: MagicMock) -> None:
    """StringIO is not a tty, so no escape sequences should be emitted."""
    buf = io.StringIO()
    url, source = resolve_client_url(None)
    print_summary(
        url=url,
        source=source,
        probe_device=False,
        stream=buf,
    )
    printed = buf.getvalue()
    assert "\033[" not in printed
    assert "10.0.0.1:48322" in printed
    assert url in printed


def test_summary_reports_verbatim_urls_own_target() -> None:
    """The summary must describe the URL being opened, not the local defaults."""
    given = "https://example.test/client/?oobEnable=1&serverIP=203.0.113.7&port=1234"
    url, source = resolve_client_url(given)
    buf = io.StringIO()
    print_summary(
        url=url,
        source=source,
        probe_device=False,
        stream=buf,
    )
    printed = buf.getvalue()
    assert "203.0.113.7:1234" in printed
    assert "48322" not in printed


@patch(_LAN, return_value="10.0.0.1")
def test_summary_skips_adb_when_not_probing(_mock_lan: MagicMock) -> None:
    """--print-only must not shell out to adb."""
    buf = io.StringIO()
    with patch("cloudxr_py_test_ns.webclient.headset_browser_package") as probe:
        url, source = resolve_client_url(None)
        print_summary(
            url=url,
            source=source,
            probe_device=False,
            stream=buf,
        )
    probe.assert_not_called()


# Argument parsing -----------------------------------------------------------


def test_parse_args_defaults() -> None:
    args = _parse_args([])
    assert args.client_url is None
    assert args.print_only is False


def test_parse_args_positional_and_flag() -> None:
    args = _parse_args(["https://x.test", "--print-only"])
    assert args.client_url == "https://x.test"
    assert args.print_only is True


# main() ---------------------------------------------------------------------


@patch(_LAN, return_value="10.0.0.1")
def test_main_print_only_skips_adb(_mock_lan: MagicMock, capsys) -> None:
    with (
        patch("cloudxr_py_test_ns.webclient.require_adb_on_path") as req,
        patch("cloudxr_py_test_ns.webclient.assert_supported_headset") as identity,
        patch("cloudxr_py_test_ns.webclient.open_url_on_headset") as opener,
    ):
        rc = main(["--print-only"])
    assert rc == 0
    req.assert_not_called()
    identity.assert_not_called()
    opener.assert_not_called()
    assert "serverIP=10.0.0.1" in capsys.readouterr().out


@patch(_LAN, return_value="10.0.0.1")
def test_main_opens_resolved_url(_mock_lan: MagicMock, capsys) -> None:
    with (
        patch("cloudxr_py_test_ns.webclient.require_adb_on_path"),
        patch("cloudxr_py_test_ns.webclient.assert_exactly_one_adb_device"),
        patch("cloudxr_py_test_ns.webclient.assert_supported_headset"),
        patch("cloudxr_py_test_ns.webclient.assert_headset_awake"),
        patch(
            "cloudxr_py_test_ns.webclient.headset_browser_package",
            return_value="com.pico.browser",
        ),
        patch(
            "cloudxr_py_test_ns.webclient.open_url_on_headset", return_value=(0, "")
        ) as opener,
    ):
        rc = main([])
    assert rc == 0
    opened = opener.call_args[0][0]
    assert "serverIP=10.0.0.1" in opened
    assert "oobEnable" not in opened
    out = capsys.readouterr().out
    assert "com.pico.browser" in out
    assert "opened on headset" in out


@patch(_LAN, return_value="10.0.0.1")
def test_main_reports_adb_failure_with_hint(_mock_lan: MagicMock, capsys) -> None:
    with (
        patch("cloudxr_py_test_ns.webclient.require_adb_on_path"),
        patch("cloudxr_py_test_ns.webclient.assert_exactly_one_adb_device"),
        patch("cloudxr_py_test_ns.webclient.assert_supported_headset"),
        patch("cloudxr_py_test_ns.webclient.assert_headset_awake"),
        patch(
            "cloudxr_py_test_ns.webclient.headset_browser_package", return_value=None
        ),
        patch(
            "cloudxr_py_test_ns.webclient.open_url_on_headset",
            return_value=(1, "no devices/emulators found"),
        ),
    ):
        rc = main([])
    assert rc == 1
    captured = capsys.readouterr()
    assert "no devices/emulators found" in captured.err
    assert "No adb device" in captured.err


def test_main_rejects_non_headset_before_opening(capsys) -> None:
    with (
        patch("cloudxr_py_test_ns.webclient.require_adb_on_path"),
        patch(
            "cloudxr_py_test_ns.webclient.assert_exactly_one_adb_device",
            return_value="phone-serial",
        ),
        patch(
            "cloudxr_py_test_ns.webclient.assert_supported_headset",
            side_effect=OobAdbError("Selected ADB device is not a supported headset"),
        ),
        patch("cloudxr_py_test_ns.webclient.assert_headset_awake") as awake,
        patch("cloudxr_py_test_ns.webclient.open_url_on_headset") as opener,
    ):
        rc = main([])
    assert rc == 1
    assert "not a supported headset" in capsys.readouterr().err
    awake.assert_not_called()
    opener.assert_not_called()


def test_main_reports_preflight_failure_without_traceback() -> None:
    """An adb preflight failure is a clean exit code, not an escaping OobAdbError."""
    from cloudxr_py_test_ns.oob_teleop_adb import OobAdbError as _Err

    with patch(
        "cloudxr_py_test_ns.webclient.require_adb_on_path",
        side_effect=_Err("no adb"),
    ):
        rc = main([])
    assert rc == 1


def test_main_preflight_hint_does_not_mention_setup_oob(capsys) -> None:
    """Shared adb advice must name no CLI flag — this tool has no --setup-oob."""
    with patch("cloudxr_py_test_ns.oob_teleop_adb.shutil.which", return_value=None):
        rc = main([])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not found on PATH" in err
    assert "--setup-oob" not in err
    assert "open the teleop URL on the headset" in err


def test_main_returns_1_when_lan_unresolvable(capsys) -> None:
    """No LAN IP and no override is a clean error, not a traceback."""
    with patch(_LAN, side_effect=RuntimeError("no LAN IP")):
        rc = main([])
    assert rc == 1
    assert "no LAN IP" in capsys.readouterr().err

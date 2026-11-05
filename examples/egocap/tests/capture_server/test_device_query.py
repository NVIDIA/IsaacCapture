# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The device query: protocol, folder scan, the read-only ``/query`` route, Begin selection and the client."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import websockets

from capture_server import mcal_catalog
from capture_server import protocol as p
from capture_server.acquisition.isaac_capture import IsaacCaptureAdapter, TrackingSelection
from capture_server.client import CaptureClient, DeviceQueryError, ManusPluginConfig, RequestRejectedError
from capture_server.launch import _calibration_path
from capture_server.server import ServerConfig, _build_arg_parser
from tests.support.capture_server import start_capture_server, stop_capture_server

STATUS = p.ManusCalibrationCatalogStatus


def _mcal(directory: Path, name: str, data: bytes = b"calibration") -> Path:
    """Write one MCAL file into ``directory``, creating it."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(data)
    return path


# -- protocol -----------------------------------------------------------------------------------------------


def test_query_request_round_trips():
    request = p.QueryDevicesRequest(request_id="r-1")
    assert p.parse_query_devices(request.to_json()) == request


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(type=1),
        lambda d: d.update(version=99),
        lambda d: d.update(request_id=""),
        lambda d: d.update(request_id=7),
        lambda d: d.update(extra=1),
        lambda d: d.pop("request_id"),
    ],
)
def test_query_request_rejects_malformed_envelopes(mutate):
    data = p.QueryDevicesRequest(request_id="r-1").to_json()
    mutate(data)
    with pytest.raises(p.ProtocolError):
        p.parse_query_devices(data)


def test_devices_reply_round_trips_through_the_server_message_parser():
    entry = p.DeviceEntry(p.PluginType.MANUS, STATUS.READY, ("a.mcal", "b.mcal"))
    reply = p.DevicesReply("r-1", (entry,))
    assert p.parse_server_message(json.loads(json.dumps(reply.to_json()))) == reply


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(mcal_status=STATUS.UNAVAILABLE),
        dict(mcal_status=STATUS.READY, error_message="boom"),
        dict(mcal_status=STATUS.UNAVAILABLE, error_message=""),
        dict(mcal_status=STATUS.DISABLED, mcal_files=("a.mcal",)),
        dict(mcal_status=STATUS.READY, mcal_files=("a.mcal", "a.mcal")),
        dict(mcal_status=STATUS.READY, mcal_files=("../a.mcal",)),
    ],
)
def test_device_entry_rejects_inconsistent_listings(kwargs):
    with pytest.raises(ValueError):
        p.DeviceEntry(p.PluginType.MANUS, **kwargs)


def test_device_entry_rejects_too_many_files(monkeypatch):
    monkeypatch.setattr(p, "MAX_MCAL_FILES", 1)
    with pytest.raises(ValueError, match="at most"):
        p.DeviceEntry(p.PluginType.MANUS, STATUS.READY, ("a.mcal", "b.mcal"))


def test_devices_reply_rejects_a_repeated_device():
    entry = p.DeviceEntry(p.PluginType.MANUS, STATUS.DISABLED)
    with pytest.raises(ValueError, match="twice"):
        p.DevicesReply("r-1", (entry, entry))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(devices="x"),
        lambda d: d["devices"][0].update(mcal_files="x"),
        lambda d: d["devices"][0].update(mcal_files=[1]),
        lambda d: d["devices"][0].update(error_message=3),
        lambda d: d["devices"][0].update(mcal_status=9),
        lambda d: d["devices"][0].update(device=9),
        lambda d: d["devices"][0].update(mcal_files=["nope.txt"]),
        lambda d: d["devices"][0].update(error_message="x"),
        lambda d: d["devices"][0].update(extra=1),
    ],
)
def test_devices_reply_rejects_malformed_replies(mutate):
    data = p.DevicesReply("r-1", (p.DeviceEntry(p.PluginType.MANUS, STATUS.READY, ("a.mcal",)),)).to_json()
    mutate(data)
    with pytest.raises(p.ProtocolError):
        p.parse_server_message(data)


def test_plugin_request_carries_calibration_names_through_the_wire():
    begin = p.BeginRequest(
        tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)),
        plugins=(
            p.PluginRequest(plugin=p.PluginType.MANUS, left_calibration="l.mcal", right_calibration="r.mcal"),
        ),
        session_id="s-1",
    )
    config = p.effective_config(begin)
    assert config.plugins[0].left_calibration == "l.mcal"
    wire = json.loads(json.dumps(config.to_json()))
    assert p.CaptureConfiguration.from_json(wire) == config
    assert p._parse_plugin_request(wire["plugins"][0]).right_calibration == "r.mcal"


def test_plugin_request_without_files_keeps_its_earlier_wire_form():
    config = p.effective_config(
        p.BeginRequest(
            tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)),
            plugins=(p.PluginRequest(plugin=p.PluginType.MANUS),),
        )
    )
    assert set(config.to_json()["plugins"][0]) == {"plugin", "enabled", "required", "datasets"}


@pytest.mark.parametrize("name", ["../x.mcal", "dir/x.mcal", ".hidden.mcal", "x.txt", "", "x" * 200 + ".mcal"])
def test_plugin_request_rejects_names_that_are_not_basenames(name):
    with pytest.raises(ValueError):
        p.PluginRequest(plugin=p.PluginType.MANUS, left_calibration=name)
    with pytest.raises(p.ProtocolError):
        p._parse_plugin_request({"plugin": 1, "left_calibration": name})


def test_plugin_request_rejects_one_file_for_both_hands():
    with pytest.raises(ValueError, match="different files"):
        p.PluginRequest(plugin=p.PluginType.MANUS, left_calibration="x.mcal", right_calibration="x.mcal")
    with pytest.raises(p.ProtocolError, match="different files"):
        p._parse_plugin_request({"plugin": 1, "left_calibration": "x.mcal", "right_calibration": "x.mcal"})


# -- folder scan --------------------------------------------------------------------------------------------


def test_scan_without_a_directory_is_disabled():
    catalog = mcal_catalog.scan(None)
    assert (catalog.status, catalog.files) == (STATUS.DISABLED, ())


def test_scan_of_a_missing_directory_is_an_empty_ready_listing(tmp_path):
    catalog = mcal_catalog.scan(tmp_path / "absent")
    assert (catalog.status, catalog.files) == (STATUS.READY, ())


def test_scan_offers_only_nonempty_readable_regular_mcal_files_sorted(tmp_path):
    _mcal(tmp_path, "b.mcal")
    _mcal(tmp_path, "a.mcal")
    _mcal(tmp_path, "empty.mcal", b"")
    _mcal(tmp_path, "notes.txt")
    (tmp_path / "subdir.mcal").mkdir()
    (tmp_path / "link.mcal").symlink_to(tmp_path / "a.mcal")
    unreadable = _mcal(tmp_path, "locked.mcal")
    unreadable.chmod(0)
    try:
        catalog = mcal_catalog.scan(tmp_path)
    finally:
        unreadable.chmod(0o600)
    assert catalog.status is STATUS.READY
    # The unreadable file is only hidden when the tests do not run as root, which ignores file modes.
    expected = (
        ["a.mcal", "b.mcal"]
        if unreadable.stat().st_uid != 0 and not _is_root()
        else ["a.mcal", "b.mcal", "locked.mcal"]
    )
    assert list(catalog.files) == expected


def _is_root() -> bool:
    import os

    return os.geteuid() == 0


def test_scan_of_a_file_instead_of_a_directory_is_unavailable(tmp_path):
    path = tmp_path / "file"
    path.write_text("x")
    catalog = mcal_catalog.scan(path)
    assert catalog.status is STATUS.UNAVAILABLE and "not a directory" in catalog.error_message


def test_scan_reports_an_unreadable_directory(tmp_path, monkeypatch):
    def deny(self, *args, **kwargs):
        raise PermissionError

    monkeypatch.setattr(Path, "resolve", deny)
    catalog = mcal_catalog.scan(tmp_path)
    assert catalog.status is STATUS.UNAVAILABLE and "permission denied" in catalog.error_message


def test_scan_reports_other_io_errors(tmp_path, monkeypatch):
    def fail(self, *args, **kwargs):
        raise OSError(5, "boom")

    monkeypatch.setattr(Path, "resolve", fail)
    catalog = mcal_catalog.scan(tmp_path)
    assert catalog.status is STATUS.UNAVAILABLE and "boom" in catalog.error_message


def test_scan_over_the_limit_is_an_error_not_a_truncated_list(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "MAX_MCAL_FILES", 2)
    for name in ("a.mcal", "b.mcal", "c.mcal"):
        _mcal(tmp_path, name)
    catalog = mcal_catalog.scan(tmp_path)
    assert (catalog.status, catalog.files) == (STATUS.UNAVAILABLE, ())


def test_scan_skips_an_entry_that_vanishes_mid_scan(tmp_path, monkeypatch):
    _mcal(tmp_path, "a.mcal")
    real_stat = Path.stat

    def flaky(self, *args, **kwargs):
        return real_stat(self, *args, **kwargs)

    import os

    real_scandir = os.scandir

    class _Gone:
        name = "gone.mcal"
        path = str(tmp_path / "gone.mcal")

        def is_symlink(self):
            raise FileNotFoundError

    class _Listing:
        def __enter__(self):
            return iter([*real_scandir(tmp_path), _Gone()])

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(mcal_catalog.os, "scandir", lambda root: _Listing())
    monkeypatch.setattr(Path, "stat", flaky)
    assert list(mcal_catalog.scan(tmp_path).files) == ["a.mcal"]


def test_selection_refusal_explains_each_case(tmp_path):
    _mcal(tmp_path, "a.mcal")
    ready = mcal_catalog.scan(tmp_path)
    assert mcal_catalog.selection_refusal(ready, ["a.mcal"]) is None
    assert "not an available file" in mcal_catalog.selection_refusal(ready, ["a.mcal", "b.mcal"])
    assert "--manus-calibration-dir" in mcal_catalog.selection_refusal(mcal_catalog.scan(None), ["a.mcal"])
    broken = mcal_catalog.Catalog(STATUS.UNAVAILABLE, (), "disk on fire")
    assert "disk on fire" in mcal_catalog.selection_refusal(broken, ["a.mcal"])


def test_resolve_joins_the_name_onto_the_real_directory(tmp_path):
    assert mcal_catalog.resolve(tmp_path, "a.mcal") == tmp_path.resolve() / "a.mcal"


# -- launch and adapter -------------------------------------------------------------------------------------


def test_calibration_path_resolves_inside_the_folder_or_keeps_an_inert_name(tmp_path):
    assert _calibration_path(None, tmp_path) is None
    assert _calibration_path("a.mcal", None) == "a.mcal"
    assert _calibration_path("a.mcal", tmp_path) == str(tmp_path.resolve() / "a.mcal")


def test_adapter_passes_selected_files_to_the_manus_plugin():
    def args(**kwargs):
        adapter = IsaacCaptureAdapter.__new__(IsaacCaptureAdapter)
        adapter._selection = TrackingSelection(
            head=False, controller_left=False, controller_right=False, hand_left=False, hand_right=False, **kwargs
        )
        return adapter._manus_args()

    assert args(manus_enabled=True) == []
    assert args(manus_enabled=True, manus_left_calibration="/d/l.mcal", manus_right_calibration="/d/r.mcal") == [
        "--left-calibration=/d/l.mcal",
        "--right-calibration=/d/r.mcal",
    ]


# -- CLI ----------------------------------------------------------------------------------------------------


def test_cli_flag_defaults_to_no_folder_and_round_trips():
    parser = _build_arg_parser()
    assert parser.parse_args(["--mock"]).manus_calibration_dir is None
    assert parser.parse_args(["--mock", "--manus-calibration-dir", "/mcal"]).manus_calibration_dir == "/mcal"


# -- the /query route and Begin selection, against a running server ------------------------------------------


async def _ask(url: str, request: dict | str | bytes, *, request_id: str = "q-1"):
    """Send one message to /query; returns (reply or None, close code)."""
    ws = await websockets.connect(f"{url}/query")
    try:
        await ws.send(request if isinstance(request, (str, bytes)) else json.dumps(request))
        reply = None
        try:
            reply = p.parse_server_message(json.loads(await asyncio.wait_for(ws.recv(), 5)))
        except websockets.ConnectionClosed:
            pass  # a rejected request is closed without a reply; the close code is what the caller checks
        await asyncio.wait_for(ws.wait_closed(), 5)
        return reply, ws.close_code
    finally:
        await ws.close()


def _query(request_id: str = "q-1") -> dict:
    return p.QueryDevicesRequest(request_id=request_id).to_json()


@pytest.fixture
async def station(unused_tcp_port, tmp_path):
    """A running mock server with a managed MCAL folder that already holds two files."""
    folder = tmp_path / "mcal"
    _mcal(folder, "left.mcal")
    _mcal(folder, "right.mcal")
    server = await start_capture_server(unused_tcp_port, config=ServerConfig(manus_calibration_dir=folder))
    try:
        yield f"ws://127.0.0.1:{unused_tcp_port}", folder, server
    finally:
        await stop_capture_server(server)


async def test_query_lists_the_manus_device_with_its_mcal_files(station):
    url, folder, _ = station
    _mcal(folder, "later.mcal")
    reply, code = await _ask(url, _query("echo-me"))
    assert code == 1000
    assert reply.request_id == "echo-me"
    (entry,) = reply.devices
    assert (entry.device, entry.mcal_status) == (p.PluginType.MANUS, STATUS.READY)
    assert entry.mcal_files == ("later.mcal", "left.mcal", "right.mcal")


async def test_query_without_a_folder_reports_disabled(unused_tcp_port):
    server = await start_capture_server(unused_tcp_port)
    try:
        reply, _ = await _ask(f"ws://127.0.0.1:{unused_tcp_port}", _query())
    finally:
        await stop_capture_server(server)
    assert (reply.devices[0].mcal_status, reply.devices[0].mcal_files) == (STATUS.DISABLED, ())


async def test_query_reports_an_unusable_folder_as_unavailable(station):
    url, folder, _ = station
    for child in folder.iterdir():
        child.unlink()
    folder.rmdir()
    folder.write_text("not a directory")
    reply, _ = await _ask(url, _query())
    assert reply.devices[0].mcal_status is STATUS.UNAVAILABLE and reply.devices[0].error_message


@pytest.mark.parametrize(
    "message",
    [
        b"binary",
        "not json",
        "[]",
        json.dumps({"type": int(p.MessageType.BEGIN), "version": 1, "session_id": "s"}),
        json.dumps({"type": int(p.MessageType.STOP), "version": 1, "session_id": "s"}),
        json.dumps({"type": int(p.MessageType.QUERY_DEVICES), "version": 1}),
    ],
)
async def test_query_route_closes_on_anything_but_a_valid_device_query(station, message):
    url, _, _ = station
    reply, code = await _ask(url, message)
    assert reply is None and code == 1002


async def test_query_route_closes_when_the_peer_sends_nothing(station, monkeypatch):
    from capture_server import server as server_module

    monkeypatch.setattr(server_module, "_QUERY_DEADLINE_S", 0.2)
    url, _, _ = station
    ws = await websockets.connect(f"{url}/query")
    await asyncio.wait_for(ws.wait_closed(), 5)


async def test_query_during_an_active_capture_does_not_disturb_it(station):
    url, _, server = station
    client = CaptureClient(url)
    await client.start(sources=(p.SourceType.HEAD,))
    try:
        reply, code = await _ask(url, _query())
        assert code == 1000 and reply.devices[0].mcal_status is STATUS.READY
        assert server.server._active_session is not None and not server.server._active_session.done.is_set()
    finally:
        await client.stop()


async def test_oversized_reply_becomes_an_unavailable_entry(station, monkeypatch):
    url, _, _ = station
    monkeypatch.setattr(p, "MAX_QUERY_REPLY_BYTES", 10)
    reply, code = await _ask(url, _query())
    assert code == 1000
    assert reply.devices[0].mcal_status is STATUS.UNAVAILABLE and "too large" in reply.devices[0].error_message


async def test_a_scan_that_times_out_or_fails_is_unavailable(station, monkeypatch):
    url, _, _ = station
    monkeypatch.setattr(mcal_catalog, "scan", lambda directory: (_ for _ in ()).throw(RuntimeError("boom")))
    reply, _ = await _ask(url, _query())
    assert reply.devices[0].mcal_status is STATUS.UNAVAILABLE and "failed" in reply.devices[0].error_message


async def test_scan_timeout_is_reported_without_leaking_the_slot(station, monkeypatch):
    from capture_server import server as server_module

    url, _, server = station
    release = asyncio.Event()
    import threading

    gate = threading.Event()
    monkeypatch.setattr(
        mcal_catalog, "scan", lambda directory: (gate.wait(5), mcal_catalog.Catalog(STATUS.READY))[1]
    )
    monkeypatch.setattr(server_module, "_QUERY_DEADLINE_S", 0.2)
    reply, _ = await _ask(url, _query())
    assert "timed out" in reply.devices[0].error_message
    gate.set()
    release.set()
    for _ in range(100):
        if server.server._mcal_slots._value == server_module._MCAL_SCANS:
            break
        await asyncio.sleep(0.02)
    assert server.server._mcal_slots._value == server_module._MCAL_SCANS


async def test_begin_accepts_files_that_are_in_the_folder_and_echoes_the_names(station):
    url, _, _ = station
    client = CaptureClient(url)
    started = asyncio.Event()
    seen = []
    await client.start(
        sources=(p.SourceType.HEAD,),
        plugins=(ManusPluginConfig(left_calibration="left.mcal", right_calibration="right.mcal"),),
        on_started=lambda s: (seen.append(s), started.set()),
    )
    try:
        await asyncio.wait_for(started.wait(), 5)
        plugin = seen[0].effective_config.plugins[0]
        assert (plugin.left_calibration, plugin.right_calibration) == ("left.mcal", "right.mcal")
    finally:
        await client.stop()


async def test_begin_refuses_a_file_that_is_not_in_the_folder(station):
    url, _, _ = station
    with pytest.raises(RequestRejectedError) as excinfo:
        await CaptureClient(url).start(
            sources=(p.SourceType.HEAD,), plugins=(ManusPluginConfig(left_calibration="missing.mcal"),)
        )
    assert excinfo.value.reason == p.RejectionReason.INVALID_CONFIG
    assert "not an available file" in excinfo.value.error_message


async def test_begin_refuses_a_file_that_became_a_link_or_empty_since_listing(station):
    url, folder, _ = station
    (folder / "left.mcal").write_bytes(b"")
    with pytest.raises(RequestRejectedError):
        await CaptureClient(url).start(
            sources=(p.SourceType.HEAD,), plugins=(ManusPluginConfig(left_calibration="left.mcal"),)
        )


async def test_begin_with_files_but_no_folder_is_refused(unused_tcp_port):
    server = await start_capture_server(unused_tcp_port)
    try:
        with pytest.raises(RequestRejectedError) as excinfo:
            await CaptureClient(f"ws://127.0.0.1:{unused_tcp_port}").start(
                sources=(p.SourceType.HEAD,), plugins=(ManusPluginConfig(left_calibration="x.mcal"),)
            )
    finally:
        await stop_capture_server(server)
    assert "--manus-calibration-dir" in excinfo.value.error_message


# -- client -------------------------------------------------------------------------------------------------


async def test_client_query_devices_returns_the_listing(station):
    url, _, _ = station
    (entry,) = await CaptureClient(url).query_devices()
    assert entry.mcal_files == ("left.mcal", "right.mcal")


async def test_client_query_devices_works_while_the_same_client_is_capturing(station):
    url, _, _ = station
    client = CaptureClient(url)
    await client.start(sources=(p.SourceType.HEAD,))
    try:
        assert (await client.query_devices())[0].mcal_status is STATUS.READY
    finally:
        await client.stop()


def test_manus_plugin_config_validates_names():
    with pytest.raises(ValueError, match="basename"):
        ManusPluginConfig(left_calibration="../x.mcal")
    with pytest.raises(ValueError, match="different files"):
        ManusPluginConfig(left_calibration="x.mcal", right_calibration="x.mcal")
    assert ManusPluginConfig().plugin is p.PluginType.MANUS


async def _fake_server(handler, port):
    return await websockets.serve(handler, "127.0.0.1", port)


async def test_client_reports_a_server_that_does_not_know_the_query(unused_tcp_port):
    async def handler(ws):
        await ws.recv()
        await ws.close(code=1013, reason="owner connection already active")

    server = await _fake_server(handler, unused_tcp_port)
    try:
        with pytest.raises(DeviceQueryError, match="close code 1013"):
            await CaptureClient(f"ws://127.0.0.1:{unused_tcp_port}").query_devices()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (b"binary", "binary"),
        ("not json", "invalid"),
        (json.dumps({"type": 1}), "invalid"),
        (None, "does not answer"),
    ],
)
async def test_client_never_accepts_a_wrong_reply_as_an_answer(unused_tcp_port, reply, message):
    async def handler(ws):
        await ws.recv()
        # A well-formed reply for some other request id must not be accepted as this query's answer.
        await ws.send(reply if reply is not None else json.dumps(p.DevicesReply("someone-else", ()).to_json()))
        await ws.close()

    server = await _fake_server(handler, unused_tcp_port)
    try:
        with pytest.raises(DeviceQueryError, match=message):
            await CaptureClient(f"ws://127.0.0.1:{unused_tcp_port}").query_devices()
    finally:
        server.close()
        await server.wait_closed()


async def test_client_reports_an_unreachable_server_and_a_timeout(unused_tcp_port, monkeypatch):
    from capture_server import client as client_module

    with pytest.raises(DeviceQueryError, match="failed"):
        await CaptureClient(f"ws://127.0.0.1:{unused_tcp_port}").query_devices()

    async def silent(ws):
        await asyncio.sleep(5)

    monkeypatch.setattr(client_module, "_QUERY_TIMEOUT_S", 0.2)
    server = await _fake_server(silent, unused_tcp_port)
    try:
        with pytest.raises(DeviceQueryError):
            await CaptureClient(f"ws://127.0.0.1:{unused_tcp_port}").query_devices()
    finally:
        server.close()

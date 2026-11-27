# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plugin discovery and rejection run without vendor SDKs or hardware."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from isaac_media_transport import transport_plugins as plugins
from isaac_media_transport.transport import get_transport


def adapter(name="sample"):
    return SimpleNamespace(
        capabilities=SimpleNamespace(name=name),
        **{
            method: lambda *a, **kw: None
            for method in ("messages", "video_sender", "video_receiver", "provisioner")
        },
    )


def register(monkeypatch, *entries):
    monkeypatch.setattr(plugins.metadata, "entry_points", lambda **kw: entries)


def test_external_plugin_is_loaded_only_when_selected(monkeypatch):
    loads = []

    def load():
        loads.append(True)
        return plugins.TransportPlugin("sample", 1, adapter)

    register(monkeypatch, SimpleNamespace(name="sample", load=load))
    assert "sample" in plugins.available_transports()
    assert loads == []
    assert get_transport({"transport": "sample"}).capabilities.name == "sample"
    assert loads == [True]


@pytest.mark.parametrize("version,name", [(2, "sample"), (1, "different")])
def test_incompatible_descriptor_never_constructs_adapter(monkeypatch, version, name):
    def create():
        pytest.fail("incompatible factory ran")

    register(
        monkeypatch,
        SimpleNamespace(
            name="sample", load=lambda: plugins.TransportPlugin(name, version, create)
        ),
    )
    with pytest.raises(ValueError, match="incompatible"):
        get_transport({"transport": "sample"})


def test_duplicate_names_fail_instead_of_shadowing(monkeypatch):
    entry = SimpleNamespace(name="sample")
    register(monkeypatch, entry, entry)
    with pytest.raises(ValueError, match="Duplicate"):
        plugins.available_transports()


def test_missing_sdk_does_not_fall_back(monkeypatch):
    def load():
        raise ModuleNotFoundError("sample_sdk")

    register(monkeypatch, SimpleNamespace(name="sample", load=load))
    with pytest.raises(ValueError, match=r"Cannot load transport 'sample'.*sample_sdk"):
        get_transport({"transport": "sample"})


def test_incomplete_adapter_is_rejected(monkeypatch):
    register(
        monkeypatch,
        SimpleNamespace(
            name="sample",
            load=lambda: plugins.TransportPlugin(
                "sample",
                1,
                lambda: SimpleNamespace(capabilities=SimpleNamespace(name="sample")),
            ),
        ),
    )
    with pytest.raises(ValueError, match="missing adapter method"):
        get_transport({"transport": "sample"})


def test_setup_and_discovery_do_not_import_vendor_sdks():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
class BlockVendor:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'livekit', 'cupy', 'isaaccapture'}:
            raise AssertionError('unexpected SDK import: ' + fullname)
sys.meta_path.insert(0, BlockVendor())
from isaac_media_transport import transport
from isaac_media_transport.transport_plugins import available_transports
available_transports()
""",
        ],
        check=True,
    )


def test_latency_output_switches_and_local_measurements(tmp_path):
    from isaac_media_transport.latency import ViewerTiming, get_recorder, DISABLED

    settings = {
        "latency": {
            "enabled": False,
            "file_enabled": True,
            "terminal_enabled": True,
            "hud_enabled": True,
            "log_dir": str(tmp_path),
        }
    }
    assert get_recorder(settings) is DISABLED
    reporter = ViewerTiming(settings)
    assert not reporter.enabled
    reporter.observe(10, 20, 30)
    assert "waiting" in reporter.lines()[0]
    assert not list(tmp_path.iterdir())
    settings["latency"].update(enabled=True, file_enabled=False)
    assert get_recorder(settings) is DISABLED
    reporter = ViewerTiming(settings)
    reporter.observe(10_000_000, 15_000_000, 17_000_000)
    assert "5.0 ms" in reporter.lines()[0]
    assert "2.0 ms" in reporter.lines()[0]
    assert "end-to-end unavailable" in reporter.lines()[1]
    assert not list(tmp_path.iterdir())
    for i in range(1000):
        reporter.observe(i, i + 1, i + 2)
    assert len(reporter._samples) == 600


def test_latency_terminal_and_hud_are_independent(monkeypatch):
    import itertools
    from isaac_media_transport import latency

    now = [10.0]
    monkeypatch.setattr(latency.time, "monotonic", lambda: now[0])
    for terminal, hud in itertools.product((False, True), repeat=2):
        reporter = latency.ViewerTiming(
            {
                "latency": {
                    "enabled": True,
                    "terminal_enabled": terminal,
                    "hud_enabled": hud,
                    "report_interval_s": 2,
                }
            }
        )
        assert reporter.terminal_enabled == terminal
        assert reporter.hud_enabled == hud
        output = []
        reporter.report(output.append)
        assert not output
        now[0] += 2
        reporter.report(output.append)
        assert len(output) == int(terminal)
    with pytest.raises(ValueError, match="report_interval_s"):
        latency.ViewerTiming({"latency": {"report_interval_s": 0}})


@pytest.mark.parametrize("file_enabled", [False, True])
@pytest.mark.parametrize("terminal_enabled", [False, True])
@pytest.mark.parametrize("hud_enabled", [False, True])
def test_all_latency_output_combinations(
    tmp_path, file_enabled, terminal_enabled, hud_enabled
):
    from isaac_media_transport.latency import get_recorder, ViewerTiming

    settings = {
        "latency": {
            "enabled": True,
            "file_enabled": file_enabled,
            "terminal_enabled": terminal_enabled,
            "hud_enabled": hud_enabled,
            "run_id": "test",
            "_process_role": "viewer",
            "log_dir": str(tmp_path),
        }
    }
    recorder = get_recorder(settings)
    recorder.record("viewer_submit_begin", "cam", 1)
    assert recorder.close()
    assert bool(list(tmp_path.glob("*.jsonl"))) == file_enabled
    reporter = ViewerTiming(settings)
    assert reporter.terminal_enabled == terminal_enabled
    assert reporter.hud_enabled == hud_enabled


def test_memory_telemetry_has_no_file_and_detects_restarts(tmp_path):
    from isaac_media_transport.latency_bus import EventBuffer, SocketTail

    path = tmp_path / "events.sock"
    buffer = EventBuffer(path, capacity=2)
    try:
        reader = SocketTail(path)
        reader.follow()
        buffer.append({"record_type": "stage", "run_id": "r", "monotonic_ns": 1})
        assert reader.read()[0]["run_id"] == "r"
        buffer.append({"record_type": "stage"})
        buffer.append({"record_type": "stage"})
        buffer.append({"record_type": "stage"})
        with pytest.raises(ValueError, match="overrun"):
            reader.read()
        with pytest.raises(ValueError, match="Another process"):
            EventBuffer(path)
        assert not list(tmp_path.glob("*.jsonl"))
    finally:
        buffer.close()
    replacement = EventBuffer(path)
    try:
        with pytest.raises(ValueError, match="restarted"):
            reader.read()
    finally:
        replacement.close()

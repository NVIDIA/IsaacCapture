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

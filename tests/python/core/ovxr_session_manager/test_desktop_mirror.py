# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared mirror configuration and the production input-copy fence."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

import isaaccapture.ovxr_session_manager.desktop_mirror as desktop_mirror


def test_config_rejects_invalid_window_and_cadence():
    assert desktop_mirror.DesktopMirrorConfig().max_fps == 30.0
    for fps in (0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="max_fps"):
            desktop_mirror.DesktopMirrorConfig(max_fps=fps)
    with pytest.raises(ValueError, match="window_width"):
        desktop_mirror.DesktopMirrorConfig(window_width=0)
    with pytest.raises(ValueError, match="title"):
        desktop_mirror.DesktopMirrorConfig(title="")


def test_session_import_and_mirror_config_do_not_request_optional_ovui():
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        requests = []

        class UnavailableOVUI(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "omni.ui" or fullname.startswith("omni.ui."):
                    requests.append(fullname)
                    raise ModuleNotFoundError(fullname)

        sys.meta_path.insert(0, UnavailableOVUI())
        from isaaccapture.ovxr_session_manager import DesktopMirrorConfig

        DesktopMirrorConfig()
        assert not requests, requests
        assert "omni.ui" not in sys.modules
        """
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        # Keep child diagnostics in the assertion output on failure.
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("fail_wait", [False, True])
def test_input_copy_completion_uses_legacy_stream_and_retires_event_once(
    monkeypatch, fail_wait, mock_cuda
):
    cuda = mock_cuda

    calls = []
    monkeypatch.setattr(
        cuda, "cudaSetDevice", lambda device: (calls.append(("device", device)) or 0,)
    )
    monkeypatch.setattr(
        cuda,
        "cudaEventCreateWithFlags",
        lambda flags: (calls.append(("create", flags)) or 0, 7),
    )
    monkeypatch.setattr(
        cuda,
        "cudaEventRecord",
        lambda event, stream: (calls.append(("record", event, stream)) or 0,),
    )
    monkeypatch.setattr(
        cuda,
        "cudaEventSynchronize",
        lambda event: (calls.append(("wait", event)) or int(fail_wait),),
    )
    monkeypatch.setattr(
        cuda, "cudaEventDestroy", lambda event: (calls.append(("destroy", event)) or 0,)
    )

    completion = desktop_mirror._InputCopyCompletion(3)
    if fail_wait:
        with pytest.raises(RuntimeError, match="input-copy completion"):
            completion.wait()
    else:
        completion.wait()
    completion.close()
    completion.close()
    assert calls == [
        ("device", 3),
        ("create", cuda.cudaEventDisableTiming),
        ("record", 7, cuda.cudaStreamLegacy),
        ("wait", 7),
        ("destroy", 7),
    ]

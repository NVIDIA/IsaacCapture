# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared mirror configuration and the production input-copy fence."""

from __future__ import annotations

import ast
import inspect

import pytest

import isaaccapture.teleop_session_manager.desktop_mirror as desktop_mirror


def test_config_is_bounded_and_ovui_import_stays_lazy():
    assert desktop_mirror.DesktopMirrorConfig().max_fps == 30.0
    for fps in (0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="max_fps"):
            desktop_mirror.DesktopMirrorConfig(max_fps=fps)
    with pytest.raises(ValueError, match="window_width"):
        desktop_mirror.DesktopMirrorConfig(window_width=0)
    with pytest.raises(ValueError, match="title"):
        desktop_mirror.DesktopMirrorConfig(title="")

    module = ast.parse(inspect.getsource(desktop_mirror))
    top_level_imports = [
        node for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all(
        not (
            (isinstance(node, ast.ImportFrom) and node.module == "omni")
            or (
                isinstance(node, ast.Import)
                and any(alias.name == "omni.ui" for alias in node.names)
            )
        )
        for node in top_level_imports
    )


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

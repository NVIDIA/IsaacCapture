# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared mirror configuration and the production input-copy fence."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

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
        from pathlib import Path

        # CTest supplies the built package via PYTHONPATH, which isolated mode ignores.
        # Select that exact package explicitly without inheriting parent import state.
        sys.path.insert(0, sys.argv[1])
        requests = []

        class UnavailableOVUI(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "omni.ui" or fullname.startswith("omni.ui."):
                    requests.append(fullname)
                    raise ModuleNotFoundError(fullname)

        sys.meta_path.insert(0, UnavailableOVUI())
        import isaaccapture
        from isaaccapture.ovxr_session_manager import DesktopMirrorConfig

        assert Path(isaaccapture.__file__).resolve().parent.parent == Path(sys.argv[1])

        DesktopMirrorConfig()
        assert not requests, requests
        assert "omni.ui" not in sys.modules
        """
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(Path(desktop_mirror.__file__).resolve().parents[2]),
        ],
        capture_output=True,
        text=True,
        # Keep child diagnostics in the assertion output on failure.
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("fail_wait", [False, True])
def test_input_copy_wait_covers_current_upload_device(
    monkeypatch, fail_wait, mock_cuda
):
    calls = []
    monkeypatch.setattr(
        mock_cuda,
        "cudaStreamSynchronize",
        lambda stream: (calls.append(stream) or int(fail_wait),),
    )
    if fail_wait:
        with pytest.raises(RuntimeError, match="input-copy completion"):
            desktop_mirror._wait_input_copy()
    else:
        desktop_mirror._wait_input_copy()
    assert calls == [mock_cuda.cudaStreamLegacy]

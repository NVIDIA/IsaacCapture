# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immediate playout must be explicit and fail closed when native support is absent."""

from types import SimpleNamespace

import pytest


class Function:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.result


@pytest.fixture
def native(monkeypatch, tmp_path):
    pytest.importorskip("livekit")
    from isaac_transport_livekit import livekit_cuda

    path = tmp_path / "libtest.so"
    path.touch()
    library = SimpleNamespace(
        livekit_umb_encoded_receive_abi=Function(1),
        livekit_umb_enable_zero_playout_delay=Function(0),
    )
    monkeypatch.setattr(livekit_cuda, "version", lambda _: "1.1.20")
    monkeypatch.setattr(livekit_cuda, "FfiClient", SimpleNamespace(_instance=None))
    monkeypatch.setattr(livekit_cuda.ctypes, "CDLL", lambda _: library)
    monkeypatch.delenv("LIVEKIT_LIB_PATH", raising=False)
    monkeypatch.delenv("UMB_LIVEKIT_ENCODED_RECEIVE", raising=False)
    return livekit_cuda, library, {"livekit_cuda_library": str(path)}


def test_default_does_not_enable_or_require_playout_extension(native):
    module, library, settings = native
    del library.livekit_umb_enable_zero_playout_delay
    module.configure_native(settings)
    assert module.os.environ["LIVEKIT_LIB_PATH"] == settings["livekit_cuda_library"]


def test_immediate_calls_native_before_sdk_initialization(native):
    module, library, settings = native
    module.configure_native({**settings, "livekit_playout_mode": "immediate"})
    assert library.livekit_umb_enable_zero_playout_delay.calls == 1


def test_missing_native_support_does_not_silently_fall_back(native):
    module, library, settings = native
    del library.livekit_umb_enable_zero_playout_delay
    with pytest.raises(RuntimeError, match="--playout-control"):
        module.configure_native({**settings, "livekit_playout_mode": "immediate"})
    assert "LIVEKIT_LIB_PATH" not in module.os.environ


def test_late_native_enable_fails_instead_of_mislabeling_trial(native):
    module, library, settings = native
    library.livekit_umb_enable_zero_playout_delay.result = 1
    with pytest.raises(RuntimeError, match="fresh receiver"):
        module.configure_native({**settings, "livekit_playout_mode": "immediate"})
    assert "LIVEKIT_LIB_PATH" not in module.os.environ


def test_invalid_playout_mode_is_rejected(native):
    module, library, settings = native
    with pytest.raises(ValueError, match="default or immediate"):
        module.configure_native({**settings, "livekit_playout_mode": "auto"})
    assert library.livekit_umb_enable_zero_playout_delay.calls == 0

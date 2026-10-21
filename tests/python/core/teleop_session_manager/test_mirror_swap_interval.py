# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from isaaccapture.teleop_session_manager import owned_mirror as mirror


@pytest.fixture
def glx(monkeypatch):
    state = SimpleNamespace(interval=1, calls=[], accept=True)

    @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int)
    def swap(display, drawable, interval):
        state.calls.append((display, drawable, interval))
        if state.accept:
            state.interval = interval

    library = SimpleNamespace(
        glXGetCurrentDisplay=Mock(return_value=101),
        glXGetCurrentContext=Mock(return_value=102),
        glXGetCurrentDrawable=Mock(return_value=103),
        glXQueryContext=Mock(return_value=0),
        glXQueryExtensionsString=Mock(return_value=b"GLX_EXT_swap_control"),
        glXGetProcAddressARB=Mock(
            return_value=ctypes.cast(swap, ctypes.c_void_p).value
        ),
        glXQueryDrawable=Mock(
            side_effect=lambda _d, _w, _a, value: setattr(
                value._obj, "value", state.interval
            )
        ),
    )
    monkeypatch.setattr(mirror.sys, "platform", "linux")
    monkeypatch.setattr(mirror.ctypes, "CDLL", lambda _name: library)
    state.library = library
    state.callback = swap  # Keep the native callback alive through the call.
    return state


def test_mirror_changes_only_its_current_drawable(glx):
    assert mirror._disable_glx_swap_interval()
    assert glx.calls == [(101, 103, 0)]
    assert glx.interval == 0


def test_non_glx_context_is_not_configured(glx):
    glx.library.glXGetCurrentContext.return_value = None
    assert not mirror._disable_glx_swap_interval()
    assert not glx.calls
    glx.library.glXQueryExtensionsString.assert_not_called()


def test_unsupported_extension_is_rejected_before_function_pointer_use(glx):
    glx.library.glXQueryExtensionsString.return_value = b"GLX_EXT_swap_control_tear"
    with pytest.raises(RuntimeError, match="requires GLX_EXT_swap_control"):
        mirror._disable_glx_swap_interval()
    glx.library.glXGetProcAddressARB.assert_not_called()
    assert not glx.calls


def test_rejected_window_policy_does_not_report_success(glx):
    glx.accept = False
    with pytest.raises(RuntimeError, match="did not become zero"):
        mirror._disable_glx_swap_interval()

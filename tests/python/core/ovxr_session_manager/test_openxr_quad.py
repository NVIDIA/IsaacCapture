# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace

import pytest

from isaaccapture.ovxr_session_manager.camera_mailbox import CameraImage
from isaaccapture.ovxr_session_manager.camera_panels import CameraPanelConfig
import isaaccapture.ovxr_session_manager.openxr_quad as quad_module


class _Function:
    def __init__(self):
        self.calls = []
        self.result = 0

    def __call__(self, *args):
        self.calls.append(args)
        return self.result


def _quad(tmp_path, monkeypatch, *, panels=None):
    library = tmp_path / "layer.so"
    library.touch()
    native = SimpleNamespace(
        isaacteleop_pip_create=_Function(),
        isaacteleop_pip_max_cameras=_Function(),
        isaacteleop_pip_max_image_dimension=_Function(),
        isaacteleop_pip_attach=_Function(),
        isaacteleop_pip_destroy=_Function(),
        isaacteleop_pip_layer=_Function(),
        isaacteleop_pip_configure=_Function(),
        isaacteleop_pip_publish_rgba=_Function(),
        isaacteleop_pip_supports_cuda=_Function(),
        isaacteleop_pip_publish_cuda_array=_Function(),
    )
    monkeypatch.setattr(quad_module.ctypes, "CDLL", lambda _path: native)
    native.isaacteleop_pip_create.result = 123
    native.isaacteleop_pip_max_cameras.result = 3
    native.isaacteleop_pip_max_image_dimension.result = 4096
    return quad_module.OpenXRCameraPanels(
        panels or (CameraPanelConfig("/Camera/Head"),), library_path=str(library)
    ), native


def test_default_config_is_accepted_by_native_quad(tmp_path, monkeypatch):
    quad, native = _quad(tmp_path, monkeypatch)
    assert quad.publish(0, CameraImage(1, 1, bytes(4), 1.0, 1))
    assert native.isaacteleop_pip_configure.calls[-1][3] == 1


def test_head_locked_image_aspect_and_invalid_frame_hides_quad(
    tmp_path,
    monkeypatch,
):
    quad, native = _quad(
        tmp_path,
        monkeypatch,
        panels=(CameraPanelConfig("/Camera/Head", panel_width_m=0.5),),
    )
    with pytest.raises(ValueError, match="head_locked"):
        CameraPanelConfig("/Camera/Head", space="viewer_start")
    image = CameraImage(640, 480, bytes(640 * 480 * 4), 1.0, 1)
    assert quad.publish(0, image)
    _handle, index, enabled, head_locked, _pose, size = (
        native.isaacteleop_pip_configure.calls[-1]
    )
    assert (index, enabled, head_locked) == (0, 1, 1)
    assert size[0] == pytest.approx(0.5)
    assert size[1] == pytest.approx(0.375)
    assert not quad.publish(0, None, invalid=True)
    assert native.isaacteleop_pip_configure.calls[-1][2] == 0
    assert quad.last_errors[0] is not None


@pytest.mark.parametrize("result", [-1, -2, -3])
def test_failed_binding_retains_only_its_partial_cleanup_owner(
    tmp_path, monkeypatch, result
):
    quad, native = _quad(tmp_path, monkeypatch)
    native.isaacteleop_pip_attach.result = result
    with pytest.raises(RuntimeError, match="binding failed"):
        quad.attach(
            SimpleNamespace(xr_instance=1, xr_session=2, xr_get_instance_proc_addr=3),
            SimpleNamespace(
                vk_physical_device=4, vk_device=5, vk_queue=6, queue_family_index=0
            ),
        )
    quad.close()
    assert len(native.isaacteleop_pip_destroy.calls) == 1


def test_upload_failure_hides_last_valid_panel(tmp_path, monkeypatch):
    quad, native = _quad(tmp_path, monkeypatch)
    native.isaacteleop_pip_publish_rgba.result = -6
    assert not quad.publish(0, CameraImage(1, 1, b"\0" * 4, 1.0, 1))
    assert native.isaacteleop_pip_configure.calls[-1][2] == 0


def test_pixel_buffer_failure_hides_last_valid_panel(tmp_path, monkeypatch):
    quad, native = _quad(tmp_path, monkeypatch)
    assert quad.publish(0, CameraImage(1, 1, b"\0" * 4, 1.0, 1))
    monkeypatch.setattr(
        quad_module.ctypes,
        "create_string_buffer",
        lambda _pixels: (_ for _ in ()).throw(MemoryError()),
    )
    with pytest.raises(MemoryError):
        quad.publish(0, CameraImage(1, 1, b"\0" * 4, 2.0, 2))
    assert native.isaacteleop_pip_configure.calls[-1][2] == 0


def test_explicit_binding_preserves_loader_and_retains_failed_cleanup(
    tmp_path, monkeypatch
):
    quad, native = _quad(tmp_path, monkeypatch)
    monkeypatch.setenv("XR_ENABLE_API_LAYERS", "XR_APILAYER_OTHER")
    monkeypatch.delenv("XR_API_LAYER_PATH", raising=False)
    xr = SimpleNamespace(xr_instance=1, xr_session=2, xr_get_instance_proc_addr=3)
    vk = SimpleNamespace(
        vk_physical_device=4, vk_device=5, vk_queue=6, queue_family_index=7
    )
    quad.attach(xr, vk)
    assert native.isaacteleop_pip_attach.calls == [(123, 1, 2, 3, 4, 5, 6, 7)]
    assert os.environ["XR_ENABLE_API_LAYERS"] == "XR_APILAYER_OTHER"
    assert "XR_API_LAYER_PATH" not in os.environ
    native.isaacteleop_pip_destroy.result = -1
    with pytest.raises(RuntimeError, match="retain OVXR"):
        quad.close()
    assert quad._attached
    native.isaacteleop_pip_destroy.result = 0
    quad.close()
    assert not quad._attached


def test_three_panels_keep_independent_revisions(tmp_path, monkeypatch):
    owner, native = _quad(
        tmp_path,
        monkeypatch,
        panels=tuple(CameraPanelConfig(f"/Camera/{i}") for i in range(3)),
    )
    for index in range(3):
        assert owner.publish(index, CameraImage(1, 1, b"abcd", 1.0, 1))
    assert [call[1] for call in native.isaacteleop_pip_publish_rgba.calls] == [0, 1, 2]
    owner.publish(1, None, invalid=True)
    assert native.isaacteleop_pip_configure.calls[-1][1:3] == (1, 0)
    assert not owner.publish(0, CameraImage(1, 1, b"abcd", 1.0, 1))
    assert owner.publish(2, CameraImage(1, 1, b"efgh", 2.0, 2))


def test_old_native_layer_is_rejected_before_upload(tmp_path, monkeypatch):
    _quad(tmp_path, monkeypatch)
    monkeypatch.setattr(quad_module.ctypes, "CDLL", lambda _path: SimpleNamespace())
    with pytest.raises(AttributeError, match="isaacteleop_pip_create"):
        quad_module.OpenXRCameraPanels(
            (CameraPanelConfig("/Camera/Head"),),
            library_path=str(tmp_path / "layer.so"),
        )


@pytest.mark.parametrize("result", [0, -6, -7])
def test_gpu_quad_completion_controls_source_retirement(tmp_path, monkeypatch, result):
    quad, native = _quad(tmp_path, monkeypatch)
    quad.supports_gpu = True
    gpu = SimpleNamespace(
        closed=False, busy=False, is_array=True, pointer=123, device=1
    )
    image = CameraImage(720, 450, None, 1.0, 1, gpu)
    native.isaacteleop_pip_publish_cuda_array.result = result
    assert quad.publish(0, image) == (result == 0)
    assert gpu.busy == (result == -7)
    assert len(native.isaacteleop_pip_publish_cuda_array.calls) == 1
    assert native.isaacteleop_pip_publish_rgba.calls == []

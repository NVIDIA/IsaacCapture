# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest

from isaaccapture.ovxr_session_manager.camera_mailbox import CameraImage
from isaaccapture.ovxr_session_manager.ovxr_session import AuxiliaryRasterConfig
from isaaccapture.ovxr_session_manager.openxr_quad import OpenXRCameraQuad
import isaaccapture.ovxr_session_manager.openxr_quad as quad_module


class _Function:
    def __init__(self):
        self.calls = []
        self.result = 0

    def __call__(self, *args):
        self.calls.append(args)
        return self.result


def _quad(tmp_path, monkeypatch):
    library = tmp_path / "layer.so"
    library.touch()
    manifest = tmp_path / "layer.json"
    manifest.write_text(
        json.dumps(
            {
                "api_layer": {
                    "name": "XR_APILAYER_ISAACTELEOP_camera_quad",
                    "library_path": library.name,
                }
            }
        )
    )
    native = SimpleNamespace(
        isaacteleop_pip_attach_v3=_Function(),
        isaacteleop_pip_detach_v3=_Function(),
        isaacteleop_pip_layer_v3=_Function(),
        isaacteleop_pip_configure_v2=_Function(),
        isaacteleop_pip_publish_rgba_v2=_Function(),
        isaacteleop_pip_supports_cuda_v2=_Function(),
        isaacteleop_pip_publish_cuda_array_v2=_Function(),
    )
    monkeypatch.setattr(quad_module.ctypes, "CDLL", lambda _path: native)
    return OpenXRCameraQuad(str(manifest)), native


def test_default_config_is_accepted_by_native_quad(tmp_path, monkeypatch):
    quad, native = _quad(tmp_path, monkeypatch)
    quad.configure(AuxiliaryRasterConfig())
    assert quad.publish(CameraImage(1, 1, bytes(4), 1.0, 1))
    assert native.isaacteleop_pip_configure_v2.calls[-1][2] == 1


def test_head_locked_image_aspect_and_invalid_frame_hides_quad(tmp_path, monkeypatch):
    quad, native = _quad(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="head_locked"):
        quad.configure(SimpleNamespace(space="viewer_start"))
    quad.configure(AuxiliaryRasterConfig(space="head_locked", panel_width_m=0.5))
    image = CameraImage(640, 480, bytes(640 * 480 * 4), 1.0, 1)
    assert quad.publish(image)
    index, enabled, head_locked, _pose, size = (
        native.isaacteleop_pip_configure_v2.calls[-1]
    )
    assert (index, enabled, head_locked) == (0, 1, 1)
    assert size[0] == pytest.approx(0.5)
    assert size[1] == pytest.approx(0.375)
    assert not quad.publish(None, invalid=True)
    assert native.isaacteleop_pip_configure_v2.calls[-1][1] == 0
    assert quad.last_error is not None


@pytest.mark.parametrize("result,owns_partial", [(-1, False), (-2, False), (-3, True)])
def test_failed_binding_retains_only_its_partial_cleanup_owner(
    tmp_path, monkeypatch, result, owns_partial
):
    quad, native = _quad(tmp_path, monkeypatch)
    native.isaacteleop_pip_attach_v3.result = result
    with pytest.raises(RuntimeError, match="binding failed"):
        quad.attach(
            SimpleNamespace(xr_instance=1, xr_session=2, xr_get_instance_proc_addr=3),
            SimpleNamespace(
                vk_physical_device=4, vk_device=5, vk_queue=6, queue_family_index=0
            ),
        )
    quad.close()
    assert len(native.isaacteleop_pip_detach_v3.calls) == int(owns_partial)


def test_upload_failure_hides_last_valid_panel(tmp_path, monkeypatch):
    quad, native = _quad(tmp_path, monkeypatch)
    quad.configure(AuxiliaryRasterConfig(space="head_locked"))
    native.isaacteleop_pip_publish_rgba_v2.result = -6
    assert not quad.publish(CameraImage(1, 1, b"\0" * 4, 1.0, 1))
    assert native.isaacteleop_pip_configure_v2.calls[-1][1] == 0


def test_pixel_buffer_failure_hides_last_valid_panel(tmp_path, monkeypatch):
    quad, native = _quad(tmp_path, monkeypatch)
    quad.configure(AuxiliaryRasterConfig(space="head_locked"))
    assert quad.publish(CameraImage(1, 1, b"\0" * 4, 1.0, 1))
    monkeypatch.setattr(
        quad_module.ctypes,
        "create_string_buffer",
        lambda _pixels: (_ for _ in ()).throw(MemoryError()),
    )
    with pytest.raises(MemoryError):
        quad.publish(CameraImage(1, 1, b"\0" * 4, 2.0, 2))
    assert native.isaacteleop_pip_configure_v2.calls[-1][1] == 0


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
    assert native.isaacteleop_pip_attach_v3.calls == [(1, 2, 3, 4, 5, 6, 7)]
    assert quad_module.os.environ["XR_ENABLE_API_LAYERS"] == "XR_APILAYER_OTHER"
    assert "XR_API_LAYER_PATH" not in quad_module.os.environ
    native.isaacteleop_pip_detach_v3.result = -1
    with pytest.raises(RuntimeError, match="retain OVXR"):
        quad.close()
    assert quad._attached
    native.isaacteleop_pip_detach_v3.result = 0
    quad.close()
    assert not quad._attached


def test_three_quad_wrappers_keep_independent_revisions(tmp_path, monkeypatch):
    first, native = _quad(tmp_path, monkeypatch)
    second = OpenXRCameraQuad(str(tmp_path / "layer.json"), camera_index=1)
    third = OpenXRCameraQuad(str(tmp_path / "layer.json"), camera_index=2)
    for quad in (first, second, third):
        quad.configure(AuxiliaryRasterConfig(space="head_locked"))
        assert quad.publish(CameraImage(1, 1, b"abcd", 1.0, 1))
    assert [call[0] for call in native.isaacteleop_pip_publish_rgba_v2.calls] == [
        0,
        1,
        2,
    ]
    second.publish(None, invalid=True)
    assert native.isaacteleop_pip_configure_v2.calls[-1][:2] == (1, 0)
    assert not first.publish(CameraImage(1, 1, b"abcd", 1.0, 1))
    assert third.publish(CameraImage(1, 1, b"efgh", 2.0, 2))


def test_old_native_layer_is_rejected_before_upload(tmp_path, monkeypatch):
    _quad(tmp_path, monkeypatch)
    monkeypatch.setattr(quad_module.ctypes, "CDLL", lambda _path: SimpleNamespace())
    with pytest.raises(AttributeError, match="isaacteleop_pip_configure_v2"):
        OpenXRCameraQuad(str(tmp_path / "layer.json"))


@pytest.mark.parametrize("result", [0, -6, -7])
def test_gpu_quad_completion_controls_source_retirement(tmp_path, monkeypatch, result):
    quad, native = _quad(tmp_path, monkeypatch)
    quad.supports_gpu = True
    quad.configure(AuxiliaryRasterConfig(space="head_locked"))
    gpu = SimpleNamespace(
        closed=False, busy=False, is_array=True, pointer=123, device=1
    )
    image = CameraImage(720, 450, None, 1.0, 1, gpu)
    native.isaacteleop_pip_publish_cuda_array_v2.result = result
    assert quad.publish(image) == (result == 0)
    assert gpu.busy == (result == -7)
    assert len(native.isaacteleop_pip_publish_cuda_array_v2.calls) == 1
    assert native.isaacteleop_pip_publish_rgba_v2.calls == []

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EGO source configuration and window-only stereo adapter contracts."""

from contextlib import nullcontext
from dataclasses import fields
from types import SimpleNamespace
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

from isaaccapture_examples.camera_viz import config
from isaaccapture_examples.camera_viz.pipeline import Frame, FrameSource, SourceSpec
from repo_paths import repo_root
from isaaccapture_examples.camera_viz.sources import build_local_camera
from isaaccapture_examples.camera_viz.sources.stereo_sbs import StereoSideBySideSource


class StaticSource(FrameSource):
    def __init__(self, frame=None):
        self.frame = frame
        self.started = False

    @property
    def spec(self):
        return SourceSpec("ego", 2, 1)

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def latest(self):
        frame, self.frame = self.frame, None
        return frame


def test_example_is_stereo_and_defaults_to_off_without_opening_hardware(capsys):
    path = repo_root() / "examples/camera_viz/configs/ego.yaml"
    cfg = yaml.safe_load(path.read_text())
    assert config.resolve_stereo_debug(cfg, None, "window") == "off"
    (source,) = build_local_camera(cfg["cameras"][0])
    assert (source.spec.width, source.spec.height) == (1600, 1300)
    assert source.latest() is None
    config.build_local_entries(cfg, is_xr=False)
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("field", ["width", "height", "fps"])
def test_ego_rejects_bad_dimensions_before_sdk_start(field):
    with pytest.raises(ValueError, match="positive"):
        build_local_camera({"name": "ego", "type": "ego", "stereo": True, field: 0})


def test_ego_requires_stereo():
    with pytest.raises(ValueError, match="stereo"):
        build_local_camera({"name": "ego", "type": "ego"})


def test_ego_stop_retains_a_handle_owned_by_a_live_sdk_thread():
    (source,) = build_local_camera({"name": "ego", "type": "ego", "stereo": True})
    joins = []
    thread = SimpleNamespace(
        join=lambda timeout: joins.append(timeout), is_alive=lambda: True
    )
    handle = object()
    source._thread, source._capture = thread, handle
    source.stop()
    assert joins == [5.0]
    assert source._thread is thread and source._capture is handle
    assert source._stop.is_set()


@pytest.mark.parametrize("fails", [False, True])
def test_ego_producer_closes_capture_after_normal_or_failed_exit(monkeypatch, fails):
    (source,) = build_local_camera({"name": "ego", "type": "ego", "stereo": True})
    closed = []
    handle = SimpleNamespace(close=lambda: closed.append("close"))

    def capture_loop():
        source._capture = handle
        if fails:
            raise RuntimeError("decode failed")

    monkeypatch.setattr(source, "_capture_loop", capture_loop)
    source._produce_loop()
    assert closed == ["close"]
    assert source._capture is None


@pytest.mark.parametrize(
    "requested,renderer,current,expected",
    [(0, 2, 0, 2), (0, 0, 2, 0), (2, 2, 0, 2), (0, None, 2, 2), (1, None, 2, 1)],
)
def test_ego_uses_renderer_gpu_or_standalone_selection(
    monkeypatch, requested, renderer, current, expected
):
    class Thread:
        def __init__(self, **kwargs):
            self.started = False

        def start(self):
            self.started = True

    devices = []

    def device(identifier):
        devices.append(identifier)
        return nullcontext()

    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            empty=np.empty,
            uint8=np.uint8,
            cuda=SimpleNamespace(
                Device=device, runtime=SimpleNamespace(getDevice=lambda: current)
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "isaaccapture_examples.camera_viz.ego_preview",
        SimpleNamespace(Capture=object),
    )
    monkeypatch.setattr(
        "isaaccapture_examples.camera_viz.sources.ego.threading.Thread", Thread
    )
    cfg = {
        "cameras": [{"name": "ego", "type": "ego", "stereo": True, "gpu_id": requested}]
    }
    (entry,) = config.build_local_entries(
        cfg, is_xr=False, renderer_cuda_device_id=renderer
    )
    source = entry.source
    source.start()
    assert devices == [expected] and source._gpu_id == expected
    assert source._thread.started


@pytest.mark.parametrize("requested,renderer", [(1, 2), (2, 0), (-1, 0), (0, -1)])
def test_ego_rejects_gpu_conflicts_before_imports_or_allocation(
    monkeypatch, requested, renderer
):
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setitem(
        sys.modules, "isaaccapture_examples.camera_viz.ego_preview", None
    )
    (source,) = build_local_camera(
        {"name": "ego", "type": "ego", "stereo": True, "gpu_id": requested},
        renderer_cuda_device_id=renderer,
    )
    with pytest.raises(ValueError, match="CUDA"):
        source.start()
    assert source._capture is None and source._thread is None
    assert not source._buffers_left and not source._buffers_right


def test_renderer_gpu_context_does_not_change_other_sources(monkeypatch):
    # SyntheticSource allocates in __init__; factory tests must stay GPU-free.
    constructor = MagicMock()
    monkeypatch.setattr(
        "isaaccapture_examples.camera_viz.sources.SyntheticSource", constructor
    )
    spec = {"name": "synthetic", "type": "synthetic", "width": 4, "height": 2}
    (source,) = build_local_camera(spec, renderer_cuda_device_id=2)
    assert source is constructor.return_value
    constructor.assert_called_once_with(
        name="synthetic", width=4, height=2, fps=60.0, hue_speed_hz=0.25
    )


def test_ego_mjpeg_conversion_drop_keeps_stereo_publication_and_closes_sdk(monkeypatch):
    class Array(np.ndarray):
        def set(self, data):
            np.copyto(self, data)

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def synchronize(self):
            pass

    (source,) = build_local_camera(
        {
            "name": "ego",
            "type": "ego",
            "stereo": True,
            "format": "mjpg",
            "width": 2,
            "height": 1,
        }
    )
    left, right = bytes([1, 2, 3, 4, 5, 6]), bytes([7, 8, 9, 10, 11, 12])
    closed = []

    class Capture:
        def __init__(self, *args):
            self.calls = 0

        def next_pair(self, timeout):
            self.calls += 1
            if self.calls == 2:
                return {"left": left, "right": right, "timestamp_ns": 123}
            if self.calls == 3:
                source._stop.set()
            return None

        def close(self):
            closed.append(True)

    monkeypatch.setitem(
        sys.modules,
        "isaaccapture_examples.camera_viz.ego_preview",
        SimpleNamespace(Capture=Capture),
    )
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            empty=lambda shape, dtype: np.empty(shape, dtype=dtype).view(Array),
            uint8=np.uint8,
            cuda=SimpleNamespace(
                Device=lambda device: nullcontext(),
                Stream=lambda **kwargs: Stream(),
            ),
        ),
    )
    monkeypatch.setattr(
        "isaaccapture_examples.camera_viz.sources.ego.alloc_pinned_host", np.empty
    )
    source._buffers_left = [np.full((1, 2, 4), 255, dtype=np.uint8)]
    source._buffers_right = [np.full((1, 2, 4), 255, dtype=np.uint8)]
    source._produce_loop()
    frame = source.latest()
    assert frame is not None and frame.timestamp_ns == 123
    np.testing.assert_array_equal(frame.image[..., :3], [[[1, 2, 3], [4, 5, 6]]])
    np.testing.assert_array_equal(
        frame.image_right[..., :3], [[[7, 8, 9], [10, 11, 12]]]
    )
    assert source._frames == 1 and source._incomplete_pairs == 2
    assert source.latest() is None
    assert closed == [True] and source._capture is None


def test_sbs_cli_override_and_xr_rejection():
    cfg = {"display": {"window": {"stereo_debug": "sbs"}}}
    assert config.resolve_stereo_debug(cfg, "off", "xr") == "off"
    assert config.resolve_stereo_debug(cfg, None, "window") == "sbs"
    with pytest.raises(ValueError, match="requires --mode window"):
        config.resolve_stereo_debug(cfg, None, "xr")
    with pytest.raises(ValueError, match=r"off\|sbs"):
        config.resolve_stereo_debug({}, "invalid", "window")


def test_sbs_preserves_nv_entry_fields_and_mono_identity():
    source = StaticSource()
    entry = config.SourceEntry(
        source,
        object(),
        stereo=True,
        stereo_plane_distance_cm=3,
        lock_mode="world",
        placement_config=object(),
        compositor="televiz",
        cylinder_radius_m=4,
        cylinder_angle_deg=120,
        equirect_yaw_deg=30,
    )
    mono = config.SourceEntry(StaticSource(), None)
    result = config.apply_stereo_debug([entry, mono], "sbs")
    assert not result[0].stereo
    assert isinstance(result[0].source, StereoSideBySideSource)
    assert result[1] is mono
    for field in fields(entry):
        if field.name not in ("source", "stereo"):
            assert getattr(result[0], field.name) == getattr(entry, field.name)


def test_sbs_rejects_a_mono_frame_from_a_declared_stereo_source():
    wrapper = StereoSideBySideSource(StaticSource(Frame(object(), 123, "ego")))
    with pytest.raises(RuntimeError, match="requires a paired source"):
        wrapper.latest()


def test_sbs_copies_left_then_right_on_the_frame_device_and_stream(monkeypatch):
    class Array(np.ndarray):
        @property
        def device(self):
            return SimpleNamespace(id=2)

    class Stream:
        def __init__(self, pointer):
            self.pointer = pointer
            self.synced = False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def synchronize(self):
            self.synced = True

    streams, devices = [], []

    def external_stream(pointer):
        stream = Stream(pointer)
        streams.append(stream)
        return stream

    def device(identifier):
        devices.append(identifier)
        return nullcontext()

    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            empty=lambda shape, dtype: np.empty(shape, dtype=dtype).view(Array),
            uint8=np.uint8,
            cuda=SimpleNamespace(Device=device, ExternalStream=external_stream),
        ),
    )
    left = np.full((1, 2, 4), 17, dtype=np.uint8).view(Array)
    right = np.full((1, 2, 4), 29, dtype=np.uint8).view(Array)
    source = StaticSource(Frame(left, 123, "ego", stream=456, image_right=right))
    wrapper = StereoSideBySideSource(source)
    wrapper.start()
    frame = wrapper.latest()
    assert frame.image.shape == (1, 4, 4)
    np.testing.assert_array_equal(frame.image[:, :2], left)
    np.testing.assert_array_equal(frame.image[:, 2:], right)
    assert (frame.timestamp_ns, frame.source_id, frame.stream, frame.image_right) == (
        123,
        "ego.sbs",
        456,
        None,
    )
    assert devices == [2]
    assert streams[0].pointer == 456 and streams[0].synced
    assert wrapper.latest() is None
    wrapper.stop()
    assert not source.started

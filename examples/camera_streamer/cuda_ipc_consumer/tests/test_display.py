# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Viewer layout: one tile per stream, placeholders for quiet ones, stream updates."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

import consumer  # noqa: E402

INFO = {
    "cam0": {"width": 320, "height": 180},
    "cam1": {"width": 320, "height": 180},
    "cam2": {"width": 640, "height": 360},
}
ORDER = ["cam0", "cam1", "cam2"]


def image(name, value):
    info = INFO[name]
    return np.full((info["height"], info["width"], 3), value, dtype=np.uint8)


def tiles(latest, seen, now=10.0, scale=1.0, labels=False):
    return consumer._stream_tiles(ORDER, INFO, latest, seen, now, scale, 1.0, labels)


def is_placeholder(tile):
    # Placeholders have a red border; test images are uniform grey.
    return tuple(tile[0, 0]) == (0, 0, 200)


def test_scale_is_native_for_small_streams_and_reduced_for_large():
    assert consumer._display_scale(None, INFO) == 1.0
    assert consumer._display_scale(None, {"head": {"width": 2560}}) == 0.35
    assert consumer._display_scale(2.0, {"head": {"width": 2560}}) == 2.0


def test_live_streams_show_their_image_at_stream_size():
    latest = {name: image(name, 90) for name in ORDER}
    result = tiles(latest, dict.fromkeys(ORDER, 9.9))
    assert [tile.shape[:2] for tile in result] == [(180, 320), (180, 320), (360, 640)]
    assert not any(is_placeholder(tile) for tile in result)


def test_stream_that_never_delivered_gets_a_placeholder_of_its_size():
    latest = {"cam0": image("cam0", 90), "cam2": image("cam2", 90)}
    result = tiles(latest, {"cam0": 9.9, "cam2": 9.9})
    assert [is_placeholder(tile) for tile in result] == [False, True, False]
    assert result[1].shape[:2] == (180, 320)


def test_stream_that_stops_turns_into_a_placeholder():
    latest = {name: image(name, 90) for name in ORDER}
    seen = {"cam0": 9.9, "cam1": 9.9, "cam2": 8.5}
    assert [is_placeholder(tile) for tile in tiles(latest, seen)] == [
        False,
        False,
        True,
    ]
    # It comes back as soon as a frame arrives.
    seen["cam2"] = 10.0
    assert not is_placeholder(tiles(latest, seen)[2])


def test_no_streams_shows_a_waiting_tile():
    (tile,) = consumer._stream_tiles([], {}, {}, {}, 0.0, 1.0, 1.0, True)
    assert tile.shape[:2] == (360, 640) and is_placeholder(tile)


def test_labels_do_not_modify_the_cached_image():
    latest = {name: image(name, 90) for name in ORDER}
    tiles(latest, dict.fromkeys(ORDER, 9.9), labels=True)
    assert int(latest["cam0"].min()) == 90 and int(latest["cam0"].max()) == 90


def test_grid_fits_three_mixed_size_tiles_in_one_window():
    latest = {name: image(name, 90) for name in ORDER}
    canvas = consumer._compose_grid(tiles(latest, dict.fromkeys(ORDER, 9.9)))
    # Two 320x180 tiles side by side above the 640x360 tile.
    assert canvas.shape[:2] == (540, 640)
    assert consumer._compose_grid(tiles(latest, dict.fromkeys(ORDER, 9.9)), 3).shape[
        :2
    ] == (
        360,
        1280,
    )
    assert consumer._compose_grid([]) is None


def fake_cupy(closed):
    module = ModuleType("cupy")
    module.uint8 = np.uint8
    handles = iter(range(1, 1000))
    module.ndarray = lambda shape, dtype, memptr, strides: np.zeros(shape, dtype)
    module.cuda = SimpleNamespace(
        UnownedMemory=lambda ptr, size, owner: (ptr, size, owner),
        MemoryPointer=lambda memory, offset: memory,
        runtime=SimpleNamespace(
            cudaIpcMemLazyEnablePeerAccess=1,
            ipcOpenMemHandle=lambda _handle, _flags: next(handles),
            ipcCloseMemHandle=closed.append,
        ),
    )

    class Device:
        def __init__(self, _gpu):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    module.cuda.Device = Device
    return module


def catalog_entry(name, epoch, slots=2):
    return {
        "name": name,
        "width": 320,
        "height": 180,
        "gpu_id": 0,
        "epoch": epoch,
        "slots": [
            {
                "slot": i,
                "handle": "AAAA",
                "nbytes": 86400,
                "shape": [270, 320],
                "strides": [320, 1],
            }
            for i in range(slots)
        ],
    }


def test_client_follows_stream_updates(monkeypatch):
    closed = []
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy(closed))
    client = consumer.CudaIpcClient("unused")

    assert client.apply_streams(
        [catalog_entry("cam0", 1), catalog_entry("cam1", 2)]
    ) == {
        "cam0",
        "cam1",
    }
    assert client.stream_order == ["cam0", "cam1"] and len(client.slots) == 4

    # The same catalog again changes nothing and reopens nothing.
    assert (
        client.apply_streams([catalog_entry("cam0", 1), catalog_entry("cam1", 2)])
        == set()
    )
    assert closed == []

    # cam1 leaves, cam0 is registered again (new epoch), cam2 appears.
    changed = client.apply_streams([catalog_entry("cam0", 3), catalog_entry("cam2", 4)])
    assert changed == {"cam0", "cam1", "cam2"}
    assert client.stream_order == ["cam0", "cam2"]
    assert len(closed) == 4 and len(client.slots) == 4
    assert client.stream_info["cam0"]["epoch"] == 3

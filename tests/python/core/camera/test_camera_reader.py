# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The camera reader, against the camera plugin shipped in the package."""

import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from isaaccapture import camera
from isaaccapture.camera import (
    CameraDisconnected,
    CameraReader,
    CameraReaderError,
    CameraWindowExhausted,
)

PLUGIN = (
    Path(camera.__file__).resolve().parent.parent
    / "plugins"
    / "camera"
    / "camera_plugin"
)


def wait_until(condition, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def short_directory():
    """A temporary directory with a short path: a socket path is limited to 107 bytes."""
    path = Path(tempfile.mkdtemp(prefix="cam", dir="/tmp"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def plugin(short_directory):
    """A running plugin with synthetic cameras; yields its endpoint."""
    endpoint = str(short_directory / "cameras.sock")
    process = subprocess.Popen(
        [
            str(PLUGIN),
            f"--endpoint={endpoint}",
            "--camera=head:synthetic:width=160,height=120,fps=120",
            "--camera=wrist:synthetic:width=64,height=48,streams=1,format=gray8,fps=120",
            "--camera=sky:synthetic:width=64,height=48,streams=1,format=nv12,fps=120",
            "--camera=slow:synthetic:width=64,height=48,streams=1,format=gray8,fps=1",
        ]
    )

    def serving():
        try:
            return bool(CameraReader.list(endpoint=endpoint))
        except CameraReaderError:
            return False

    try:
        assert wait_until(serving), "the camera plugin did not start serving"
        yield endpoint, process
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=10)


def assert_synthetic(image, stream):
    """Every byte of a synthetic image is known from its frame number and its stream."""
    data = image.data
    frame = int.from_bytes(data[:8].tobytes(), "little")
    assert image.device_frame_id == frame
    assert np.all(data[8:] == (frame + 17 * stream) & 0xFF)


def test_a_reader_lists_what_the_plugin_publishes(plugin):
    endpoint, _ = plugin
    groups = CameraReader.list(endpoint=endpoint)
    assert [group.id for group in groups] == ["head", "wrist", "sky", "slow"]
    assert all(group.unsupported == "" for group in groups)
    left, right = groups[0].streams
    assert (left.id, right.id) == ("head/left", "head/right")
    assert (left.width, left.height, left.pixel_format, left.codec) == (
        160,
        120,
        "bgra8",
        "none",
    )
    assert left.strides == [640]
    assert left.frame_rate == 120.0
    assert groups[2].streams[0].pixel_format == "nv12"
    assert groups[2].streams[0].plane_count == 2


def test_images_are_read_only_views_of_the_plugins_buffers(plugin):
    endpoint, _ = plugin
    with CameraReader.open("head", endpoint=endpoint) as reader:
        assert reader.group.id == "head"
        assert reader.group.session_id != 0
        with reader.next(2.0) as frames:
            assert len(frames) == 2
            assert frames.sequence >= 1
            assert (
                frames.timeline.capture_ns
                <= frames.timeline.sdk_ready_ns
                <= frames.timeline.published_ns
            )
            for stream in range(2):
                image = frames.image(stream)
                assert image.stream.id == ("head/left", "head/right")[stream]
                assert_synthetic(image, stream)
                pixels = image.array()
                assert pixels.shape == (120, 160, 4)
                assert pixels.dtype == np.uint8
                assert pixels.base is not None  # a view, not a copy
                assert not pixels.flags.writeable
                with pytest.raises(ValueError):
                    pixels[0, 0, 0] = 1
            assert (
                frames.image(0).array()[5, 7, 2]
                == frames.image(0).data[5 * 640 + 7 * 4 + 2]
            )
            with pytest.raises(IndexError):
                frames.image(2)


def test_each_pixel_format_has_its_shape(plugin):
    endpoint, _ = plugin
    with (
        CameraReader.open("wrist", endpoint=endpoint) as reader,
        reader.next(2.0) as frames,
    ):
        gray = frames.image(0)
        assert gray.array().shape == (48, 64)
        assert_synthetic(gray, 0)

    with (
        CameraReader.open("sky", endpoint=endpoint) as reader,
        reader.next(2.0) as frames,
    ):
        nv12 = frames.image(0)
        with pytest.raises(ValueError):
            nv12.array()
        assert nv12.plane(0).shape == (48, 64)
        assert nv12.plane(1).shape == (24, 32, 2)
        assert nv12.data.shape == (64 * 48 * 3 // 2,)
        with pytest.raises(IndexError):
            nv12.plane(2)


def test_a_loop_keeps_the_array_of_one_frame_while_it_asks_for_the_next(plugin):
    endpoint, _ = plugin
    with CameraReader.open("head", endpoint=endpoint) as reader:
        previous = None
        for count, frames in enumerate(reader):
            # `pixels` still names the array of the frame before while the next frame set is
            # asked for. The default window of two is what lets this loop run.
            pixels = frames.image(0).array()
            assert_synthetic(frames.image(0), 0)
            assert pixels.shape == (120, 160, 4)
            if previous is not None:
                assert frames.sequence == previous + 1 + frames.skipped
            previous = frames.sequence
            if count == 30:
                break


def test_an_array_shows_its_frame_for_as_long_as_it_exists(plugin):
    endpoint, _ = plugin
    reader = CameraReader.open("head", endpoint=endpoint)
    frames = reader.next(2.0)
    pixels = frames.image(0).array()
    kept = pixels.copy()

    # The frame set is let go of, the reader is closed and both are forgotten. The array still
    # holds the frame: the camera goes through its pool many times and the pixels stay.
    frames.release()
    reader.close()
    del frames, reader
    time.sleep(0.3)
    assert np.array_equal(pixels, kept)
    assert int.from_bytes(pixels.tobytes()[:8], "little") == int.from_bytes(
        kept.tobytes()[:8], "little"
    )


def test_a_reader_that_was_busy_gets_what_the_camera_saw_last(plugin):
    endpoint, _ = plugin
    with CameraReader.open("head", endpoint=endpoint) as reader:
        reader.next(2.0).release()
        for _ in range(3):
            time.sleep(
                0.3
            )  # busy elsewhere, holding nothing; the camera runs at 120 fps
            frames = reader.next(2.0)
            age_ms = (time.monotonic_ns() - frames.timeline.published_ns) / 1e6
            # A frame set put aside when the reader left would be 300 ms old.
            assert 0 <= age_ms < 200
            assert frames.skipped > 10
            frames.release()


def test_a_full_window_is_an_error_that_says_what_holds_it(plugin):
    endpoint, _ = plugin
    with CameraReader.open("head", endpoint=endpoint, window=1) as reader:
        held = reader.next(2.0)
        assert held is not None
        with pytest.raises(CameraWindowExhausted, match="holds 1 frame set"):
            reader.next(0.1)
        first = held.image(0).data[:8].tobytes()
        time.sleep(0.1)  # the camera has moved on a dozen frames
        # The held frame was not rewritten.
        assert held.image(0).data[:8].tobytes() == first

        # Letting go of the frame set is not enough while an array of it is kept.
        pixels = held.image(0).array()
        held.release()
        with pytest.raises(RuntimeError, match="given back"):
            held.image(0)
        with pytest.raises(
            CameraWindowExhausted, match="An array taken from a frame set"
        ):
            reader.next(0.1)
        del pixels
        newer = reader.next(2.0)
        assert newer is not None
        assert newer.skipped > 0
        assert (
            newer.sequence > held.sequence
        )  # what is known of a frame set stays readable
        newer.release()


def test_a_borrowed_frame_set_goes_back_when_its_block_ends(plugin):
    endpoint, _ = plugin
    with CameraReader.open("head", endpoint=endpoint, window=1) as reader:
        with reader.next(2.0).borrow() as frames:
            image = frames.image(0)
            pixels = image.array()
            assert pixels.shape == (120, 160, 4)
            assert_synthetic(image, 0)
            del pixels
        # Given back at the end of the block: its images are no longer to be had, and the
        # reader's window of one is free again.
        with pytest.raises(RuntimeError, match="given back"):
            image.array()
        with pytest.raises(RuntimeError, match="given back"):
            frames.image(0)

        # Leaving the block while an array still exists is an error, not a frame set that
        # stays held without a word. The array itself is still good: it keeps its frame.
        with pytest.raises(BufferError, match="still exists at the end of the block"):
            with reader.next(2.0).borrow() as frames:
                kept = frames.image(0).array()
        before = kept.copy()
        time.sleep(0.1)
        assert np.array_equal(kept, before)
        del kept
        assert reader.next(2.0) is not None

    # An error inside the block is the one that is seen.
    with CameraReader.open("head", endpoint=endpoint) as reader:
        with pytest.raises(KeyError):
            with reader.next(2.0).borrow() as frames:
                kept = frames.image(0).array()
                assert kept.shape == (120, 160, 4)
                raise KeyError("from the block")


def test_an_in_order_reader_gets_every_frame_set(plugin):
    endpoint, _ = plugin
    with CameraReader.open(
        "wrist", endpoint=endpoint, window=3, order="in_order"
    ) as reader:
        held = [reader.next(2.0) for _ in range(3)]
        assert all(frames is not None for frames in held)
        with pytest.raises(CameraWindowExhausted):
            reader.next(0.05)  # three are held together
        # Each follows the one before; whatever was passed over is counted, never lost.
        for before, after in zip(held, held[1:]):
            assert after.sequence == before.sequence + 1 + after.skipped
        for frames in held:
            assert_synthetic(frames.image(0), 0)
            frames.release()


def test_a_reader_says_why_it_could_not_be_opened(plugin, short_directory):
    endpoint, _ = plugin
    with pytest.raises(CameraReaderError, match="no group 'chest'"):
        CameraReader.open("chest", endpoint=endpoint)
    with pytest.raises(CameraReaderError, match="does not fit the pool"):
        CameraReader.open("head", endpoint=endpoint, window=8)
    with pytest.raises(ValueError, match="in_order"):
        CameraReader.open("head", endpoint=endpoint, order="sideways")
    with pytest.raises(CameraReaderError, match="no camera plugin answers on"):
        CameraReader.open("head", endpoint=str(short_directory / "nobody.sock"))
    closed = CameraReader.open("head", endpoint=endpoint)
    closed.close()
    with pytest.raises(CameraReaderError, match="closed"):
        closed.next(0.1)


def test_closing_from_another_thread_ends_a_wait(plugin):
    endpoint, _ = plugin
    reader = CameraReader.open("slow", endpoint=endpoint)
    # Taken right as it is published, so the next frame set is a second away.
    reader.next(3.0).release()
    reader.next(3.0).release()

    outcome = []

    def wait():
        try:
            outcome.append(reader.next(5.0))
        except CameraReaderError as error:
            outcome.append(error)

    thread = threading.Thread(target=wait)
    thread.start()
    time.sleep(0.2)  # the thread now waits, with the interpreter lock released
    reader.close()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert isinstance(outcome[0], CameraReaderError)
    assert "closed" in str(outcome[0])


def test_a_reader_is_told_when_the_plugin_is_gone(plugin):
    endpoint, process = plugin
    reader = CameraReader.open("head", endpoint=endpoint)
    process.send_signal(signal.SIGTERM)
    process.wait(timeout=10)
    with pytest.raises(CameraDisconnected):
        for _ in range(100):  # what was already on its way is still delivered
            frames = reader.next(2.0)
            if frames is not None:
                frames.release()


def test_the_plugin_manager_starts_the_plugin_of_the_package(
    short_directory, monkeypatch
):
    from isaaccapture.plugin_manager import PluginManager

    # A user's runtime and configuration directories, for this test and the plugin it starts.
    config = short_directory / "config" / "isaaccapture"
    config.mkdir(parents=True)
    (short_directory / "run").mkdir()
    (config / "cameras.yaml").write_text(
        "cameras:\n  head:\n    backend: synthetic\n    width: 160\n    height: 120\n    fps: 120\n"
    )
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(short_directory / "run"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(short_directory / "config"))
    socket_file = short_directory / "run" / "isaaccapture" / "camera.sock"

    # Nothing runs yet, and a reader does not start anything: it says where it looked.
    with pytest.raises(CameraReaderError, match="no camera plugin answers on"):
        CameraReader.open("head")
    assert not socket_file.exists()

    # The plugin in the package is started like any other plugin. Neither it nor the reader is
    # told where the other is.
    manager = PluginManager([str(PLUGIN.parent.parent)])
    assert "camera" in manager.get_plugin_names()
    plugin = manager.start("camera", "camera_test")
    try:
        assert wait_until(socket_file.is_socket)

        def opened():
            try:
                return CameraReader.open("head")
            except CameraReaderError:
                return None

        assert wait_until(lambda: opened() is not None)
        with CameraReader.open("head") as reader:
            for count, frames in enumerate(reader):
                assert_synthetic(frames.image(0), 0)
                if count == 5:
                    break
        plugin.check_health()
    finally:
        plugin.stop()

    # Stopped through its handle, the plugin leaves nothing behind.
    assert wait_until(lambda: not socket_file.exists())

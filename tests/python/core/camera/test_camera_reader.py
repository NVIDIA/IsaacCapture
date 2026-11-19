# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The camera reader, against the synthetic camera plugin shipped in the package.

The tests of cameras in CUDA memory need a CUDA device and CuPy, and are skipped without:
``uv run --extra dev --extra cuda pytest``.
"""

import select
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from isaaccapture.camera import (
    CameraDisconnected,
    CameraReader,
    CameraReaderError,
    CameraWindowExhausted,
)
from isaaccapture.teleop_session_manager import BUNDLED_PLUGINS_DIR

# The synthetic camera plugin among the plugins bundled in the package.
PLUGIN = BUNDLED_PLUGINS_DIR / "camera_synthetic" / "camera_synthetic_plugin"


def wait_until(condition, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


def published(**plugin):
    """The ids of the groups this user's camera plugins publish, or the plugin that is named."""
    try:
        return [group.id for group in CameraReader.list(**plugin)]
    except CameraReaderError:
        return []  # the plugin that was named does not serve yet


@pytest.fixture
def short_directory():
    """A temporary directory with a short path: a socket path is limited to 107 bytes."""
    path = Path(tempfile.mkdtemp(prefix="cam", dir="/tmp"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def endpoint_directory(short_directory, monkeypatch):
    """The endpoint directory of a user whose directories are this test's own.

    A plugin that is not told where to listen and a reader that looks for a plugin both look
    there, and the test's process and the plugins it starts share it. With the real ones a
    test would meet the developer's own plugins and cameras.yaml.
    """
    (short_directory / "run").mkdir()
    (short_directory / "config" / "isaaccapture").mkdir(parents=True)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(short_directory / "run"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(short_directory / "config"))
    return short_directory / "run" / "isaaccapture"


@pytest.fixture
def start_plugin(short_directory):
    """Starts plugin processes for a test; each is stopped and waited for when the test ends."""
    processes = []

    def start(*arguments):
        processes.append(subprocess.Popen([str(PLUGIN), *arguments]))
        return processes[-1]

    yield start
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@pytest.fixture
def plugin(short_directory, start_plugin):
    """A running plugin with synthetic cameras; gives its endpoint and its process."""
    endpoint = str(short_directory / "cameras.sock")
    process = start_plugin(
        f"--endpoint={endpoint}",
        "--camera=head:width=160,height=120,fps=120",
        "--camera=wrist:width=64,height=48,streams=1,format=gray8,fps=120",
        "--camera=sky:width=64,height=48,streams=1,format=nv12,fps=120",
        "--camera=slow:width=64,height=48,streams=1,format=gray8,fps=1",
    )
    assert wait_until(lambda: published(endpoint=endpoint)), (
        "the camera plugin did not start serving"
    )
    return endpoint, process


@pytest.fixture
def cuda_plugin(short_directory, start_plugin):
    """A running plugin whose synthetic cameras are in CUDA memory; gives its endpoint and CuPy."""
    cupy = pytest.importorskip("cupy")
    if not cupy.cuda.is_available():
        pytest.skip("no CUDA device")
    endpoint = str(short_directory / "cameras.sock")
    start_plugin(
        f"--endpoint={endpoint}",
        "--camera=head:memory=gpu,width=160,height=120,fps=120",
        "--camera=sky:memory=gpu,width=66,height=48,streams=1,format=nv12,fps=120",
    )
    assert wait_until(lambda: published(endpoint=endpoint)), (
        "the camera plugin did not start serving"
    )
    return endpoint, cupy


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
    # Each listing says which plugin publishes the group, also when the plugin was named.
    assert all(group.endpoint == endpoint for group in groups)
    left, right = groups[0].streams
    assert (left.id, right.id) == ("head/left", "head/right")
    assert (left.width, left.height, left.pixel_format) == (160, 120, "bgra8")
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


def test_an_image_is_asked_for_in_the_memory_it_is_in(plugin):
    endpoint, _ = plugin
    with CameraReader.open("head", endpoint=endpoint) as reader:
        stream = reader.group.streams[0]
        assert stream.memory == "host"
        assert not stream.on_gpu
        assert stream.cuda_device is None
        with reader.next(2.0) as frames:
            with pytest.raises(ValueError, match="host memory"):
                frames.image(0).cuda_array()
            with pytest.raises(ValueError, match="host memory"):
                frames.image(0).cuda_plane(0)


def test_an_image_in_cuda_memory_is_read_on_the_device(cuda_plugin):
    endpoint, cupy = cuda_plugin
    head, sky = CameraReader.list(endpoint=endpoint)
    assert [stream.memory for stream in head.streams] == ["cuda", "cuda"]
    assert all(stream.on_gpu for stream in head.streams)
    assert head.streams[0].cuda_device.startswith("GPU-")
    assert head.streams[0].cuda_device == sky.streams[0].cuda_device

    with CameraReader.open("head", endpoint=endpoint) as reader:
        with reader.next(2.0) as frames:
            for stream in range(2):
                image = frames.image(stream)
                plane = image.cuda_array()
                assert plane.shape == (120, 160, 4)
                assert plane.device == 0
                pixels = cupy.asarray(plane)
                assert pixels.shape == (120, 160, 4)
                assert pixels.dtype == cupy.uint8
                # A view of the plugin's buffer on the device, not a copy.
                assert pixels.data.ptr == plane.__cuda_array_interface__["data"][0]
                on_host = cupy.asnumpy(pixels).reshape(-1)
                frame = int.from_bytes(on_host[:8].tobytes(), "little")
                assert image.device_frame_id == frame
                assert np.all(on_host[8:] == (frame + 17 * stream) & 0xFF)
                # A kernel on the device reads it where it is.
                assert int(pixels[5:, :, :].max()) == (frame + 17 * stream) & 0xFF
                del pixels, plane

            image = frames.image(0)
            for on_the_host in (
                image.array,
                lambda: image.plane(0),
                lambda: image.data,
            ):
                with pytest.raises(ValueError, match="on a GPU"):
                    on_the_host()
            with pytest.raises(IndexError):
                image.cuda_plane(1)

    # Rows that are longer than their pixels, and two planes.
    with CameraReader.open("sky", endpoint=endpoint) as reader:
        with reader.next(2.0) as frames:
            image = frames.image(0)
            assert image.stream.strides == [66, 66]
            luma = cupy.asarray(image.cuda_plane(0))
            chroma = cupy.asarray(image.cuda_plane(1))
            assert luma.shape == (48, 66)
            assert chroma.shape == (24, 33, 2)
            assert chroma.data.ptr - luma.data.ptr == 66 * 48
            frame = image.device_frame_id
            assert int(luma[40, 60]) == frame & 0xFF
            assert int(chroma[20, 30, 1]) == frame & 0xFF
            with pytest.raises(ValueError, match="one plane"):
                image.cuda_array()
            del luma, chroma


def test_an_array_on_the_device_shows_its_frame_for_as_long_as_it_exists(cuda_plugin):
    endpoint, cupy = cuda_plugin
    with CameraReader.open("head", endpoint=endpoint, window=1) as reader:
        frames = reader.next(2.0)
        pixels = cupy.asarray(frames.image(0).cuda_array())
        kept = cupy.asnumpy(pixels)

        # The frame set is let go of; the array still holds it. The camera goes through its
        # pool many times and the pixels on the device stay, and the window stays taken.
        frames.release()
        with pytest.raises(RuntimeError, match="given back"):
            frames.image(0)
        with pytest.raises(
            CameraWindowExhausted, match="An array taken from a frame set"
        ):
            reader.next(0.1)
        time.sleep(0.3)
        assert np.array_equal(cupy.asnumpy(pixels), kept)

        del pixels
        newer = reader.next(2.0)
        assert newer is not None
        assert newer.skipped > 0
        newer.release()

        # A block a frame set was borrowed for ends with its arrays gone, on the device too.
        with pytest.raises(BufferError, match="still exists at the end of the block"):
            with reader.next(2.0).borrow() as frames:
                plane = frames.image(0).cuda_array()
        del plane
        assert reader.next(2.0) is not None


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
            left = time.monotonic_ns()
            time.sleep(
                0.3
            )  # busy elsewhere, holding nothing; the camera runs at 120 fps
            frames = reader.next(2.0)
            # A frame set put aside when the reader left would have been published then, with
            # nothing skipped. This one was published while the reader was away.
            assert (
                left + 150_000_000 < frames.timeline.published_ns <= time.monotonic_ns()
            )
            assert frames.skipped > 0
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


@pytest.mark.parametrize("order", ["newest", "in_order"])
def test_a_reader_is_read_from_an_event_loop(plugin, order):
    endpoint, _ = plugin
    with CameraReader.open("head", endpoint=endpoint, order=order) as reader:
        sequences = []

        def read_what_came():
            while (frames := reader.next(0)) is not None:
                sequences.append(frames.sequence)
                frames.release()

        if order == "newest":
            # Nothing is sent to a newest-order reader that has not asked.
            assert select.select([reader.fileno()], [], [], 0.2)[0] == []
        read_what_came()  # the first request
        deadline = time.monotonic() + 10.0
        while len(sequences) < 20 and time.monotonic() < deadline:
            if select.select([reader.fileno()], [], [], 2.0)[0]:
                read_what_came()
        # Each reading asked for the next frame set, so the loop kept being woken.
        assert len(sequences) >= 20
        assert sequences == sorted(set(sequences))


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


def test_a_reader_finds_the_plugin_that_serves_its_camera(
    endpoint_directory, start_plugin
):
    # Neither side is told where the other is: the plugin has no --endpoint and the reader no
    # endpoint and no instance.
    start_plugin("--camera=head:width=160,height=120,fps=120")
    assert wait_until(lambda: published() == ["head"])
    # The plugin's instance name is its backend's.
    assert (endpoint_directory / "camera-synthetic.sock").is_socket()
    with CameraReader.open("head") as reader:
        assert reader.group.id == "head"
        with reader.next(2.0) as frames:
            assert frames.image(0).array().shape == (120, 160, 4)
            assert_synthetic(frames.image(0), 0)


def test_a_listing_names_the_endpoint_of_its_plugin(endpoint_directory, start_plugin):
    assert CameraReader.list() == []  # no plugin running is not an error

    start_plugin(
        "--camera=head:width=160,height=120",
        "--camera=wrist:width=64,height=48,streams=1",
    )
    assert wait_until(lambda: published() == ["head", "wrist"])
    socket_file = str(endpoint_directory / "camera-synthetic.sock")
    groups = CameraReader.list()
    assert [(group.id, group.endpoint) for group in groups] == [
        ("head", socket_file),
        ("wrist", socket_file),
    ]
    assert (
        repr(groups[0]) == f"<GroupListing 'head' with 2 stream(s) on '{socket_file}'>"
    )
    with pytest.raises(AttributeError):
        groups[0].endpoint = "elsewhere"

    # The same when the plugin is named, by its socket path or by its instance.
    for named in ({"endpoint": socket_file}, {"instance": "synthetic"}):
        assert [(group.id, group.endpoint) for group in CameraReader.list(**named)] == [
            ("head", socket_file),
            ("wrist", socket_file),
        ]


def test_two_plugins_side_by_side_are_both_found_by_group(
    endpoint_directory, start_plugin
):
    # A second synthetic plugin runs beside the first under another instance name, with a
    # camera of its own.
    start_plugin("--camera=head:width=160,height=120,fps=120")
    start_plugin(
        "--instance=bench",
        "--camera=wrist:width=64,height=48,streams=1,format=gray8,fps=120",
    )
    assert wait_until(lambda: published() == ["wrist", "head"])

    # In the order of the socket paths, each group with the socket of the plugin that has it.
    assert [(group.id, group.endpoint) for group in CameraReader.list()] == [
        ("wrist", str(endpoint_directory / "camera-bench.sock")),
        ("head", str(endpoint_directory / "camera-synthetic.sock")),
    ]
    with CameraReader.open("head") as head, CameraReader.open("wrist") as wrist:
        with head.next(2.0) as frames:
            assert frames.image(0).array().shape == (120, 160, 4)
            assert_synthetic(frames.image(0), 0)
        with wrist.next(2.0) as frames:
            assert frames.image(0).array().shape == (48, 64)
            assert_synthetic(frames.image(0), 0)

    # A reader that names an instance reads from that plugin and no other.
    with CameraReader.open("wrist", instance="bench") as wrist:
        assert wrist.group.id == "wrist"
    with pytest.raises(CameraReaderError, match="no group 'wrist'"):
        CameraReader.open("wrist", instance="synthetic")

    # A listing that names an instance is that plugin's alone, and an error if it is not there.
    bench = CameraReader.list(instance="bench")
    assert [(group.id, group.endpoint) for group in bench] == [
        ("wrist", str(endpoint_directory / "camera-bench.sock"))
    ]
    with pytest.raises(CameraReaderError, match="no camera plugin answers on"):
        CameraReader.list(instance="absent")


def test_a_group_on_two_plugins_is_an_error_that_names_both(
    endpoint_directory, start_plugin
):
    start_plugin("--camera=head:width=64,height=48")
    start_plugin("--instance=second", "--camera=head:width=64,height=48")
    assert wait_until(lambda: published() == ["head", "head"])

    # The reader does not pick one of them.
    with pytest.raises(CameraReaderError) as error:
        CameraReader.open("head")
    message = str(error.value)
    assert "published by more than one camera plugin" in message
    assert f"'{endpoint_directory / 'camera-second.sock'}'" in message
    assert f"'{endpoint_directory / 'camera-synthetic.sock'}'" in message

    # Naming the plugin says which.
    with CameraReader.open("head", instance="second") as reader:
        assert reader.next(2.0) is not None


def test_a_group_nobody_publishes_is_an_error_that_says_where_the_reader_looked(
    endpoint_directory, start_plugin
):
    # No plugin runs.
    with pytest.raises(CameraReaderError) as error:
        CameraReader.open("head")
    message = str(error.value)
    assert f"in '{endpoint_directory}' publishes group 'head'" in message
    assert "No camera plugin is running there" in message

    # A plugin runs, with another camera: the error also says what is published, and where.
    start_plugin("--camera=head:width=64,height=48")
    assert wait_until(lambda: published() == ["head"])
    with pytest.raises(CameraReaderError) as error:
        CameraReader.open("chest")
    message = str(error.value)
    assert f"in '{endpoint_directory}' publishes group 'chest'" in message
    assert f"'head' by '{endpoint_directory / 'camera-synthetic.sock'}'" in message


def test_the_plugin_manager_starts_the_plugin_of_the_package(
    endpoint_directory, short_directory
):
    from isaaccapture.plugin_manager import PluginManager

    # The user's configuration file names the camera; the plugin is given no argument.
    (short_directory / "config" / "isaaccapture" / "cameras.yaml").write_text(
        "cameras:\n  head:\n    backend: synthetic\n    width: 160\n    height: 120\n    fps: 120\n"
    )
    socket_file = endpoint_directory / "camera-synthetic.sock"

    # Nothing runs yet, and a reader does not start anything: it says that none is running.
    with pytest.raises(CameraReaderError, match="No camera plugin is running there"):
        CameraReader.open("head")
    assert not socket_file.exists()

    # The plugin in the package is started like any other plugin. Neither it nor the reader is
    # told where the other is.
    manager = PluginManager([str(BUNDLED_PLUGINS_DIR)])
    assert "camera_synthetic" in manager.get_plugin_names()
    plugin = manager.start("camera_synthetic", "camera_test")
    try:
        assert wait_until(lambda: published() == ["head"])
        assert socket_file.is_socket()
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

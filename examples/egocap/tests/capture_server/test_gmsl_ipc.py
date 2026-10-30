"""The GMSL frame-feed camera source, against an in-process fake publisher speaking the receiver's protocol.

No GPU: the fake publisher keeps its "ring slots" as NumPy arrays and the test slot mapper hands them back
by handle, so the protocol client, the frame source, the launch factory and the CLI are all exercised
exactly as with the real feed except for the CUDA IPC mapping itself (``cuda_slots.py``).
"""

from __future__ import annotations

import asyncio
import base64
import json
import socket
import threading
import time

import cv2
import numpy as np
import pytest

from capture_server import protocol as p
from capture_server.client import CaptureClient
from capture_server.gmsl_ipc import FeedClient, FeedError, StreamInfo, probe_feed
from capture_server.gmsl_ipc import source as gmsl_source
from capture_server.launch import GmslCamera, build_gmsl_source_factory, gmsl_station_cameras
from capture_server.server import ServerConfig, _build_arg_parser, _gmsl_cameras, main
from tests.support.capture_server import start_capture_server, stop_capture_server

WIDTH, HEIGHT = 64, 48
RING = 3


def nv12_frame(seq: int, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """A synthetic NV12 frame (rows x pitch) whose luma encodes ``seq`` so a served frame is identifiable."""
    rows = np.zeros((height * 3 // 2, width), dtype=np.uint8)
    rows[:height] = (seq * 7) % 256
    rows[height:] = 128  # neutral chroma
    return rows


class FakePublisher:
    """A protocol-faithful stand-in for the receiver's CUDA IPC publisher over a Unix socket in tmp_path."""

    def __init__(self, socket_path: str, streams: dict[str, tuple[int, int]]) -> None:
        self.socket_path = socket_path
        self.rings = {name: [nv12_frame(0, w, h) for _ in range(RING)] for name, (w, h) in streams.items()}
        self.sizes = dict(streams)
        self.seq = dict.fromkeys(streams, 0)
        self.clients: list[tuple[socket.socket, frozenset[str]]] = []
        self.rejections: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(socket_path)
        self._server.listen(4)
        self._server.settimeout(0.2)
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _catalog(self, name: str, with_slots: bool) -> dict:
        width, height = self.sizes[name]
        entry = {
            "name": name,
            "width": width,
            "height": height,
            "pixel_format": "nv12",
            "dtype": "uint8",
            "gpu_id": 0,
            "slot_count": RING,
            "layout": {},
        }
        if with_slots:
            entry["slots"] = [
                {
                    "slot": i,
                    "handle": base64.b64encode(f"{name}/{i}".encode()).decode(),
                    "nbytes": int(self.rings[name][i].nbytes),
                    "shape": list(self.rings[name][i].shape),
                    "strides": [width, 1],
                }
                for i in range(RING)
            ]
        return entry

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                client, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                hello = {
                    "type": "hello",
                    "version": 2,
                    "features": ["subscriptions", "frame_metadata", "slot_generation", "nv12"],
                    "owner": "fake",
                    "socket_path": self.socket_path,
                    "streams": [self._catalog(n, False) for n in self.rings],
                }
                client.sendall((json.dumps(hello) + "\n").encode())
                line = b""
                while b"\n" not in line:
                    line += client.recv(4096)
                request = json.loads(line.split(b"\n", 1)[0])
                names = request.get("streams") or list(self.rings)
                unknown = [n for n in names if n not in self.rings]
                if unknown:
                    self.rejections.append(str(unknown))
                    client.sendall(
                        (json.dumps({"type": "error", "error": f"unknown stream(s): {unknown}"}) + "\n").encode()
                    )
                    client.close()
                    continue
                reply = {"type": "subscribed", "version": 2, "streams": [self._catalog(n, True) for n in names]}
                client.sendall((json.dumps(reply) + "\n").encode())
            except OSError:
                client.close()
                continue
            with self._lock:
                self.clients.append((client, frozenset(names)))

    def publish(self, name: str, metadata: dict | None = None) -> int:
        """Write the next frame of ``name`` into its ring and notify subscribers; returns its seq."""
        with self._lock:
            self.seq[name] += 1
            seq = self.seq[name]
            slot = seq % RING
            self.rings[name][slot][:] = nv12_frame(seq, *self.sizes[name])
            msg = {
                "type": "frame",
                "stream": name,
                "slot": slot,
                "slot_generation": seq // RING,
                "seq": seq,
                "timestamp_ns": seq * 16_666_667,
            }
            if metadata is not None:
                msg["metadata"] = metadata
            data = (json.dumps(msg) + "\n").encode()
            alive = []
            for client, names in self.clients:
                if name not in names:
                    alive.append((client, names))
                    continue
                try:
                    client.sendall(data)
                    alive.append((client, names))
                except OSError:
                    client.close()
            self.clients = alive
        return seq

    def close(self) -> None:
        self._stop.set()
        self._server.close()
        with self._lock:
            for client, _ in self.clients:
                client.close()
            self.clients = []


class FakeSlotMapper:
    """Hands back the fake publisher's ring arrays by the handle it advertised."""

    def __init__(self, publisher: FakePublisher) -> None:
        self._publisher = publisher
        self.closed: list[str] = []

    def open(self, stream: StreamInfo):
        mapper = self

        class Reader:
            def read(self, slot: int) -> np.ndarray:
                handle = base64.b64decode(stream.slots[slot]["handle"]).decode()
                name, index = handle.split("/")
                return mapper._publisher.rings[name][int(index)].copy()

            def close(self) -> None:
                mapper.closed.append(stream.name)

        return Reader()


@pytest.fixture
def publisher(tmp_path):
    pub = FakePublisher(str(tmp_path / "feed.sock"), {"cam0": (WIDTH, HEIGHT), "cam4": (32, 24)})
    try:
        yield pub
    finally:
        pub.close()


def wait_until(predicate, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met in time")


def test_probe_returns_the_subscribed_catalog(publisher):
    catalog = probe_feed(publisher.socket_path, ("cam0",))
    assert set(catalog) == {"cam0"}
    assert (catalog["cam0"].width, catalog["cam0"].height, catalog["cam0"].pixel_format) == (WIDTH, HEIGHT, "nv12")
    assert catalog["cam0"].slot_count == RING and len(catalog["cam0"].slots) == RING


def test_probe_rejects_an_unknown_stream_with_the_publishers_reason(publisher):
    with pytest.raises(FeedError, match="unknown stream"):
        probe_feed(publisher.socket_path, ("nope",))


def test_probe_fails_clearly_when_no_feed_is_listening(tmp_path):
    with pytest.raises(FeedError, match="cannot connect"):
        probe_feed(str(tmp_path / "missing.sock"))


def test_client_keeps_only_the_newest_notice_and_reads_its_slot(publisher):
    feed = FeedClient(publisher.socket_path, ("cam0",), slot_mapper=FakeSlotMapper(publisher))
    feed.connect()
    try:
        assert feed.latest("cam0") is None
        for _ in range(3):
            last = publisher.publish("cam0", {"sensor_timestamp_ns": 123})
        wait_until(lambda: feed.latest("cam0") is not None and feed.latest("cam0").seq == last)
        notice = feed.latest("cam0")
        assert notice.metadata == {"sensor_timestamp_ns": 123}
        pixels = feed.read_slot("cam0", notice.slot)
        assert pixels.shape == (HEIGHT * 3 // 2, WIDTH) and int(pixels[0, 0]) == (last * 7) % 256
    finally:
        feed.close()


def test_client_surfaces_a_publisher_shutdown_on_the_next_poll(publisher):
    feed = FeedClient(publisher.socket_path, ("cam0",), slot_mapper=FakeSlotMapper(publisher))
    feed.connect()
    try:
        publisher.close()

        def failed() -> bool:
            try:
                feed.latest("cam0")
            except FeedError:
                return True
            return False

        wait_until(failed, timeout_s=3.0)
    finally:
        feed.close()


def test_decode_nv12_crops_the_pitch_and_yields_bgr():
    info = StreamInfo("s", 4, 2, "nv12", 0, 1)
    rows = np.full((3, 8), 200, dtype=np.uint8)  # pitch 8 > width 4
    rows[2:] = 128
    bgr = gmsl_source.decode_to_bgr(rows, info)
    assert bgr.shape == (2, 4, 3) and bgr.dtype == np.uint8
    assert np.all(np.abs(bgr.astype(int) - 200) < 20)  # neutral chroma: grey of the luma value


def test_decode_rejects_an_unknown_pixel_format():
    with pytest.raises(ValueError, match="pixel format"):
        gmsl_source.decode_to_bgr(np.zeros((3, 4), dtype=np.uint8), StreamInfo("s", 4, 2, "yuyv", 0, 1))


def test_pipeline_serves_each_announced_frame_once_downscaled_as_jpeg(publisher):
    feed = FeedClient(publisher.socket_path, ("cam0",), slot_mapper=FakeSlotMapper(publisher))
    feed.connect()
    try:
        with gmsl_source.Pipeline(feed=feed, stream="cam0", output_size=(32, 24), jpeg_quality=90) as pipeline:
            queue = (
                pipeline.create(gmsl_source.node.VideoEncoder)
                .build(
                    pipeline.create(gmsl_source.node.Camera)
                    .build(gmsl_source.CameraBoardSocket.CAM_A)
                    .requestOutput((32, 24)),
                    frameRate=60,
                    profile=gmsl_source.VideoEncoderProperties.Profile.MJPEG,
                    quality=90,
                )
                .out.createOutputQueue()
            )
            pipeline.start()
            assert queue.tryGet() is None  # nothing announced yet
            seq = publisher.publish("cam0", {"sensor_timestamp_ns": 5_000_000_000})
            wait_until(lambda: feed.latest("cam0") is not None)
            frame = queue.tryGet()
            assert frame is not None and frame.getSequenceNum() == seq
            assert (frame.getWidth(), frame.getHeight()) == (32, 24)
            assert frame.getTimestamp() == pytest.approx(5.0)
            decoded = cv2.imdecode(np.frombuffer(frame.getData(), np.uint8), cv2.IMREAD_COLOR)
            assert decoded.shape == (24, 32, 3)
            # The same frame is never served twice; a newer one is, and intermediate ones are skipped.
            assert queue.tryGet() is None
            publisher.publish("cam0")
            newest = publisher.publish("cam0")
            wait_until(lambda: feed.latest("cam0").seq == newest)
            assert queue.tryGet().getSequenceNum() == newest
    finally:
        feed.close()


def test_pipeline_skips_a_frame_whose_slot_was_overwritten(publisher):
    """A notice older than the ring depth means the slot now holds a different frame: never serve it."""
    feed = FeedClient(publisher.socket_path, ("cam0",), slot_mapper=FakeSlotMapper(publisher))
    feed.connect()
    try:
        pipeline = gmsl_source.Pipeline(feed=feed, stream="cam0")
        pipeline.__enter__()
        first = publisher.publish("cam0")
        wait_until(lambda: feed.latest("cam0") is not None and feed.latest("cam0").seq == first)
        stale = feed.latest("cam0")
        for _ in range(RING):
            last = publisher.publish("cam0")
        wait_until(lambda: feed.latest("cam0").seq == last)
        # The encoder decided on ``stale`` before the ring wrapped: the copy is not that frame's bytes.
        assert pipeline._encode(stale) is None
        assert pipeline._encode(feed.latest("cam0")).getSequenceNum() == last
    finally:
        feed.close()


def _started_pipeline(feed, stream, **kwargs):
    """A started pipeline with its adapter-side queue, built through the SDK-shaped node API."""
    pipeline = gmsl_source.Pipeline(feed=feed, stream=stream, **kwargs).__enter__()
    camera_out = pipeline.create(gmsl_source.node.Camera).build(gmsl_source.CameraBoardSocket.CAM_A)
    queue = (
        pipeline.create(gmsl_source.node.VideoEncoder)
        .build(
            camera_out.requestOutput((8, 8)),
            frameRate=60,
            profile=gmsl_source.VideoEncoderProperties.Profile.MJPEG,
        )
        .out.createOutputQueue()
    )
    pipeline.start()
    return pipeline, queue


def test_pipeline_encodes_ahead_of_the_poll_and_delivers_in_order(publisher):
    """Frames announced between polls are already encoded when polled, and never come out of order."""
    feed = FeedClient(publisher.socket_path, ("cam0",), slot_mapper=FakeSlotMapper(publisher))
    feed.connect()
    try:
        pipeline, queue = _started_pipeline(feed, "cam0")
        seen = []
        for _ in range(12):
            seq = publisher.publish("cam0")
            wait_until(lambda: pipeline._pending and pipeline._pending[-1].getSequenceNum() == seq)
            frame = queue.tryGet()
            assert frame is not None
            seen.append(frame.getSequenceNum())
        assert seen == sorted(seen) and len(set(seen)) == len(seen)
        assert queue.tryGet() is None  # nothing new: an empty poll after the short grace, not a repeat
        pipeline.stop()
    finally:
        feed.close()


def test_pipeline_queue_is_bounded_and_keeps_the_newest_frames(publisher):
    """A consumer that stops polling never sees frames older than the queue depth."""
    feed = FeedClient(publisher.socket_path, ("cam0",), slot_mapper=FakeSlotMapper(publisher))
    feed.connect()
    try:
        pipeline, queue = _started_pipeline(feed, "cam0")
        published = []
        for _ in range(gmsl_source.QUEUE_DEPTH + 3):
            published.append(publisher.publish("cam0"))
            wait_until(lambda: pipeline._pending and pipeline._pending[-1].getSequenceNum() == published[-1])
        delivered = []
        while (frame := queue.tryGet()) is not None:
            delivered.append(frame.getSequenceNum())
        assert len(delivered) <= gmsl_source.QUEUE_DEPTH
        assert delivered == published[-len(delivered) :]
        pipeline.stop()
    finally:
        feed.close()


def test_a_slow_camera_does_not_stall_another(publisher, monkeypatch):
    """Each stream encodes on its own thread: a slow encode on one camera leaves the other's cadence intact."""
    feed = FeedClient(publisher.socket_path, ("cam0", "cam4"), slot_mapper=FakeSlotMapper(publisher))
    feed.connect()
    try:
        slow, slow_queue = _started_pipeline(feed, "cam0")
        fast, fast_queue = _started_pipeline(feed, "cam4")
        real_encode = slow._encode

        def slow_encode(notice):
            time.sleep(0.2)
            return real_encode(notice)

        monkeypatch.setattr(slow, "_encode", slow_encode)
        publisher.publish("cam0")
        fast_seq = publisher.publish("cam4")
        started = time.monotonic()
        wait_until(lambda: fast._pending and fast._pending[-1].getSequenceNum() == fast_seq, timeout_s=0.1)
        assert fast_queue.tryGet().getSequenceNum() == fast_seq
        assert time.monotonic() - started < 0.15  # well inside the slow camera's 200 ms encode
        wait_until(lambda: bool(slow._pending), timeout_s=1.0)
        assert slow_queue.tryGet() is not None
        slow.stop()
        fast.stop()
    finally:
        feed.close()


def test_stop_joins_the_encoder_thread_and_closes_the_queue(publisher):
    feed = FeedClient(publisher.socket_path, ("cam0",), slot_mapper=FakeSlotMapper(publisher))
    feed.connect()
    try:
        pipeline, queue = _started_pipeline(feed, "cam0")
        worker = pipeline._worker
        assert worker is not None and worker.is_alive()
        pipeline.stop()
        assert not worker.is_alive()
        assert not pipeline.isRunning()
        publisher.publish("cam0")
        assert queue.tryGet() is None  # closed queue never yields, even if frames keep arriving
    finally:
        feed.close()


def test_an_encoder_failure_is_raised_from_the_next_poll(publisher, monkeypatch):
    """A background encode failure surfaces where the adapter can report the camera as failed."""
    feed = FeedClient(publisher.socket_path, ("cam0",), slot_mapper=FakeSlotMapper(publisher))
    feed.connect()
    try:
        pipeline, queue = _started_pipeline(feed, "cam0")

        def broken(notice):
            raise RuntimeError("encoder exploded")

        monkeypatch.setattr(pipeline, "_encode", broken)
        publisher.publish("cam0")
        wait_until(lambda: pipeline._failure is not None)
        with pytest.raises(RuntimeError, match="encoder exploded"):
            queue.tryGet()
        pipeline.stop()
    finally:
        feed.close()


def test_feed_wait_for_newer_wakes_on_notice_failure_and_close(publisher):
    feed = FeedClient(publisher.socket_path, ("cam0",), slot_mapper=FakeSlotMapper(publisher))
    feed.connect()
    seq = None
    try:
        assert feed.wait_for_newer("cam0", None, 0.05) is None  # nothing announced: times out
        result = {}

        def waiter():
            result["notice"] = feed.wait_for_newer("cam0", None, 2.0)

        thread = threading.Thread(target=waiter)
        thread.start()
        seq = publisher.publish("cam0")
        thread.join(timeout=2.0)
        assert result["notice"].seq == seq
        assert feed.wait_for_newer("cam0", seq, 0.05) is None  # not newer than what we have
    finally:
        feed.close()
    # After close a wait for a frame that will never come returns immediately instead of sleeping it out.
    started = time.monotonic()
    assert feed.wait_for_newer("cam0", seq, 5.0) is None
    assert time.monotonic() - started < 1.0


def test_pipeline_fails_to_enter_for_an_unsubscribed_stream(publisher):
    feed = FeedClient(publisher.socket_path, ("cam0",), slot_mapper=FakeSlotMapper(publisher))
    feed.connect()
    try:
        with pytest.raises(RuntimeError, match="not in the frame feed subscription"):
            gmsl_source.Pipeline(feed=feed, stream="cam4").__enter__()
    finally:
        feed.close()


def test_station_cameras_take_the_feed_size_unless_overridden(publisher):
    catalog = probe_feed(publisher.socket_path)
    cameras = {"head": GmslCamera("cam0", (32, 24)), "wrist": GmslCamera("cam4")}
    stations = gmsl_station_cameras(cameras, catalog, fps=60.0, jpeg_quality=70)
    assert (stations["head"].width, stations["head"].height, stations["head"].device_id) == (32, 24, "cam0")
    assert (stations["wrist"].width, stations["wrist"].height, stations["wrist"].fps) == (32, 24, 60.0)
    with pytest.raises(ValueError, match="does not publish"):
        gmsl_station_cameras({"x": GmslCamera("cam9")}, catalog, fps=60.0, jpeg_quality=70)


def test_gmsl_camera_option_parsing():
    parsed = _gmsl_cameras(["head=cam0:1280x992", "wl=cam4"])
    assert parsed["head"] == GmslCamera("cam0", (1280, 992)) and parsed["wl"] == GmslCamera("cam4", None)
    for bad in ["head", "head=", "head=cam0:1280", "head=cam0:0x10"]:
        with pytest.raises(ValueError):
            _gmsl_cameras([bad])
    with pytest.raises(ValueError, match="twice"):
        _gmsl_cameras(["a=cam0", "a=cam4"])


def test_gmsl_flags_are_exclusive_with_mock_replay_and_oak_cameras(capsys, tmp_path):
    parser = _build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--mock", "--gmsl-ipc", "x.sock"])
    with pytest.raises(SystemExit):
        main(["--gmsl-ipc", "x.sock", "--gmsl-camera", "a=cam0", "--camera", "b=MXID"])
    assert "--camera only applies to real OAK-D" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--gmsl-camera", "a=cam0"])
    assert "--gmsl-camera only applies with --gmsl-ipc" in capsys.readouterr().err


def test_missing_feed_is_a_setup_error(tmp_path, capsys):
    assert main(["--gmsl-ipc", str(tmp_path / "none.sock"), "--gmsl-camera", "a=cam0", "--port", "0"]) == 2
    assert "cannot connect to the frame feed" in capsys.readouterr().err


async def test_live_feed_cameras_stream_through_the_server(publisher, unused_tcp_port):
    catalog = probe_feed(publisher.socket_path)
    cameras = {"head": GmslCamera("cam0", (32, 24)), "wrist": GmslCamera("cam4")}
    stations = gmsl_station_cameras(cameras, catalog, fps=60.0, jpeg_quality=80)
    mapper = FakeSlotMapper(publisher)
    factory = build_gmsl_source_factory(publisher.socket_path, cameras, stations, slot_mapper=mapper)
    assert factory.mock is False
    running = await start_capture_server(
        unused_tcp_port, source_factory=factory, config=ServerConfig(mock=False, cameras=stations)
    )
    try:
        started, images = [], []
        finished = asyncio.Event()
        client = CaptureClient(f"ws://127.0.0.1:{unused_tcp_port}")
        # A camera-only session: no tracking sources, so no IsaacCapture is needed.
        await client.start(
            sources=(),
            cameras=(
                {"name": "head", "type": p.CameraType.OAK_D_COLOR},
                {"name": "wrist", "type": p.CameraType.OAK_D_COLOR},
            ),
            on_started=lambda e: started.append(e),
            on_capture_image=lambda i: images.append(i),
            on_stopped=lambda r: finished.set(),
        )
        for _ in range(40):
            await asyncio.sleep(0.05)
            publisher.publish("cam0")
            publisher.publish("cam4")
        assert started and started[0].mode is p.CaptureMode.LIVE
        names = {i.name for i in images}
        assert names == {"head", "wrist"}, f"images only from {names}"
        head = next(i for i in images if i.name == "head")
        decoded = cv2.imdecode(np.frombuffer(head.jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (24, 32, 3)
        await client.stop()
        await asyncio.wait_for(finished.wait(), timeout=5)
        # The session's subscription was released with the acquisition.
        assert set(mapper.closed) == {"cam0", "cam4"}
    finally:
        await stop_capture_server(running)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace

from cuda_ipc import CudaIpcFramePublisher
from slot_guard import commit_slot, read_slot, write_slot


def test_delayed_broadcast_discards_frames_after_multiple_ring_wraps(tmp_path):
    publisher = CudaIpcFramePublisher(str(tmp_path / "ipc.sock"))
    paths = [tmp_path / str(i) for i in range(3)]
    for path in paths:
        path.write_bytes(bytes(8))
    publisher._streams["eye"] = SimpleNamespace(
        slots=[SimpleNamespace(guard_path=str(p)) for p in paths]
    )
    publisher._started = True
    pixels = [None] * 3
    for seq in range(1, 8):
        slot = (seq - 1) % 3
        with publisher.write_slot("eye", slot, publisher._stop):
            pixels[slot] = seq
        publisher.publish("eye", slot, (seq - 1) // 3 + 1, seq, seq)
    sent = []
    publisher._clients = [SimpleNamespace(streams={"eye"}, sock=None)]

    def send(sock, msg):
        with publisher._current_frame(msg) as valid:
            assert valid
            assert pixels[msg["slot"]] == msg["seq"]
        sent.append(msg["seq"])
        if msg["seq"] == 7:
            publisher._stop.set()

    publisher._send_json = send
    publisher._broadcast_loop()
    assert sent == [5, 6, 7]
    # A message already in the socket must also be checked at consumption time.
    with read_slot(paths[0], 1) as valid:
        assert not valid
    publisher._guard_dir.cleanup()


def test_reader_holds_pixels_until_read_finishes(tmp_path):
    path = tmp_path / "slot"
    path.write_bytes(bytes(8))
    commit_slot(path, 1)
    waiting, finished = threading.Event(), threading.Event()
    stop = threading.Event()
    original_wait = stop.wait

    def observed_wait(timeout):
        waiting.set()
        return original_wait(timeout)

    stop.wait = observed_wait

    def overwrite():
        with write_slot(path, stop):
            finished.set()

    with read_slot(path, 1) as valid:
        assert valid
        worker = threading.Thread(target=overwrite)
        worker.start()
        assert waiting.wait(2)
        assert not finished.is_set()
    worker.join(2)
    assert finished.is_set()
    with read_slot(path, 1) as valid:
        assert not valid


def test_stop_interrupts_writer_waiting_on_reader(tmp_path):
    path = tmp_path / "slot"
    path.write_bytes(bytes(8))
    commit_slot(path, 1)
    stop, cancelled = threading.Event(), threading.Event()

    def write():
        try:
            with write_slot(path, stop):
                raise AssertionError("must not overwrite a reader")
        except InterruptedError:
            cancelled.set()

    with read_slot(path, 1):
        worker = threading.Thread(target=write)
        worker.start()
        stop.set()
        worker.join(2)
        assert cancelled.is_set()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Slot marker contract, exercised across real processes."""

import multiprocessing
import os
from pathlib import Path
import stat

import pytest

from ipc_protocol import (
    LAYOUT_VERSION,
    MAX_SLOTS,
    MAX_STREAMS,
    REGION_SIZE,
    SlotMarkers,
)

SHM_DIR = Path("/dev/shm")


def _descriptor(markers):
    return (markers.name, markers.size, markers.layout_version, markers.epoch)


def _increment_worker(descriptor, stream, slot, count, results):
    markers = SlotMarkers.attach(*descriptor)
    try:
        results.put([markers.begin_write(stream, slot) for _ in range(count)])
    finally:
        markers.close()


def _begin_once(descriptor, stream, slot, out):
    markers = SlotMarkers.attach(*descriptor)
    out.put(markers.begin_write(stream, slot))
    markers.close()


def _load_once(descriptor, stream, slot, out):
    markers = SlotMarkers.attach(*descriptor)
    out.put(markers.load(stream, slot))
    markers.close()


def _attach_outcome(descriptor, out):
    try:
        SlotMarkers.attach(*descriptor).close()
        out.put("attached")
    except RuntimeError as error:
        out.put(f"refused: {error}")


@pytest.fixture
def ctx():
    return multiprocessing.get_context("spawn")


@pytest.fixture
def region():
    markers = SlotMarkers.create("/ipc_protocol_test", [3, 2])
    yield markers
    markers.close()


def test_child_increment_is_observed_by_parent_and_vice_versa(ctx, region):
    out = ctx.Queue()
    child = ctx.Process(target=_begin_once, args=(_descriptor(region), 0, 1, out))
    child.start()
    assert out.get(timeout=30) == 1
    child.join(timeout=30)
    assert region.load(0, 1) == 1

    assert region.begin_write(0, 1) == 2
    child = ctx.Process(target=_load_once, args=(_descriptor(region), 0, 1, out))
    child.start()
    assert out.get(timeout=30) == 2
    child.join(timeout=30)
    # Other slots and streams are untouched.
    assert region.load(0, 0) == 0
    assert region.load(1, 1) == 0


def test_concurrent_processes_get_unique_consecutive_generations(ctx, region):
    processes, per_process = 6, 2000
    results = ctx.Queue()
    workers = [
        ctx.Process(
            target=_increment_worker,
            args=(_descriptor(region), 1, 0, per_process, results),
        )
        for _ in range(processes)
    ]
    for worker in workers:
        worker.start()
    batches = [results.get(timeout=120) for _ in workers]
    for worker in workers:
        worker.join(timeout=30)
        assert worker.exitcode == 0
    returned = [value for batch in batches for value in batch]
    assert len(returned) == processes * per_process
    assert len(set(returned)) == len(returned)
    assert sorted(returned) == list(range(1, processes * per_process + 1))
    for batch in batches:
        assert batch == sorted(batch)
    assert region.load(1, 0) == processes * per_process


def test_attach_accepts_the_advertised_region(region):
    attached = SlotMarkers.attach(*_descriptor(region))
    try:
        assert attached.stream_count == 2
        assert [attached.slot_count(0), attached.slot_count(1)] == [3, 2]
        assert attached.epoch == region.epoch
        assert attached.size == REGION_SIZE
        assert attached.layout_version == LAYOUT_VERSION
    finally:
        attached.close()


@pytest.mark.parametrize(
    "field, mutate",
    [
        ("size", lambda d: (d[0], d[1] + 64, d[2], d[3])),
        ("smaller size", lambda d: (d[0], d[1] - 64, d[2], d[3])),
        ("layout_version", lambda d: (d[0], d[1], d[2] + 1, d[3])),
        ("epoch", lambda d: (d[0], d[1], d[2], d[3] ^ 1)),
        ("missing name", lambda d: (d[0] + "_absent", d[1], d[2], d[3])),
    ],
)
def test_attach_refuses_mismatched_descriptor(region, field, mutate):
    with pytest.raises(RuntimeError):
        SlotMarkers.attach(*mutate(_descriptor(region)))


def test_attach_refuses_wrong_size_object_and_unready_header():
    name = f"ipc_protocol_test_raw_{os.getpid()}"
    path = SHM_DIR / name
    try:
        # Right size but never initialised: the header was never marked ready.
        path.write_bytes(b"\0" * REGION_SIZE)
        with pytest.raises(RuntimeError, match="not the advertised region"):
            SlotMarkers.attach("/" + name, REGION_SIZE, LAYOUT_VERSION, 1)
        # Object whose size differs from the advertised one.
        path.write_bytes(b"\0" * (REGION_SIZE // 2))
        with pytest.raises(RuntimeError, match="wrong size"):
            SlotMarkers.attach("/" + name, REGION_SIZE, LAYOUT_VERSION, 1)
    finally:
        path.unlink(missing_ok=True)


def test_attach_in_a_fresh_process_reports_refusal(ctx, region):
    out = ctx.Queue()
    bad_epoch = _descriptor(region)[:3] + (region.epoch ^ 1,)
    child = ctx.Process(target=_attach_outcome, args=(bad_epoch, out))
    child.start()
    assert out.get(timeout=30).startswith("refused")
    child.join(timeout=30)


@pytest.mark.parametrize("stream, slot", [(2, 0), (0, 3), (1, 2), (99, 0), (0, 99)])
def test_out_of_range_indexes_raise(region, stream, slot):
    with pytest.raises(IndexError):
        region.begin_write(stream, slot)
    with pytest.raises(IndexError):
        region.load(stream, slot)
    with pytest.raises(IndexError):
        region.unchanged(stream, slot, 0)
    # A refused write advanced nothing.
    assert all(
        region.load(s, k) == 0 for s in range(2) for k in range(region.slot_count(s))
    )


def test_negative_indexes_are_refused(region):
    with pytest.raises((IndexError, TypeError, OverflowError)):
        region.begin_write(-1, 0)
    with pytest.raises((IndexError, TypeError, OverflowError)):
        region.begin_write(0, -1)


def test_slot_count_out_of_range_raises(region):
    with pytest.raises(IndexError):
        region.slot_count(2)


@pytest.mark.parametrize(
    "counts",
    [[], [0], [1, 0], [MAX_SLOTS + 1], [1] * (MAX_STREAMS + 1)],
)
def test_create_refuses_bad_stream_or_slot_counts(counts):
    with pytest.raises(ValueError):
        SlotMarkers.create("/ipc_protocol_test", counts)


def test_create_accepts_the_documented_limits():
    markers = SlotMarkers.create("/ipc_protocol_test", [MAX_SLOTS] * MAX_STREAMS)
    try:
        assert markers.begin_write(MAX_STREAMS - 1, MAX_SLOTS - 1) == 1
    finally:
        markers.close()


def test_region_object_is_private_and_named_uniquely():
    first = SlotMarkers.create("/ipc_protocol_test", [1])
    second = SlotMarkers.create("/ipc_protocol_test", [1])
    try:
        assert first.name != second.name
        assert first.epoch != second.epoch
        for markers in (first, second):
            assert markers.name.startswith("/ipc_protocol_test_")
            path = SHM_DIR / markers.name.lstrip("/")
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            assert path.stat().st_size == REGION_SIZE
    finally:
        first.close()
        second.close()


def test_creator_close_removes_the_name_and_is_idempotent(region):
    descriptor = _descriptor(region)
    reader = SlotMarkers.attach(*descriptor)
    path = SHM_DIR / region.name.lstrip("/")
    assert path.exists()
    region.begin_write(0, 0)
    region.close()
    region.close()
    assert not path.exists()
    with pytest.raises(RuntimeError):
        SlotMarkers.attach(*descriptor)
    # An already attached reader keeps its mapping and its last view.
    assert reader.load(0, 0) == 1
    reader.close()
    reader.close()


def test_closed_region_raises_on_use(region):
    reader = SlotMarkers.attach(*_descriptor(region))
    reader.close()
    for call in (
        lambda: reader.begin_write(0, 0),
        lambda: reader.load(0, 0),
        lambda: reader.unchanged(0, 0, 0),
        lambda: reader.epoch,
        lambda: reader.stream_count,
        lambda: reader.slot_count(0),
    ):
        with pytest.raises(RuntimeError, match="closed"):
            call()


def test_reader_close_does_not_remove_the_name(region):
    reader = SlotMarkers.attach(*_descriptor(region))
    reader.close()
    again = SlotMarkers.attach(*_descriptor(region))
    again.close()


def test_unchanged_tracks_every_write_attempt(region):
    generation = region.begin_write(1, 1)
    assert region.unchanged(1, 1, generation)
    region.begin_write(1, 1)
    assert not region.unchanged(1, 1, generation)


def test_generation_exhaustion_limit_is_not_exposed():
    pytest.skip(
        "kMaxGeneration (2**62) is not exposed by the module and cannot be reached by incrementing; not tested"
    )

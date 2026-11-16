# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The camera frame bridge: pairing, copy and verify, its records and its program.

No GPU and no CuPy: slots are numpy arrays or files, and the receiver is a fake
that speaks the camera receiver's notification protocol.
"""

from __future__ import annotations

import ast
import bisect
import importlib.util
import json
import os
import random
import re
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from camera_bench import SOF_NS, STREAMS, FakeReceiver, notification, open_file_slots
from repo_paths import repo_root

from isaaccapture_examples.agentic_remote.agent.camera import bridge_wire as wire
from isaaccapture_examples.agentic_remote.agent.camera import frame_bridge
from isaaccapture_examples.agentic_remote.agent.camera.frame_bridge import (
    PairTracker,
    ReceiverLink,
    SlotReader,
    StreamRefused,
    StreamRules,
    check_catalog,
)

CONFIGS = repo_root() / "examples" / "agentic_remote" / "configs"
CAMERA = Path(frame_bridge.__file__).resolve().parent
SMALL = StreamRules(width=8, height=4)  # 48 bytes per eye
PERIOD_NS = 16_666_667
MS = 1_000_000


class Rig:
    """A PairTracker on numpy slots and a clock the test moves."""

    def __init__(self, depth: int = 4, rules: StreamRules = SMALL) -> None:
        self.depth = depth
        self.rules = rules
        self.now = 1_000 * MS
        shape = (rules.height * 3 // 2, rules.width)
        self.slots = {
            (name, index): np.zeros(shape, np.uint8)
            for name in STREAMS
            for index in range(depth)
        }
        self.copy_ns = 1 * MS
        self.reads: list[tuple[str, int]] = []
        self.write_index = 0
        self.tracker = PairTracker(
            rules,
            depth,
            SlotReader(self.slots, self._to_host),
            monotonic_ns=lambda: self.now,
        )

    def _to_host(self, array: np.ndarray) -> np.ndarray:
        self.now += self.copy_ns // 2
        return array.copy()

    def notes(self, seq: int, slot: int, timestamp_ns: int | None = None, **meta):
        stamp = self.now if timestamp_ns is None else timestamp_ns
        fields = {"width": self.rules.width, "height": self.rules.height, **meta}
        return [notification(name, seq, slot, stamp, **fields) for name in STREAMS]

    def publish(self, seq: int, timestamp_ns: int | None = None, **meta) -> int:
        """Fill the next slot with `seq`, tell the tracker, move one period on."""
        slot = self.write_index
        for name in STREAMS:
            self.slots[(name, slot)][:] = seq % 251
        for note in self.notes(seq, slot, timestamp_ns, **meta):
            self.tracker.on_notification(note)
        self.write_index = (slot + 1) % self.depth
        self.now += PERIOD_NS
        return slot

    def ask(self, after=(None, None), timeout_s: float = 1.0) -> None:
        self.tracker.request(after[0], after[1], timeout_s)

    def count(self, name: str) -> int:
        return self.tracker.counters[name]


def taken_pair(rig: Rig):
    outcome = rig.tracker.take()
    assert outcome is not None and outcome.record["type"] == wire.PAIR, outcome
    return outcome


# 6.2: pairing


def test_pair_needs_both_eyes_same_session_seq_and_slot():
    rig = Rig()
    rig.ask()
    left, right = rig.notes(5, 0)
    rig.tracker.on_notification(left)
    assert rig.count("pairs") == 0 and rig.count("copies") == 0

    # The right eye of another seq: the lone left eye is dropped.
    rig.tracker.on_notification(rig.notes(6, 1)[1])
    assert rig.count("pairs") == 0 and rig.count("unpaired") == 1
    # The same seq in another slot.
    rig.tracker.on_notification(rig.notes(6, 2)[0])
    assert rig.count("pairs") == 0 and rig.count("eyes_differ") == 1
    # The same seq and slot from another session.
    rig.tracker.on_notification(rig.notes(7, 2)[0])
    rig.tracker.on_notification(rig.notes(7, 2, session=99)[1])
    assert rig.count("pairs") == 0 and rig.count("session_changes") == 1
    # The same seq and slot with another publish time.
    rig.tracker.on_notification(rig.notes(8, 3, session=99)[0])
    rig.tracker.on_notification(rig.notes(8, 3, rig.now + 1, session=99)[1])
    assert rig.count("pairs") == 0 and rig.count("eyes_differ") == 2
    assert rig.count("copies") == 0

    # Both eyes of one session, seq and slot, in either order.
    left, right = rig.notes(9, 0, session=99)
    rig.tracker.on_notification(right)
    rig.tracker.on_notification(left)
    assert rig.count("pairs") == 1 and rig.count("copies") == 1


def test_a_seq_that_starts_again_in_a_session_drops_the_waiting_eye():
    rig = Rig()
    rig.ask(timeout_s=100.0)
    # One eye of seq 1000 came, the other never did.
    rig.tracker.on_notification(rig.notes(1000, 0)[0])
    # The receiver's seq began again, in the same session.
    rig.publish(1)
    assert rig.count("unpaired") == 1 and rig.count("pairs") == 1
    assert rig.count("copies") == 1
    rig.publish(2)
    assert taken_pair(rig).record["seq"] == 1
    assert rig.tracker.refusal is None


def test_a_pair_that_is_not_newer_than_the_one_named_says_so():
    rig = Rig()
    rig.ask(after=(41, 1000), timeout_s=0.05)
    rig.publish(1)
    assert rig.count("pairs") == 1 and rig.count("copies") == 0
    rig.now += 50 * MS
    outcome = rig.tracker.take()
    assert outcome.record["type"] == wire.NONE
    assert "seq 1 is not after seq 1000" in outcome.record["reason"]


def test_missing_capture_time_is_refused():
    rig = Rig()
    rig.ask(timeout_s=0.1)
    rig.publish(1, sof=(None, SOF_NS))
    rig.publish(2, sof=(SOF_NS, None))
    rig.publish(3, left_sensor_sof_ns=0)
    assert rig.count("no_capture_time") == 3
    assert rig.count("pairs") == 0 and rig.count("copies") == 0
    rig.now += 100 * MS
    outcome = rig.tracker.take()
    assert outcome.record["type"] == wire.NONE
    assert "seq 3 has no capture time" in outcome.record["reason"]
    assert outcome.record["counters"]["no_capture_time"] == 3
    # There is no fallback time: the next pair with both times is the first one.
    rig.ask()
    rig.publish(4)
    rig.publish(5)
    assert taken_pair(rig).record["seq"] == 4


@pytest.mark.parametrize(
    ("meta", "counter", "text"),
    [
        ({"sof": (SOF_NS, SOF_NS + 5_000_001)}, "skew", "5000001 ns apart"),
        ({"sof": (SOF_NS + 6_000_000, SOF_NS)}, "skew", "-6000000 ns apart"),
        ({"tsc_delta_ns": None}, "skew", "None ns apart"),
        ({"clock": "tai"}, "clock_domain", "'tai' clock"),
        ({"clock": None}, "clock_domain", "None clock"),
        ({"receiver_rx_payload_done_mono_ns": None}, "no_arrival_stamp", "arrival"),
    ],
)
def test_a_pair_the_times_of_which_cannot_be_used_is_refused(meta, counter, text):
    rig = Rig()
    rig.ask(timeout_s=0.05)
    rig.publish(1, **meta)
    assert rig.count(counter) == 1 and rig.count("copies") == 0
    rig.now += 50 * MS
    outcome = rig.tracker.take()
    assert outcome.record["type"] == wire.NONE and text in outcome.record["reason"]


def test_a_skew_at_the_limit_passes():
    rig = Rig()
    rig.ask()
    rig.publish(1, sof=(SOF_NS, SOF_NS + 5_000_000))
    assert rig.count("copies") == 1


@pytest.mark.parametrize(
    ("meta", "text"),
    [
        ({"devices": (0, 1)}, "Argus devices 0 and 1; the calibration is for 5 (left)"),
        ({"devices": (7, 5)}, "Argus devices 7 and 5"),
        ({"rotation": 180}, "rotation 180; the calibration needs 0"),
        ({"width": 1280}, "is 1280 x 4, not 8 x 4"),
        ({"role": "mono"}, "carries the 'mono' eye"),
        ({"device_index": None}, "Argus devices None and None"),
        # Numbers that compare equal and are no whole numbers.
        ({"rotation": 0.0}, "rotation 0.0; the calibration needs 0"),
        ({"rotation": False}, "rotation False; the calibration needs 0"),
        ({"devices": (5.0, 7.0)}, "Argus devices 5.0 and 7.0"),
    ],
)
def test_a_stream_that_is_not_the_calibrated_one_ends_the_bridge(meta, text):
    rig = Rig()
    rig.ask()
    rig.publish(1, **meta)
    assert rig.count("copies") == 0
    outcome = rig.tracker.take()
    assert outcome.record["type"] == wire.ERROR and text in outcome.record["reason"]
    # It stays refused: a good pair later changes nothing.
    rig.ask()
    rig.publish(2)
    assert rig.tracker.take().record["type"] == wire.ERROR
    assert rig.count("copies") == 0


# 6.2: copy and verify


def test_nothing_is_copied_without_a_request():
    rig = Rig()
    for seq in range(1, 10):
        rig.publish(seq)
    assert rig.count("pairs") == 9 and rig.count("copies") == 0
    assert rig.tracker.take() is None
    # No backlog: a request gets the next pair to arrive, not one of those.
    rig.ask()
    assert rig.tracker.take() is None
    rig.publish(10)
    rig.publish(11)
    assert taken_pair(rig).record["seq"] == 10


def test_copy_is_confirmed_by_a_later_publish():
    rig = Rig(depth=4)
    rig.ask()
    start = rig.now
    slot = rig.publish(20)
    assert rig.count("copies") == 1
    # Copied at once, but not handed out before a later publish confirms it.
    assert rig.tracker.take() is None
    rig.publish(21)
    outcome = taken_pair(rig)
    record, (left, right) = outcome.record, outcome.payloads
    assert bytes(left) == bytes(right) == bytes([20]) * SMALL.eye_bytes
    assert record["session_id"] == 41 and record["seq"] == 20
    assert record["sof_wall_ns"] == [SOF_NS, SOF_NS + 1000]
    assert record["device_index"] == [5, 7] and record["rotation"] == 0
    assert record["receiver_rx_mono_ns"] == start - 4 * MS
    assert record["left_bytes"] == record["right_bytes"] == SMALL.eye_bytes
    check = record["check"]
    assert check["slot"] == slot and check["ring_depth"] == 4
    assert check["publish_mono_ns"] == start
    assert check["copy_start_mono_ns"] == start
    assert check["copy_end_mono_ns"] == start + rig.copy_ns
    assert check["confirmed_by_seq"] == 21
    assert check["confirmed_by_timestamp_ns"] > check["copy_end_mono_ns"]
    wire.check(record)
    assert rig.count("handed_out") == 1
    assert rig.tracker.take() is None


@pytest.mark.parametrize("gap", [1, 2, 3])
def test_any_publish_before_the_slot_is_reused_confirms(gap):
    # Ring depth 4: the slot of seq S is written again after the publish of S + 3.
    rig = Rig(depth=4)
    rig.ask()
    rig.publish(20)
    rig.publish(20 + gap)
    assert taken_pair(rig).record["check"]["confirmed_by_seq"] == 20 + gap


def test_overwritten_slot_is_discarded():
    rig = Rig(depth=4)
    rig.ask()
    # The copy of seq 20 took so long that seq 21 to 23 were published before it
    # ended: after seq 23 (20 + depth - 1) the receiver writes the slot of 20 again.
    rig.copy_ns = 60 * MS
    start = rig.now
    rig.publish(20)
    end = start + rig.copy_ns
    rig.copy_ns = 1 * MS
    rig.publish(21, start + 17 * MS)
    rig.publish(22, start + 34 * MS)
    assert rig.tracker.take() is None
    assert rig.count("overwritten") == 0
    rig.publish(23, start + 51 * MS)
    assert start + 51 * MS <= end
    assert rig.count("overwritten") == 1 and rig.count("handed_out") == 0
    assert rig.tracker.take() is None

    # The same request is served by the next whole pair, seq 23 itself, once a
    # later publish has confirmed that copy.
    assert rig.count("copies") == 2
    rig.publish(24)
    assert taken_pair(rig).record["seq"] == 23


def test_a_publish_during_the_copy_neither_confirms_nor_discards():
    rig = Rig(depth=4)
    rig.ask()
    rig.copy_ns = 30 * MS
    start = rig.now
    rig.publish(20)
    rig.publish(21, start + 17 * MS)  # before the copy ended: says nothing
    assert rig.tracker.take() is None and rig.count("overwritten") == 0
    rig.publish(22, start + 34 * MS)  # after it, and the slot was not yet reused
    assert taken_pair(rig).record["check"]["confirmed_by_seq"] == 22


def test_a_copy_nothing_confirmed_in_time_is_discarded():
    rig = Rig(depth=4)
    rig.ask(timeout_s=2.0)
    rig.publish(20)
    # Notifications were lost: the next one is past the last safe seq.
    rig.publish(24)
    assert rig.count("unconfirmed") == 1
    assert rig.tracker.take() is None
    # The request goes on with seq 24, and then nothing at all comes for 0.25 s.
    assert rig.count("copies") == 2
    rig.now += 251 * MS
    assert rig.tracker.take() is None
    assert rig.count("unconfirmed") == 2
    rig.publish(26)
    rig.publish(27)
    assert taken_pair(rig).record["seq"] == 26


def test_a_session_change_drops_the_waiting_eye_and_the_copy():
    rig = Rig(depth=4)
    rig.ask()
    rig.publish(20)
    assert rig.count("copies") == 1
    # The sender rebuilt its pipeline: seq starts again in a new session.
    rig.publish(1, session=42)
    assert rig.count("unconfirmed") == 1 and rig.count("session_changes") == 1
    assert rig.count("copies") == 2
    rig.publish(2, session=42)
    record = taken_pair(rig).record
    assert (record["session_id"], record["seq"]) == (42, 1)


def test_never_the_same_pair_twice():
    rig = Rig()
    rig.ask(after=(41, 30))
    rig.publish(29)
    rig.publish(30)
    assert rig.count("copies") == 0
    rig.publish(31)
    rig.publish(32)
    assert taken_pair(rig).record["seq"] == 31
    # A lower seq of another session is a new pair.
    rig.ask(after=(41, 31))
    rig.publish(3, session=50)
    rig.publish(4, session=50)
    assert taken_pair(rig).record["seq"] == 3


def test_a_request_ends_at_its_timeout_and_no_later():
    rig = Rig()
    rig.ask(timeout_s=0.5)
    rig.now += 499 * MS
    assert rig.tracker.take() is None
    # A copy that waits for its confirmation does not stretch the request.
    rig.publish(1)
    assert rig.count("copies") == 1
    outcome = rig.tracker.take()
    assert outcome.record["type"] == wire.NONE
    assert outcome.record["reason"] == "no verified stereo pair within 0.5 s"
    assert rig.tracker.take() is None
    # The dropped copy is not handed to the next request.
    rig.ask()
    rig.publish(2)
    rig.publish(3)
    assert taken_pair(rig).record["seq"] == 2


def test_notifications_that_are_not_usable_are_counted_and_ignored():
    rig = Rig()
    rig.ask()
    rig.tracker.on_notification({"type": "hello"})
    rig.tracker.on_notification(notification("head.left", 1, 0, rig.now) | {"seq": "1"})
    rig.tracker.on_notification({"type": "frame", "stream": "head.left"})
    other = notification("head.left", 1, 0, rig.now) | {"stream": "wrist1"}
    rig.tracker.on_notification(other)
    assert rig.count("malformed") == 2 and rig.count("notifications") == 0
    assert rig.tracker.refusal is None


def test_a_slot_that_cannot_be_copied_ends_the_bridge():
    rig = Rig()

    def broken(array):
        raise RuntimeError("no such buffer")

    rig.tracker._read = SlotReader(rig.slots, broken)
    rig.ask()
    rig.publish(1)
    outcome = rig.tracker.take()
    assert outcome.record["type"] == wire.ERROR
    assert "copying slot 0 failed" in outcome.record["reason"]


def test_every_pair_handed_out_is_whole_whatever_the_timing():
    """The proof of 6.2 against a receiver that overwrites slots without a lock."""
    rng = random.Random(1410)
    handed = torn = discarded = 0
    for depth in (3, 4, 6, 12):
        # The receiver's whole history first: per slot the write starts, and for
        # each write its publish time (None: the decode failed) and its seq.
        writes: list[list[int]] = [[] for _ in range(depth)]
        facts: list[list[tuple[int | None, int]]] = [[] for _ in range(depth)]
        notes: list[dict] = []
        seq, slot, clock = 0, 0, 1_000 * MS
        for _ in range(4000):
            seq += rng.choice((1, 1, 1, 2))
            write = clock + rng.randrange(1, 4) * MS
            failed = rng.random() < 0.05
            publish = None if failed else write + rng.randrange(2, 9) * MS
            writes[slot].append(write)
            facts[slot].append((publish, seq))
            if failed:
                clock = write + 9 * MS
                continue
            clock = publish + rng.randrange(1, 8) * MS
            for name in STREAMS:
                if rng.random() > 0.03:  # the receiver's queue drops some
                    notes.append(notification(name, seq, slot, publish, **small()))
            slot = (slot + 1) % depth

        rig = Rig(depth=depth)
        tracker = rig.tracker

        def read(stream, slot, rig=rig, writes=writes, facts=facts):
            # What the slot holds at the moment this eye is read.
            rig.now += rig.copy_ns // 2
            index = bisect.bisect_right(writes[slot], rig.now) - 1
            publish, seq = facts[slot][index] if index >= 0 else (None, 0)
            whole = publish is not None and rig.now >= publish
            return np.full(SMALL.eye_bytes, seq % 251 if whole else 255, np.uint8)

        tracker._read = read
        tracker.request(None, None, 1e6)
        seen = None
        for note in notes:
            # The bridge reads now and then late, and a copy can take long.
            late = rng.randrange(3) if rng.random() < 0.8 else rng.randrange(60)
            rig.now = max(rig.now, note["timestamp_ns"] + late * MS)
            rig.copy_ns = rng.choice((1, 1, 1, 2, 2, 5, 20, 60, 150)) * MS
            tracker.on_notification(note)
            held = tracker._held
            if held is not None and held is not seen:
                seen = held
                want = bytes([held.seq % 251]) * SMALL.eye_bytes
                torn += bytes(held.left) != want or bytes(held.right) != want
            outcome = tracker.take()
            if outcome is None:
                continue
            record, (left, right) = outcome.record, outcome.payloads
            assert record["type"] == wire.PAIR
            want = bytes([record["seq"] % 251]) * SMALL.eye_bytes
            assert bytes(left) == want and bytes(right) == want, record
            handed += 1
            tracker.request(record["session_id"], record["seq"], 1e6)
        discarded += tracker.counters["overwritten"] + tracker.counters["unconfirmed"]
    # The test means something only if copies were torn and were all caught.
    assert handed > 400 and torn > 50 and discarded >= torn, (handed, torn, discarded)
    print(f"handed out {handed}, torn copies {torn}, discarded {discarded}")


def small() -> dict:
    return {"width": SMALL.width, "height": SMALL.height}


# 6.1: the catalog


def catalog(**changes) -> dict:
    def entry(name: str) -> dict:
        fields = {
            "name": name,
            "width": 2560,
            "height": 1984,
            "pixel_format": "nv12",
            "gpu_id": 0,
            "slots": [
                {
                    "slot": index,
                    "handle": "",
                    "nbytes": 2560 * 1984 * 3 // 2,
                    "shape": [2976, 2560],
                    "strides": [2560, 1],
                }
                for index in range(12)
            ],
        }
        return fields | changes.get(name, {}) | changes.get("both", {})

    return {"type": "subscribed", "streams": [entry(name) for name in STREAMS]}


def test_the_catalog_of_this_rig_is_accepted():
    assert check_catalog(catalog(), StreamRules()) == (12, 0)


@pytest.mark.parametrize(
    ("changes", "text"),
    [
        ({"both": {"width": 1280, "height": 992}}, "is 1280 x 992 nv12; the calib"),
        ({"both": {"pixel_format": "rgba8"}}, "2560 x 1984 rgba8"),
        ({"head.right": {"name": "wrist1"}}, "has no stream head.right"),
        ({"both": {"gpu_id": 1}}, "slots are on GPU [1]"),
        ({"head.left": {"slots": catalog()["streams"][0]["slots"][:6]}}, "depth"),
        ({"both": {"slots": catalog()["streams"][0]["slots"][:2]}}, "fewer than 3"),
        (
            {"both": {"slots": [{"slot": 0, "shape": [1984, 2560, 4]}] * 3}},
            "not numbered",
        ),
        (
            {
                "both": {
                    "slots": [
                        {"slot": n, "shape": [1984, 2560, 4], "strides": [10240, 4, 1]}
                        for n in range(3)
                    ]
                }
            },
            "not a packed NV12 image",
        ),
        ({"both": {"slots": [1, 2, 3]}}, "slots that are no objects"),
        ({"both": {"slots": [None, {"slot": 1}, "x"]}}, "slots that are no objects"),
    ],
)
def test_a_catalog_that_does_not_fit_the_calibration_is_refused(changes, text):
    with pytest.raises(StreamRefused, match=re.escape(text)):
        check_catalog(catalog(**changes), StreamRules())


# The records


def test_records_survive_the_wire():
    records = [
        wire.hello(
            width=2560,
            height=1984,
            pixel_format="nv12",
            ring_depth=12,
            gpu_id=0,
            streams=STREAMS,
        ),
        wire.next_request(None, None, 0.5),
        wire.next_request(41, 7, 2),
        wire.none("nothing came", {"skew": 2}),
        wire.error("wrong devices"),
        wire.close(),
    ]
    for record in records:
        line = wire.pack(record)
        assert line.endswith(b"\n") and line.count(b"\n") == 1
        assert wire.parse(line[:-1]) == record


@pytest.mark.parametrize(
    "record",
    [
        [],
        {"type": "frame"},
        {"type": "next", "after_session": None, "after_seq": None},
        {"type": "next", "after_session": None, "after_seq": None, "timeout_s": -1},
        {"type": "next", "after_session": True, "after_seq": None, "timeout_s": 1},
        {"type": "next", "after_session": 1.5, "after_seq": None, "timeout_s": 1},
        {"type": "none"},
        {"type": "hello", "version": 1},
        {
            "type": "pair",
            "session_id": 1,
            "seq": 1,
            "sof_wall_ns": [1],
            "device_index": [5, 7],
            "rotation": 0,
            "receiver_rx_mono_ns": 1,
            "check": {},
            "left_bytes": 4,
            "right_bytes": 4,
        },
    ],
)
def test_a_bad_record_is_refused(record):
    with pytest.raises(wire.WireError):
        wire.check(record)


def test_a_line_that_is_no_json_is_refused():
    with pytest.raises(wire.WireError, match="not a JSON line"):
        wire.parse(b"\xff{")
    with pytest.raises(wire.WireError, match="too long"):
        wire.pack(wire.error("x" * wire.MAX_LINE_BYTES))


@pytest.mark.parametrize("kind", [[], {}, 1, None, True])
def test_a_record_type_that_is_no_text_is_a_wire_error(kind):
    with pytest.raises(wire.WireError, match="unknown record type"):
        wire.check({"type": kind})
    with pytest.raises(wire.WireError, match="unknown record type"):
        wire.parse(json.dumps({"type": kind}).encode())


def test_a_line_nested_too_deep_is_a_wire_error():
    with pytest.raises(wire.WireError, match="not a JSON line"):
        wire.parse(b"[" * 60000)


def test_a_line_over_the_limit_is_refused_also_when_it_comes_whole():
    ours, theirs = socket.socketpair()
    with ours, theirs:
        theirs.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 21)
        fill = "x" * (wire.MAX_LINE_BYTES - len('{"type":"none","reason":""}'))
        longest = json.dumps(
            {"type": "none", "reason": fill[:-1]}, separators=(",", ":")
        )
        assert len(longest) == wire.MAX_LINE_BYTES - 1  # what pack() lets through
        too_long = json.dumps({"type": "none", "reason": fill}, separators=(",", ":"))
        theirs.sendall(longest.encode() + b"\n" + too_long.encode() + b"\n")
        time.sleep(0.2)  # both lines are in one receive
        reader = wire.Reader(ours)
        deadline = time.monotonic() + 2.0
        assert len(reader.record(deadline)["reason"]) == len(fill) - 1
        with pytest.raises(wire.WireError, match="a line longer than"):
            reader.record(deadline)


def test_the_reader_keeps_records_and_raw_bytes_apart():
    ours, theirs = socket.socketpair()
    with ours, theirs:
        reader = wire.Reader(ours)
        pixels = bytes(range(256)) * 40
        stream = (
            wire.pack(wire.none("first"))
            + wire.pack(wire.error("then bytes"))
            + pixels
            + wire.pack(wire.close())
        )
        # In pieces that cut through lines and through the raw bytes.
        for start in range(0, len(stream), 997):
            theirs.sendall(stream[start : start + 997])
        deadline = time.monotonic() + 2.0
        assert reader.record(deadline)["reason"] == "first"
        assert reader.record(deadline)["reason"] == "then bytes"
        assert bytes(reader.exact(len(pixels), deadline)) == pixels
        assert reader.record(deadline) == wire.close()

        with pytest.raises(wire.WireTimeout):
            reader.record(time.monotonic() + 0.05)
        # Half a line, a timeout, the other half: nothing is lost.
        line = wire.pack(wire.none("in two halves"))
        theirs.sendall(line[:9])
        with pytest.raises(wire.WireTimeout):
            reader.record(time.monotonic() + 0.05)
        theirs.sendall(line[9:])
        assert reader.record(time.monotonic() + 2.0)["reason"] == "in two halves"

        theirs.sendall(b"x" * (wire.MAX_LINE_BYTES + 2))
        with pytest.raises(wire.WireError, match="a line longer than 65536 bytes"):
            reader.record(time.monotonic() + 2.0)
        theirs.close()
        rest = wire.Reader(ours)
        with pytest.raises(wire.WireClosed):
            while True:
                rest.exact(1000, time.monotonic() + 2.0)


def test_a_socket_in_a_deep_folder_goes_to_the_users_runtime_folder(
    tmp_path, monkeypatch
):
    assert wire.socket_path("/run/x", "b.sock") == "/run/x/b.sock"
    runtime = tmp_path / "rt"
    runtime.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    deep = tmp_path / ("d" * 120)
    path = wire.socket_path(str(deep), "b.sock")
    assert path == str(runtime / "agentic_remote" / "b.sock")
    assert stat.S_IMODE(os.stat(runtime / "agentic_remote").st_mode) == 0o700
    with pytest.raises(OSError, match="too long"):
        wire.socket_path(str(deep), "b" * 120)


def test_nobody_listening_is_an_error_not_a_hang(tmp_path):
    with pytest.raises(OSError):
        wire.dial(str(tmp_path / "nobody.sock"))


# The bridge program, on the fake receiver


class RunningBridge:
    """frame_bridge.main() on a thread, and the agent program's end of its socket."""

    def __init__(self, receiver: FakeReceiver, folder: Path, *extra: str) -> None:
        self.path = str(folder / "bridge.sock")
        self.code: int | None = None
        argv = [
            "--receiver-socket",
            receiver.socket_path,
            "--listen",
            self.path,
            "--width",
            str(receiver.width),
            "--height",
            str(receiver.height),
            "--accept-timeout-s",
            "5",
            *extra,
        ]
        self.thread = threading.Thread(target=self._run, args=(argv,), daemon=True)
        self.thread.start()

    def _run(self, argv: list[str]) -> None:
        self.code = frame_bridge.main(argv, open_slots=open_file_slots)

    def dial(self) -> tuple[socket.socket, wire.Reader]:
        deadline = time.monotonic() + 5.0
        while not os.path.exists(self.path):
            assert self.thread.is_alive() and time.monotonic() < deadline
            time.sleep(0.01)
        time.sleep(0.05)
        sock = wire.dial(self.path)
        return sock, wire.Reader(sock)

    def ended(self) -> int | None:
        self.thread.join(5.0)
        assert not self.thread.is_alive()
        return self.code


@pytest.fixture
def receiver(tmp_path):
    fake = FakeReceiver(tmp_path)
    yield fake
    fake.stop()


class Publisher:
    """Publishes a pair every few milliseconds, like the receiver at work."""

    def __init__(self, receiver: FakeReceiver, period_s: float = 0.005) -> None:
        self._receiver = receiver
        self._period_s = period_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._receiver.publish()
            time.sleep(self._period_s)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(2.0)


def test_the_bridge_program_hands_out_verified_pairs(receiver, tmp_path):
    bridge = RunningBridge(receiver, tmp_path)
    sock, reader = bridge.dial()
    with sock:
        assert stat.S_IMODE(os.stat(bridge.path).st_mode) == 0o600
        deadline = time.monotonic() + 5.0
        hello = reader.record(deadline)
        assert hello == wire.hello(
            width=64,
            height=48,
            pixel_format="nv12",
            ring_depth=4,
            gpu_id=0,
            streams=STREAMS,
        )
        assert receiver.subscriptions == [list(STREAMS)]

        # Nothing published: "none" after the timeout asked for, not later.
        asked = time.monotonic()
        wire.send(sock, wire.next_request(None, None, 0.2))
        answer = reader.record(deadline)
        assert answer["type"] == wire.NONE and 0.2 <= time.monotonic() - asked < 0.6

        publisher = Publisher(receiver)
        last = (None, None)
        try:
            for _ in range(5):
                wire.send(sock, wire.next_request(*last, 2.0))
                pair = reader.record(time.monotonic() + 5.0)
                assert pair["type"] == wire.PAIR, pair
                left = reader.exact(pair["left_bytes"], time.monotonic() + 2.0)
                right = reader.exact(pair["right_bytes"], time.monotonic() + 2.0)
                seq = pair["seq"]
                assert left == bytes([receiver.pixel(seq, 0)]) * (72 * 64)
                assert right == bytes([receiver.pixel(seq, 1)]) * (72 * 64)
                assert last[1] is None or seq > last[1]
                check = pair["check"]
                assert check["publish_mono_ns"] <= check["copy_start_mono_ns"]
                assert check["copy_start_mono_ns"] <= check["copy_end_mono_ns"]
                assert seq < check["confirmed_by_seq"] <= seq + 3
                assert check["confirmed_by_timestamp_ns"] > check["copy_end_mono_ns"]
                assert pair["receiver_rx_mono_ns"] <= time.monotonic_ns()
                last = (pair["session_id"], seq)
        finally:
            publisher.stop()
        wire.send(sock, wire.close())
    assert bridge.ended() == frame_bridge.EXIT_OK
    assert not os.path.exists(bridge.path)


def test_the_bridge_program_ends_when_the_agent_program_goes(receiver, tmp_path):
    bridge = RunningBridge(receiver, tmp_path)
    sock, reader = bridge.dial()
    reader.record(time.monotonic() + 5.0)
    sock.close()
    assert bridge.ended() == frame_bridge.EXIT_OK


def test_the_bridge_program_refuses_other_cameras_with_a_clear_line(tmp_path, capfd):
    receiver = FakeReceiver(tmp_path, devices=(0, 1), rotation=180)
    try:
        bridge = RunningBridge(receiver, tmp_path)
        sock, reader = bridge.dial()
        with sock:
            reader.record(time.monotonic() + 5.0)
            wire.send(sock, wire.next_request(None, None, 2.0))
            receiver.publish()
            answer = reader.record(time.monotonic() + 5.0)
            assert answer["type"] == wire.ERROR
            assert "rotation 180; the calibration needs 0" in answer["reason"]
        assert bridge.ended() == frame_bridge.EXIT_FAILED
        assert "camera frame bridge: head.left is sent with rotation 180" in (
            capfd.readouterr().err
        )
    finally:
        receiver.stop()


def test_the_bridge_program_says_when_the_receiver_goes(receiver, tmp_path):
    bridge = RunningBridge(receiver, tmp_path)
    sock, reader = bridge.dial()
    with sock:
        reader.record(time.monotonic() + 5.0)
        receiver.wait_client()
        receiver.stop()
        wire.send(sock, wire.next_request(None, None, 2.0))
        answer = reader.record(time.monotonic() + 5.0)
        assert answer == wire.error("the camera receiver closed its socket")
    assert bridge.ended() == frame_bridge.EXIT_FAILED


def test_the_bridge_program_does_not_start_on_the_wrong_stream(tmp_path, capfd):
    # No receiver at all.
    code = frame_bridge.main(
        ["--receiver-socket", str(tmp_path / "no.sock"), "--listen", "unused"],
        open_slots=open_file_slots,
    )
    assert code == frame_bridge.EXIT_FAILED
    assert "camera frame bridge: no camera receiver at" in capfd.readouterr().err

    # A receiver with images of another size than the calibration's.
    receiver = FakeReceiver(tmp_path)
    try:
        listen = tmp_path / "bridge.sock"
        code = frame_bridge.main(
            ["--receiver-socket", receiver.socket_path, "--listen", str(listen)],
            open_slots=open_file_slots,
        )
        assert code == frame_bridge.EXIT_FAILED and not listen.exists()
        assert (
            "stream head.left is 64 x 48 nv12; the calibration needs 2560 x 1984"
            in (capfd.readouterr().err)
        )
    finally:
        receiver.stop()

    # Without the pull request's folder the real slots cannot be opened.
    assert frame_bridge.main(["--receiver-socket", "a", "--listen", "b"]) == (
        frame_bridge.EXIT_NOT_RUN
        if not os.environ.get(frame_bridge.CAMERA_PR_ENV)
        else 1
    )
    assert frame_bridge.main(["--no-such-flag"]) == frame_bridge.EXIT_NOT_RUN
    capfd.readouterr()


def test_the_bridge_program_sets_the_gpu_itself(receiver, tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    seen = {}

    def open_slots(subscribed, camera_pr, gpu_id):
        seen.update(os.environ)
        raise RuntimeError("stop here")

    code = frame_bridge.main(
        [
            "--receiver-socket",
            receiver.socket_path,
            "--listen",
            str(tmp_path / "b.sock"),
            "--width",
            "64",
            "--height",
            "48",
        ],
        open_slots=open_slots,
    )
    assert code == frame_bridge.EXIT_NOT_RUN
    assert seen["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert seen["CUDA_VISIBLE_DEVICES"] == "0"


def test_the_bridge_needs_only_the_standard_library_to_start():
    """It runs in the camera consumer's environment: no numpy 2, no APAI, no yaml."""
    for name in ("bridge_wire.py", "frame_bridge.py"):
        tree = ast.parse((CAMERA / name).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    assert (node.level, node.module) == (1, None), ast.unparse(node)
                    assert [alias.name for alias in node.names] == ["bridge_wire"]
                    continue
                modules = [node.module.split(".")[0]]
            else:
                continue
            for module in modules:
                assert module in sys.stdlib_module_names, (name, module)


# Against the camera pull request's own files (skipped without its folder)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_the_receivers_own_publisher_speaks_to_the_bridge(camera_pr_dir, tmp_path):
    ipc = load(camera_pr_dir / "camera_receiver" / "cuda_ipc.py", "pr_cuda_ipc")
    publisher = ipc.CudaIpcFramePublisher(socket_path=str(tmp_path / "rx.sock"))
    # register_stream() needs CuPy; the catalog entries are put in by hand.
    for name in STREAMS:
        publisher._streams[name] = ipc._StreamInfo(
            name=name,
            width=2560,
            height=1984,
            pixel_format="nv12",
            dtype="uint8",
            gpu_id=0,
            layout={},
            slots=tuple(
                ipc._SlotInfo(index, "", 2560 * 1984 * 3 // 2, (2976, 2560), (2560, 1))
                for index in range(12)
            ),
        )
    publisher.start()
    link = ReceiverLink(publisher.socket_path, STREAMS)
    try:
        subscribed = link.attach()
        assert (
            link.hello["version"] == 2 and "slot_generation" in link.hello["features"]
        )
        assert check_catalog(subscribed, StreamRules()) == (12, 0)
        deadline = time.monotonic() + 5.0
        while not publisher._clients:
            assert time.monotonic() < deadline
            time.sleep(0.01)

        slots = {key: np.full((2976, 2560), 9, np.uint8) for key in [(STREAMS[0], 3)]}
        slots[(STREAMS[1], 3)] = np.full((2976, 2560), 10, np.uint8)
        tracker = PairTracker(
            StreamRules(), 12, SlotReader(slots, lambda array: array.copy())
        )
        tracker.request(None, None, 5.0)
        stamp = time.monotonic_ns()
        for seq, slot, at in ((70, 3, stamp), (71, 4, stamp + 50 * MS)):
            for name in STREAMS:
                note = notification(name, seq, slot, at)
                publisher.publish(name, slot, 1, seq, at, metadata=note["metadata"])
        got = []
        while len(got) < 4:
            assert time.monotonic() < deadline
            got += link.read_messages(0.2)
        assert [(m["stream"], m["seq"], m["slot"]) for m in got] == [
            ("head.left", 70, 3),
            ("head.right", 70, 3),
            ("head.left", 71, 4),
            ("head.right", 71, 4),
        ]
        for message in got[:2]:
            tracker.on_notification(message)
        assert tracker.counters["copies"] == 1
        if time.monotonic_ns() < stamp + 50 * MS:
            for message in got[2:]:
                tracker.on_notification(message)
            outcome = tracker.take()
            assert outcome.record["type"] == wire.PAIR
            assert outcome.record["check"]["confirmed_by_seq"] == 71
            assert bytes(outcome.payloads[1].reshape(-1)[:2]) == b"\n\n"
    finally:
        link.close()
        publisher.stop()
    assert link.read_messages(0.0) is None


def test_the_pull_request_still_has_what_the_bridge_uses(camera_pr_dir):
    consumer = load(camera_pr_dir / "cuda_ipc_consumer" / "consumer.py", "pr_consumer")
    client = consumer.CudaIpcClient("", [])
    assert callable(client._open_streams) and client.slots == {}
    assert {"array", "slot", "stream"} <= set(consumer.IpcSlot.__dataclass_fields__)
    # CuPy is imported inside functions only, so the module loads without it.
    receiver = (camera_pr_dir / "camera_receiver" / "camera_receiver.py").read_text()
    for key in (
        "session_id",
        "stream_rotation",
        "role",
        "device_index",
        "width",
        "height",
        "sender_wall_clock",
        "left_sensor_sof_ns",
        "right_sensor_sof_ns",
        "tsc_delta_ns",
        "receiver_rx_payload_done_mono_ns",
    ):
        assert f'"{key}"' in receiver, key
    # One thread per stream, and the write index moves only after a publish.
    assert "max_workers=1" in receiver
    assert "runtime.write_index = (slot + 1) % runtime.config.ring_depth" in receiver
    wire_format = (camera_pr_dir / "camera_receiver" / "avs_wire.py").read_text()
    assert '"tai" if flags & BUNDLE_SENDER_CLOCK_TAI else "realtime"' in wire_format


def test_the_bridge_starts_with_the_consumers_interpreter(camera_pr_dir):
    python = camera_pr_dir / "cuda_ipc_consumer" / ".venv" / "bin" / "python"
    if not python.exists():
        pytest.skip(f"{python} is not there")
    env = {
        "PYTHONPATH": str(repo_root() / "examples" / "agentic_remote" / "python"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": os.environ.get("PATH", ""),
    }
    done = subprocess.run(
        [str(python), "-m", frame_bridge.__name__, "--help"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "--receiver-socket" in done.stdout


# The two camera configuration files


def test_the_sender_configuration_is_this_rigs_calibrated_pair():
    sender = yaml.safe_load((CONFIGS / "camera_sender_rebot.yaml").read_text())
    rules = StreamRules()
    (head,) = sender["streams"]
    assert head["id"] == "head" and head["type"] == "stereo"
    assert tuple(camera["device"] for camera in head["cameras"]) == rules.devices
    output = head["output"]
    assert output["rotation"] == rules.rotation
    assert (output["width"], output["height"]) == (rules.width, rules.height)
    assert output["codec"] == "hevc" and output["resize_mode"] == "strict"


def test_the_receiver_configuration_matches_the_bridge():
    config = yaml.safe_load((CONFIGS / "camera_receiver_rebot.yaml").read_text())
    rules = StreamRules()
    assert config["receiver"]["gpu_id"] == 0
    (head,) = config["streams"]
    assert head["index"] == 0 and head["type"] == "stereo"
    assert (head["width"], head["height"]) == (rules.width, rules.height)
    assert head["output_format"] == rules.pixel_format
    assert head["ring_depth"] == 12
    assert (head["ipc"]["left"], head["ipc"]["right"]) == rules.streams
    sender = yaml.safe_load((CONFIGS / "camera_sender_rebot.yaml").read_text())
    assert sender["network"]["port"] == config["receiver"]["port"]


def test_the_configuration_names_the_devices_of_the_calibration(experiment_dir):
    manifest = json.loads(
        (experiment_dir / ".state/thor/calibration/manifest.json").read_text()
    )
    rules = StreamRules()
    order = manifest["camera_order"]
    assert (order["left"]["argus_device"], order["right"]["argus_device"]) == (
        rules.devices
    )
    assert manifest["raw_image_size_wh"] == [rules.width, rules.height]
    assert manifest["raw_rotation_degrees"] == rules.rotation

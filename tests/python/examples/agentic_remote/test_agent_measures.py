# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a run of the agent program measures of the split over two machines.

The agent program runs on the workstation and the motors are on the Thor. The
launcher's Measures takes every ROBOT_STATE of the agent's link, what the arm
adapter measured of a move and of a jaw command, and every camera frame the
camera returns, and writes them into the run folder on a thread of its own.

Three kinds of run. A scripted stream: ROBOT_STATEs with known single joint
reads, clocks and move progress on a link whose two clocks the test sets, so
every number in the files is known. The simulated arm: the real permit loop
against the real robot app (SimRig of test_arm_adapter.py). APAI's mock scene:
the whole launcher, as test_launcher.py runs it.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import test_arm_adapter
import test_launcher
from isaaccapture_examples.agentic_remote.agent import arm_adapter, launcher
from isaaccapture_examples.agentic_remote.agent.config import RunOptions
from isaaccapture_examples.agentic_remote.agent.launcher import Measures
from isaaccapture_examples.agentic_remote.agent.link_client import (
    STOP_SHOWN_WAIT_S,
    Arrival,
)
from isaaccapture_examples.agentic_remote.protocol.messages import (
    GripperState,
    JointSample,
    RobotState,
    Stop,
)
from link_bench import until
from test_agent_link import Rig as LinkRig
from test_agent_link import a_state
from test_agent_options import Evidence
from test_arm_adapter import LONG, SimRig
from test_launcher import LOOK, PICK, QUICK, Stage

from conftest import EXPERIMENT_DIR_ENV, import_experiment

apai = pytest.mark.apai
experiment = pytest.mark.experiment
# The fixtures of the two neighbours, for the tests of this file.
make_sim = test_arm_adapter.make_sim
sim = test_arm_adapter.sim
make_stage = test_launcher.make_stage
stage = test_launcher.stage

# The scripted stream. The Thor publishes a ROBOT_STATE every 50 ms of its
# monotonic clock, which starts at 1000 s where the workstation's reads 500 s.
THOR_0, HERE_0 = 1000.0, 500.0
PERIOD_MS = 50
# Every ROBOT_STATE takes 2 ms over the robot connection; every fifth 12 ms.
SHORT_TRANSIT, LONG_TRANSIT = 0.002, 0.012
# The Thor reads one of its two joints every 25 ms, and a ROBOT_STATE repeats
# the reads of its last 200 ms: each read arrives in four ROBOT_STATEs.
READ_EVERY_MS, WINDOW_MS = 25, 200
# How old the newest joint reading is when a ROBOT_STATE is built, in s.
READING_AGE = 0.003
# The move of the script: ROBOT_STATEs 10 to 22 name it.
MOVE_ID, MOVE_TOTAL = "agent-7", 20
MOVE_STATES = range(10, 23)
# The jaw command of the script runs while ROBOT_STATEs 24 to 27 arrive.
JAW_STATES = range(24, 28)
# What the arm adapter measured of that move, in s of the workstation's clock.
MOVE_FACTS: dict[str, Any] = {
    "move_id": MOVE_ID,
    "setpoints": MOVE_TOTAL,
    "permits": 9,
    "fresh_reads": 1,
    "fresh_wait_s": 0.05,
    "grid_wait_s": 0.25,
    "whole_check_s": 0.004,
    "first_result_after_s": 0.010,
    "armed_after_s": 0.030,
    "first_permit_after_s": 0.040,
    "first_sent_after_s": 0.100,
    "last_progress_after_s": 0.480,
    "all_sent_after_s": 0.500,
    "result_after_s": 0.620,
    "passes": 9,
    "pass_total_s": 0.018,
    "max_pass_s": 0.004,
    "status": "completed",
    "ended": "completed",
}


def thor_ms(k: int) -> int:
    """When ROBOT_STATE k of the script is built, in ms after THOR_0."""
    return PERIOD_MS * k


def read(i: int) -> JointSample:
    """Single joint read i of the script: it ends i * 25 ms after THOR_0."""
    took = 0.004 + 0.001 * (i % 3)  # 4, 5 or 6 ms
    end = THOR_0 + READ_EVERY_MS * i / 1000
    return JointSample(i % 2, 0.01 * i, end - took, end, True)


def reads_in(k: int, at_ms: int | None = None) -> list[JointSample]:
    """The reads ROBOT_STATE k repeats: those of its last 200 ms."""
    at_ms = thor_ms(k) if at_ms is None else at_ms
    newest = at_ms // READ_EVERY_MS
    oldest = (at_ms - WINDOW_MS) // READ_EVERY_MS + 1
    return [read(i) for i in range(oldest, newest + 1)]


def move_index(k: int) -> int:
    """Setpoints the Thor reports as sent in ROBOT_STATE k: 0, 0, 1, 3, 6, ..."""
    return 0 if k < 12 else min(MOVE_TOTAL, int(1 + 2.5 * (k - 12)))


def missed(k: int) -> int:
    """The Thor's running count of missed setpoint deadlines: 3, then 4, then 5."""
    return 3 + (k >= 15) + (k >= 19)


def scripted_state(k: int, seq: int, at_ms: int | None = None) -> RobotState:
    """ROBOT_STATE k of the script."""
    at_ms = thor_ms(k) if at_ms is None else at_ms
    recent = reads_in(k, at_ms)
    newest = {sample.joint: sample for sample in recent}
    health: dict[str, Any] = {
        "reading_age_s": READING_AGE,
        "missed_deadlines": missed(k),
        "permit": None,
    }
    fields: dict[str, Any] = {"motion": "hold"}
    if k in MOVE_STATES:
        sent = move_index(k)
        fields.update(move_id=MOVE_ID, move_total=MOVE_TOTAL, move_index=sent)
        fields.update(motion="moving", move_stage="sending" if k > 10 else "armed")
        if k > 10:  # the first PERMIT is held from ROBOT_STATE 11 on
            health["permit"] = {
                "move_id": MOVE_ID,
                "through_index": min(MOVE_TOTAL - 1, sent + 9),
                "evidence_age_s": 0.20 + 0.01 * (k - 11),
                "expires_in_s": 0.9,
            }
        fields.update(target_q=[0.001 * sent, 0.0], target_t=THOR_0 + at_ms / 1000)
    opening = 1.0 - 0.2 * (k - 24) if k in JAW_STATES else 1.0
    if k > JAW_STATES[-1]:
        opening = 0.4
    target = 0.4 if k >= JAW_STATES[0] else 1.0
    return dataclasses.replace(
        a_state(seq, THOR_0 + at_ms / 1000),
        joints=[newest[joint] for joint in sorted(newest)],
        recent=recent,
        gripper=GripperState(
            opening, THOR_0 + at_ms / 1000, True, True, target, False, "empty"
        ),
        health=health,
        **fields,
    )


class Stream:
    """The scripted stream on a link whose clocks the test sets, into a Measures."""

    def __init__(self, folder: Path, recording: Any = None) -> None:
        self.rig = LinkRig()
        self.measures = Measures(folder, recording)
        self.measures.attach(self.rig.link)
        self.folder = folder
        self.sent = 0  # ROBOT_STATEs delivered
        self.seq = 0
        self.delivered: dict[tuple[int, float], JointSample] = {}

    def arrive(self, k: int, *, skip_seq: int = 0, at_ms: int | None = None) -> float:
        """Deliver ROBOT_STATE k; returns its arrival on the workstation's clock."""
        self.seq += 1 + skip_seq
        state = scripted_state(k, self.seq, at_ms)
        transit = LONG_TRANSIT if k % 5 == 3 else SHORT_TRANSIT
        at_ms = thor_ms(k) if at_ms is None else at_ms
        here = HERE_0 + at_ms / 1000 + transit
        self.rig.arrive(state, at=here)
        self.sent += 1
        for sample in state.recent:
            self.delivered[(sample.joint, sample.t_end)] = sample
        return here

    def settled(self) -> None:
        """Wait until the measuring thread has counted every delivered state."""
        until(lambda: self.measures.states == self.sent, "the states to be counted")

    def run(self) -> None:
        """ROBOT_STATEs 0 to 30 with the move and the jaw command between them."""
        for k in range(31):
            if k == JAW_STATES[0]:
                self.settled()
                jaw = {"mode": "plain", "opening": 0.4, "effort": 0.5}
                self.measures.note(arm_adapter.MEASURED_JAW_BEGIN, jaw)
            # One ROBOT_STATE is lost on the way: 29 never arrives.
            self.arrive(k, skip_seq=int(k == 29))
            if k == MOVE_STATES[-1]:
                self.settled()
                self.measures.note(arm_adapter.MEASURED_MOVE, MOVE_FACTS)
            if k == JAW_STATES[-1]:
                self.settled()
                done = {**jaw, "sent_to_result_s": 0.150, "status": "completed"}
                done.update(outcome="reached", result_opening=0.4, waited_s=0.2)
                self.measures.note(arm_adapter.MEASURED_JAW, done)
        self.settled()

    def close(self) -> list[str]:
        try:
            return self.measures.close()
        finally:
            self.rig.close()

    def lines(self, name: str) -> list[dict[str, Any]]:
        return read_lines(self.folder / name)


def read_lines(path: Path) -> list[dict[str, Any]]:
    """The JSON lines of one of a run's files."""
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def stream(tmp_path: Path) -> Iterator[Stream]:
    made = Stream(tmp_path)
    yield made
    made.close()


def summary_row(lines: list[str], what: str) -> list[str]:
    """The numbers of the summary's line for one kind of duration, as text."""
    start = f"  {what}: "
    (line,) = (line for line in lines if line.startswith(start))
    return line[len(start) :].split(", ")


# The link: what an arrival listener gets


def test_an_arrival_listener_gets_each_state_with_its_clock_sample() -> None:
    rig = LinkRig()
    seen: list[tuple[int, Arrival, str]] = []
    order: list[str] = []

    def broken(state: RobotState, arrival: Arrival) -> None:
        raise ValueError("a listener's own fault")

    try:
        link = rig.link
        link.add_state_listener(lambda state: order.append(f"state {state.seq}"))
        link.add_arrival_listener(broken)
        link.add_arrival_listener(
            lambda state, arrival: (
                order.append(f"arrival {state.seq}"),
                seen.append((state.seq, arrival, threading.current_thread().name)),
            )
        )
        # The Thor's clock is 500 s ahead; the transits are 2, 7 and 3 ms.
        rig.arrive(a_state(1, mono=1000.00), at=500.002)
        rig.arrive(a_state(2, mono=1000.05), at=500.057)
        rig.arrive(a_state(3, mono=1000.10), at=500.103)
        until(lambda: len(seen) == 3, "the listener")
    finally:
        rig.close()
    assert [seq for seq, _, _ in seen] == [1, 2, 3]
    # On the link's receive thread, behind the state listeners; a listener
    # that raises costs the others nothing.
    assert {name for _, _, name in seen} == {"workstation-client"}
    assert order[:2] == ["state 1", "arrival 1"]
    first, second, third = (arrival for _, arrival, _ in seen)
    assert (first.at_s, second.at_s, third.at_s) == (500.002, 500.057, 500.103)
    # A clock sample is the arrival less the Thor's clock in the state; the
    # offset is the smallest kept, which is what thor_now() uses.
    assert second.sample_s == pytest.approx(500.057 - 1000.05)
    assert third.offset_s == pytest.approx(-499.998)
    assert link.thor_now()[0] == pytest.approx(500.103 - third.offset_s)
    assert [a.samples for a in (first, second, third)] == [1, 2, 3]
    assert first.spread_s == 0.0 and third.spread_s == pytest.approx(0.005)
    above = [a.transit_above_shortest_s for a in (first, second, third)]
    assert above == pytest.approx([0.0, 0.005, 0.001])
    assert all(a.stop_shown_s is None for a in (first, second, third))


def test_a_stop_is_timed_to_the_first_robot_state_that_shows_it() -> None:
    rig = LinkRig()
    shown: list[float | None] = []
    try:
        link = rig.link
        link.add_arrival_listener(lambda state, a: shown.append(a.stop_shown_s))
        rig.arrive(a_state(1, mono=1000.00, stop_number=4), at=500.00)
        link.stop("Ctrl+C", "agent_terminal")
        assert isinstance(rig.thor.next_message(), Stop)  # it left first
        # A ROBOT_STATE built before the STOP arrived does not show it.
        rig.arrive(a_state(2, mono=1000.05, stop_number=4), at=500.02)
        rig.arrive(a_state(3, mono=1000.10, stop_number=5), at=500.03)
        rig.arrive(a_state(4, mono=1000.15, stop_number=5), at=500.08)
        # A second STOP while the first is not shown is not timed in its place.
        rig.now.t = 500.10
        link.stop("again", "agent_terminal")
        rig.now.t = 500.15
        link.stop("and again", "agent_terminal")
        rig.arrive(a_state(5, mono=1000.20, stop_number=7), at=500.20)
        # One the robot app shows only after STOP_SHOWN_WAIT_S gives no time.
        link.stop("late", "agent_terminal")
        late = 500.20 + STOP_SHOWN_WAIT_S + 0.5
        rig.arrive(a_state(6, mono=1000.25 + STOP_SHOWN_WAIT_S, stop_number=8), at=late)
        until(lambda: len(shown) == 6, "the listener")
    finally:
        rig.close()
    assert shown[:2] == [None, None]
    assert shown[2] == pytest.approx(0.03) and shown[3] is None
    assert shown[4] == pytest.approx(0.10) and shown[5] is None


# The scripted stream: every number in the files is known


def test_the_single_joint_reads_of_a_known_stream_are_counted_once_each(
    stream: Stream,
) -> None:
    stream.run()
    lines = stream.close()
    reads = list(stream.delivered.values())
    took = sorted(sample.t_end - sample.t_start for sample in reads)
    # 31 ROBOT_STATEs repeated 8 reads each; 68 different reads were made.
    assert len(reads) == 68 and stream.sent == 31
    count, median, p95, longest = summary_row(
        lines, "single joint read, on the Thor's clock"
    )
    assert int(count) == 68
    assert (median, p95, longest) == ("5.00", "6.00", "6.00")
    # The same per second in link-latency.jsonl: nothing twice, nothing lost.
    seconds = stream.lines(launcher.LATENCY_FILE)
    assert len(seconds) >= 2
    assert sum(line["states"] for line in seconds) == 31
    assert sum(line["joint_reads"]["count"] for line in seconds) == 68
    total = sum(line["joint_reads"]["total_s"] for line in seconds)
    assert total == pytest.approx(sum(took), abs=1e-5)
    assert max(line["joint_reads"]["longest_s"] for line in seconds) == 0.006
    # And per move: the reads the ROBOT_STATEs of the move brought first,
    # which ended in the 650 ms from 19 * 25 ms to 44 * 25 ms.
    (move,) = stream.lines(launcher.MOVES_FILE)
    during = [read(i) for i in range(19, 45)]
    assert move["thor"]["joint_reads"] == {
        "count": 26,
        "total_s": pytest.approx(sum(s.t_end - s.t_start for s in during), abs=1e-6),
        "longest_s": 0.006,
    }


def test_a_known_move_gives_its_stretches_its_pace_and_its_missed_deadlines(
    stream: Stream,
) -> None:
    stream.run()
    lines = stream.close()
    (move,) = stream.lines(launcher.MOVES_FILE)
    # What the arm adapter measured, on the workstation's clock, as it was.
    for name, value in MOVE_FACTS.items():
        assert move[name] == value, name
    # The stretches between its points: from the first setpoint reported to
    # the last rise of the count, from there to the final RESULT, and the
    # settling, from every setpoint reported to the final RESULT.
    assert move["sending_s"] == pytest.approx(0.380)
    assert move["after_last_progress_s"] == pytest.approx(0.140)
    assert move["settle_s"] == pytest.approx(0.120)
    # What the ROBOT_STATEs that named the move say, on the Thor's clock. The
    # first setpoint was reported at 600 ms and the twentieth at 1000 ms.
    thor = move["thor"]
    assert thor["states"] == len(MOVE_STATES) == 13
    assert thor["setpoints_per_s"] == pytest.approx(19 / 0.4)
    assert thor["missed_deadlines"] == 2  # the count went from 3 to 5
    # The held permit: how far it ran ahead at the least, and how old the
    # camera frame behind it was, as the Thor counts it.
    assert thor["permit_lead_min"] == 0
    assert thor["frame_age_first_s"] == pytest.approx(0.20)
    assert thor["frame_age_max_s"] == pytest.approx(0.31)
    # In the summary: one move, and the camera frame's age of each of the 12
    # ROBOT_STATEs that showed its permit.
    assert summary_row(lines, "MOVE sent to its first RESULT") == [
        "1",
        "10.00",
        "10.00",
        "10.00",
    ]
    assert summary_row(lines, "MOVE sent to the first PERMIT")[1] == "40.00"
    assert summary_row(lines, "MOVE sent to the first setpoint reported")[1] == (
        "100.00"
    )
    settling = "move: every setpoint reported to final RESULT"
    assert summary_row(lines, settling)[1] == "120.00"
    sending = "move: first setpoint reported to the last rise of the count"
    assert summary_row(lines, sending)[1] == "380.00"
    assert summary_row(lines, "move: wait for a fresh obstacle grid")[3] == "250.00"
    assert summary_row(lines, "move: longest check ahead pass")[3] == "4.00"
    age = summary_row(lines, "camera frame age under a PERMIT, on the Thor")
    assert age == ["12", "250.00", "310.00", "310.00"]
    (pace,) = (line for line in lines if line.startswith("setpoint pace: "))
    assert pace == (
        "setpoint pace: 1 move(s), median 47.5 a second, lowest 47.5, highest 47.5"
    )
    (counts,) = (line for line in lines if line.startswith("ROBOT_STATEs "))
    assert counts.startswith("ROBOT_STATEs 31, not arrived 1; missed setpoint ")
    assert "deadlines 2; single joint reads not valid 0, possibly missed at 0 " in (
        counts
    )
    # The second in which the move ended names its first RESULT.
    seconds = stream.lines(launcher.LATENCY_FILE)
    assert [line["first_result_after_s"] for line in seconds].count([0.01]) == 1
    assert sum(line["seq_missing"] for line in seconds) == 1


def test_the_transit_and_the_clocks_of_a_known_stream(stream: Stream) -> None:
    stream.run()
    lines = stream.close()
    # Six of the 31 ROBOT_STATEs took 10 ms longer than the shortest.
    transit = summary_row(lines, "ROBOT_STATE transit above the shortest")
    assert transit == ["31", "0.00", "10.00", "10.00"]
    gap = summary_row(lines, "ROBOT_STATE arrival to the next")
    assert gap == ["30", "50.00", "60.00", "60.00"]
    seconds = stream.lines(launcher.LATENCY_FILE)
    last = seconds[-1]
    # The workstation's clock less the Thor's, with the shortest transit in it.
    assert last["clock_offset_s"] == pytest.approx(HERE_0 - THOR_0 + SHORT_TRANSIT)
    assert last["clock_spread_s"] == pytest.approx(LONG_TRANSIT - SHORT_TRANSIT)
    assert last["clock_samples"] == 31 and last["clock_steps"] == 0
    assert last["reading_age_s"] == READING_AGE and last["missed_deadlines"] == 5
    assert max(line["transit_above_shortest_s"]["longest"] for line in seconds) == (
        pytest.approx(0.010)
    )
    assert max(line["arrival_gap_max_s"] for line in seconds) == pytest.approx(0.06)
    # At most one line a second of the workstation's clock, and a last one
    # for what came since when the measuring ended.
    stamps = [line["mono_s"] for line in seconds[:-1]]
    assert len(seconds) == 2 and stamps[0] - (HERE_0 + SHORT_TRANSIT) >= 1.0 - 1e-6
    assert seconds[-1]["mono_s"] == pytest.approx(HERE_0 + 1.5 + SHORT_TRANSIT)
    (clock,) = (line for line in lines if line.startswith("clock: "))
    assert "less the Thor's is -499.998000 s with the shortest transit" in clock
    assert "the newest 31 ROBOT_STATEs spread over 10.00 ms" in clock


def test_robot_state_jsonl_holds_the_target_and_the_jaws_while_the_arm_works(
    stream: Stream,
) -> None:
    stream.run()
    stream.close()
    rows = stream.lines(launcher.STATES_FILE)
    by_seq = {row["seq"]: row for row in rows}
    assert len(by_seq) == len(rows)  # no ROBOT_STATE twice
    # Every ROBOT_STATE of the move, with the last target the Thor sent.
    moving = [row for row in rows if row["move_id"] == MOVE_ID]
    assert [row["seq"] for row in moving] == [k + 1 for k in MOVE_STATES]
    for row in moving:
        k = row["seq"] - 1
        assert row["move_index"] == move_index(k) and row["move_total"] == MOVE_TOTAL
        assert row["target_q"] == [0.001 * move_index(k), 0.0]
        assert row["target_t"] == row["clock_mono_s"] == THOR_0 + thor_ms(k) / 1000
        assert row["missed_deadlines"] == missed(k)
        # The measured joints: the newest read of each, 25 ms apart.
        assert row["q"] == pytest.approx([0.01 * (2 * k), 0.01 * (2 * k - 1)])
        assert row["sample_span_s"] == pytest.approx(0.025)
        assert row["arrived_mono_s"] == pytest.approx(
            HERE_0 + thor_ms(k) / 1000 + (LONG_TRANSIT if k % 5 == 3 else SHORT_TRANSIT)
        )
    assert moving[1]["permit"]["through_index"] == 9
    # Every ROBOT_STATE while the jaw command ran: the opening and its target.
    for k in JAW_STATES:
        jaw = by_seq[k + 1]["jaw"]
        assert jaw["opening"] == pytest.approx(1.0 - 0.2 * (k - 24))
        assert jaw["last_target"] == 0.4 and jaw["valid"] is True
    # While nothing runs, one line a second: fewer than the ROBOT_STATEs.
    idle = [row for row in rows if not row["move_id"] and row["seq"] - 1 < 10]
    assert [row["seq"] for row in idle] == [1]
    assert len(rows) == 1 + len(MOVE_STATES) + len(JAW_STATES)
    # The reads between two lines are counted in the later one: the last
    # line is of ROBOT_STATE 27, which brought the reads up to number 54.
    assert rows[0]["reads"]["count"] == 8
    assert sum(row["reads"]["count"] for row in rows) == 62


def test_a_jaw_command_gets_one_line_with_the_opening_against_its_target(
    stream: Stream,
) -> None:
    stream.run()
    lines = stream.close()
    (jaw,) = stream.lines(launcher.JAWS_FILE)
    assert (jaw["mode"], jaw["opening"], jaw["effort"]) == ("plain", 0.4, 0.5)
    assert jaw["sent_to_result_s"] == 0.15 and jaw["status"] == "completed"
    assert jaw["result_opening"] == 0.4
    # The four ROBOT_STATEs that came while it ran: the Thor's clock, the
    # measured opening, the last target sent.
    thor = jaw["thor"]
    assert thor["states"] == 4 and thor["load"] == "empty"
    assert thor["trace"] == [
        [THOR_0 + thor_ms(k) / 1000, pytest.approx(1.0 - 0.2 * (k - 24)), 0.4]
        for k in JAW_STATES
    ]
    assert thor["target_gap_max"] == pytest.approx(0.6)
    assert summary_row(lines, "jaw command sent to its RESULT") == [
        "1",
        "150.00",
        "150.00",
        "150.00",
    ]
    seconds = stream.lines(launcher.LATENCY_FILE)
    assert [line["jaw_result_s"] for line in seconds].count([0.15]) == 1


def test_lost_robot_states_and_reads_that_may_be_missed_are_counted(
    tmp_path: Path,
) -> None:
    evidence = Evidence()
    stream = Stream(tmp_path, evidence)
    try:
        for k in range(4):
            stream.arrive(k)
        # Two ROBOT_STATEs are lost and the next comes 500 ms later: its
        # window of 200 ms does not reach back to the one before.
        stream.arrive(4, skip_seq=2, at_ms=thor_ms(3) + 500)
        # A joint that cannot be read arrives as a reading that is not valid.
        bad = scripted_state(5, stream.seq + 1, thor_ms(3) + 710)
        failed = dataclasses.replace(bad.joints[0], valid=False, t_end=bad.clock_mono_s)
        bad = dataclasses.replace(bad, joints=[failed, bad.joints[1]])
        stream.rig.arrive(bad, at=HERE_0 + 0.862)
        stream.sent += 1
        stream.settled()
    finally:
        lines = stream.close()
    (counts,) = (line for line in lines if line.startswith("ROBOT_STATEs "))
    assert counts.startswith("ROBOT_STATEs 6, not arrived 2; ")
    assert "single joint reads not valid 1, possibly missed at 1 gap(s)" in counts
    seconds = stream.lines(launcher.LATENCY_FILE)
    assert sum(line["read_gaps"] for line in seconds) == 1
    assert sum(line["seq_missing"] for line in seconds) == 2
    # A ROBOT_STATE with a joint that has no valid reading gives no pose.
    poses = [data["seq"] for kind, data in evidence.events if kind == "robot.feedback"]
    assert poses == [1, 3, 7]
    states = [data["seq"] for kind, data in evidence.events if kind != "robot.feedback"]
    assert states == [1, 2, 3, 4, 7, 8]


def test_a_new_session_of_the_thor_starts_the_counting_of_its_reads_anew(
    stream: Stream,
) -> None:
    for k in range(3):
        stream.arrive(k)
    # A restarted robot app: another session, and its clock starts lower.
    again = dataclasses.replace(scripted_state(0, 1), session_id="s2")
    stream.rig.arrive(again, at=HERE_0 + 5.0)
    stream.sent += 1
    stream.settled()
    lines = stream.close()
    # The 8 reads of its first ROBOT_STATE are counted again; the 12 before
    # them once each. No arrival gap is taken across the restart.
    assert summary_row(lines, "single joint read, on the Thor's clock")[0] == "20"
    assert summary_row(lines, "ROBOT_STATE arrival to the next")[0] == "2"


def test_a_stop_of_this_link_is_in_the_summary_with_the_time_until_it_showed(
    stream: Stream,
) -> None:
    stream.arrive(0)
    stream.rig.now.t = HERE_0 + 0.010
    stream.rig.link.stop("Ctrl+C at the agent terminal", "agent_terminal")
    later = dataclasses.replace(scripted_state(1, 2), stop_number=1, stop_level="stop")
    stream.rig.arrive(later, at=HERE_0 + 0.052)
    stream.sent += 1
    stream.settled()
    lines = stream.close()
    row = summary_row(lines, "STOP sent to the ROBOT_STATE that shows it")
    assert row == ["1", "42.00", "42.00", "42.00"]
    (second,) = stream.lines(launcher.LATENCY_FILE)
    assert second["stop_shown_s"] == [pytest.approx(0.042)]


def test_the_summary_is_the_file_and_names_what_was_not_measured(
    tmp_path: Path,
) -> None:
    said: list[str] = []
    (tmp_path / "empty").mkdir()
    empty = Measures(tmp_path / "empty")
    # Nothing arrived: the files are there, the summary says so, nothing is said.
    lines = empty.close(said.append)
    assert said == [] and lines[2] == "  nothing was measured: no ROBOT_STATE arrived"
    for name in (*launcher.MEASURE_FILES, launcher.SUMMARY_FILE):
        assert (tmp_path / "empty" / name).is_file(), name
    made = Stream(tmp_path)
    try:
        made.run()
        lines = made.measures.close(said.append)
    finally:
        made.rig.close()
    assert said == lines and lines[0] == f"latency summary of {tmp_path}"
    assert lines[1] == "in ms: count, median, 95th percentile, longest"
    written = (tmp_path / launcher.SUMMARY_FILE).read_text()
    assert written == "\n".join(lines) + "\n"
    # After the close nothing more is taken.
    made.measures.note(arm_adapter.MEASURED_MOVE, MOVE_FACTS)
    assert len(read_lines(tmp_path / launcher.MOVES_FILE)) == 1


# The experiment's debug recorder: a steady feed of the measured joints


def test_the_debug_recorder_gets_ten_poses_a_second_and_each_read_once(
    tmp_path: Path,
) -> None:
    evidence = Evidence()
    made = Stream(tmp_path, evidence)
    arrived = {}
    try:
        for k in range(31):
            arrived[k] = made.arrive(k)
        made.settled()
    finally:
        made.close()
    assert [hook.rsplit(".", 2)[1:] for hook in evidence.hooks] == [
        ["Measures", "on_state"]
    ]
    poses = [data for kind, data in evidence.events if kind == "robot.feedback"]
    # The Thor published 20 ROBOT_STATEs a second: every second one is a pose.
    assert [pose["seq"] for pose in poses] == [k + 1 for k in range(0, 31, 2)]
    for pose in poses:
        k = pose["seq"] - 1
        # The measured joints, and how far apart they were read.
        assert pose["state"]["q"] == pytest.approx([0.01 * 2 * k, 0.01 * (2 * k - 1)])
        assert pose["sample_span_s"] == pytest.approx(0.025)
        # Its time: the arrival on this machine less the reading's age.
        assert pose["state"]["t"] == pytest.approx(arrived[k] - READING_AGE)
        assert pose["read_failures"] == 0
        assert pose["source"] == launcher.FEEDBACK_SOURCE
        assert pose["clock"] == launcher.FEEDBACK_CLOCK
    # Every ROBOT_STATE as one robot.state event, with each single joint read
    # in exactly one of them: joint, value, start and end on the Thor's clock.
    states = [data for kind, data in evidence.events if kind == launcher.STATE_EVENT]
    assert [data["seq"] for data in states] == list(range(1, 32))
    reads = [tuple(item) for data in states for item in data["reads"]]
    assert len(reads) == len(set(reads)) == 68
    assert set(reads) == {
        (s.joint, s.value, s.t_start, s.t_end) for s in made.delivered.values()
    }
    moving = states[12]
    assert moving["move_id"] == MOVE_ID and moving["move_index"] == 1
    assert moving["target_q"] == [0.001, 0.0] and moving["missed_deadlines"] == 3
    assert moving["arrived_mono_s"] == pytest.approx(arrived[12])
    assert states[25]["jaw"]["opening"] == pytest.approx(0.8)


# Camera deliveries


def test_each_camera_frame_gets_one_line_with_its_time_and_the_time_of_return(
    tmp_path: Path,
) -> None:
    measures = Measures(tmp_path)
    capture = {"backend": "link_stereo", "seq": 12, "clock_raw_age_s": 0.081}
    capture["check"] = {"publish_mono_ns": 7, "path": Path("/somewhere")}
    frames = [
        SimpleNamespace(t=time.monotonic() - 0.2, capture=capture),
        SimpleNamespace(t=time.monotonic() - 0.1, capture=None),
        SimpleNamespace(),  # no time and no capture facts: still a line
    ]
    before = time.monotonic()
    for frame in frames:
        measures.note_frame(frame)
    after = time.monotonic()
    capture["seq"] = 13  # what the camera does with its dict later is not written
    lines = measures.close()
    rows = read_lines(tmp_path / launcher.CAMERA_FILE)
    assert len(rows) == 3
    first, second, third = rows
    assert (first["capture"]["seq"], first["capture"]["backend"]) == (12, "link_stereo")
    assert first["capture"]["check"] == {"publish_mono_ns": 7, "path": "/somewhere"}
    assert first["frame_t"] == frames[0].t
    assert before - 1e-6 <= first["returned_mono_s"] <= after + 1e-6
    age = first["returned_mono_s"] - first["frame_t"]
    assert first["age_at_return_s"] == pytest.approx(age, abs=2e-6)
    assert 0.2 - 1e-6 <= age < 0.2 + (after - before) + 0.05
    assert second["capture"] == {} and second["age_at_return_s"] >= 0.1
    assert third["frame_t"] is None and third["age_at_return_s"] is None
    assert summary_row(lines, "camera frame: its time to its return")[0] == "2"


# Nothing of it runs on a thread that drives the arm


class Spy:
    """In place of launcher._open_append: who opens and writes each file.

    .threads are the names of the threads that opened or wrote; .written maps
    a file's name to its text so far. While .hold is set no write returns.
    """

    def __init__(self) -> None:
        self.threads: set[str] = set()
        self.written: dict[str, str] = {}
        self.hold = threading.Event()
        self.hold.set()
        self.waiting = threading.Event()
        self._lock = threading.Lock()

    def __call__(self, path: Path) -> Any:
        self.threads.add(threading.current_thread().name)
        spy = self

        class File:
            def write(self, text: str) -> None:
                spy.threads.add(threading.current_thread().name)
                spy.waiting.set()
                assert spy.hold.wait(30.0), "the test never let the write go on"
                with spy._lock:
                    spy.written[path.name] = spy.written.get(path.name, "") + text

            def close(self) -> None:
                spy.threads.add(threading.current_thread().name)

            def __enter__(self) -> File:
                return self

            def __exit__(self, *exc: Any) -> None:
                self.close()

        return File()

    def rows(self, name: str) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.written[name].splitlines()]


def measured(rig: SimRig, folder: Path, **options: Any) -> Measures:
    """A Measures on the rig's link that the rig's arm adapter hands its facts to."""
    measures = Measures(folder, **options)
    measures.attach(rig.link)
    rig.adapter._measure = measures.note
    return measures


@apai
def test_a_move_on_the_simulated_arm_is_measured_and_written_off_the_permit_thread(
    sim: SimRig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = Spy()
    monkeypatch.setattr(launcher, "_open_append", spy)
    measures = measured(sim, tmp_path)
    here = threading.current_thread().name  # this thread sends the PERMITs
    with sim.camera(0.1):
        assert sim.move(LONG, 1.2) is True
    stats = dict(sim.adapter.last_move)
    permits = sim.of("permit")
    sim.adapter.set_gripper(3.1, effort=0.4)  # a jaw command, to half open
    until(lambda: launcher.JAWS_FILE in spy.written, "the jaw command's line")
    lines = measures.close()
    # Every file was opened and written on the one measuring thread, and the
    # move's and the jaw command's lines among them.
    assert len(permits) > 5 and stats["permits"] == len(permits)
    assert spy.threads == {launcher.MEASURE_THREAD} and here != launcher.MEASURE_THREAD
    # No camera is in this run, so its file got no line.
    assert set(spy.written) == {
        *(set(launcher.MEASURE_FILES) - {launcher.CAMERA_FILE}),
        launcher.SUMMARY_FILE,
    }
    assert list(tmp_path.iterdir()) == []  # the spy took every write
    (move,) = spy.rows(launcher.MOVES_FILE)
    assert move["move_id"] == stats["move_id"] and move["ended"] == "completed"
    assert (move["status"], move["setpoints"], move["sent"]) == ("completed", 60, 60)
    # The stretches before the MOVE: fresh joint readings, a fresh obstacle
    # grid, the check of the whole move.
    assert move["fresh_reads"] == 1 and 0.0 < move["fresh_wait_s"] < 0.6
    assert 0.0 <= move["grid_wait_s"] < 0.5 and 0.0 < move["whole_check_s"] < 0.5
    # The points behind it, in their order, from just before the MOVE left.
    order = [
        move["first_result_after_s"],
        move["armed_after_s"],
        move["first_permit_after_s"],
        move["first_sent_after_s"],
        move["result_after_s"],
    ]
    assert order == sorted(order) and order[0] > 0.0
    assert move["armed_after_s"] == pytest.approx(stats["armed_after_s"], abs=1e-6)
    # 60 setpoints at 50 a second take 1.2 s; the RESULT follows the settling.
    assert 1.1 < move["result_after_s"] - move["first_sent_after_s"] < 2.5
    assert 1.0 < move["sending_s"] < 1.5
    assert move["sending_s"] + move["after_last_progress_s"] == pytest.approx(
        move["result_after_s"] - move["first_sent_after_s"], abs=1e-5
    )
    # The settling is known only when a ROBOT_STATE showed all 60 as sent.
    assert (move["settle_s"] is None) is ("all_sent_after_s" not in move)
    # The check ahead's passes in bulk.
    assert move["passes"] >= move["permits"] > 5
    assert move["max_pass_s"] <= move["pass_total_s"] < 1.0
    # What the Thor reported of the move: its pace on its own clock, and how
    # many setpoints were late.
    thor = move["thor"]
    assert thor["states"] >= 10
    assert thor["setpoints_per_s"] == pytest.approx(50.0, rel=0.15)
    assert thor["missed_deadlines"] >= 0
    assert thor["permit_lead_min"] >= 0
    assert 0.0 <= thor["frame_age_first_s"] <= thor["frame_age_max_s"] < 1.0
    rows = spy.rows(launcher.STATES_FILE)
    states = [row for row in rows if row["move_id"] == move["move_id"]]
    assert len(states) == thor["states"]
    assert states[-1]["target_q"][0] == pytest.approx(LONG[0], abs=0.01)
    assert states[-1]["q"][0] == pytest.approx(states[-1]["target_q"][0], abs=0.05)
    # The jaw command: sent to its RESULT, and the opening on the way.
    (jaw,) = spy.rows(launcher.JAWS_FILE)
    assert (jaw["mode"], jaw["status"]) == ("plain", "completed")
    assert jaw["opening"] == pytest.approx(0.5) and jaw["effort"] == 0.4
    assert 0.0 < jaw["sent_to_result_s"] <= jaw["waited_s"] < 12.0
    assert jaw["thor"]["states"] >= 1 and jaw["thor"]["trace"]
    assert jaw["thor"]["trace"][-1][2] == pytest.approx(0.5)  # the last target
    # The summary: one move, one jaw command, and the link's own numbers.
    assert summary_row(lines, "MOVE sent to its first RESULT")[0] == "1"
    assert summary_row(lines, "jaw command sent to its RESULT")[0] == "1"
    assert int(summary_row(lines, "ROBOT_STATE transit above the shortest")[0]) > 20
    assert spy.written[launcher.SUMMARY_FILE] == "\n".join(lines) + "\n"
    seconds = spy.rows(launcher.LATENCY_FILE)
    assert seconds and sum(line["states"] for line in seconds) == measures.states


@apai
def test_a_move_the_checker_refuses_before_it_is_sent_gets_a_line_too(
    sim: SimRig, tmp_path: Path
) -> None:
    measures = measured(sim, tmp_path)
    # An obstacle within the reach of the first permit: only those setpoints
    # are checked before a MOVE is sent.
    sim.obstacle_beyond(0.01)
    with pytest.raises(test_arm_adapter.types().SafetyViolation):
        sim.move(LONG, 1.2)
    # A call that is refused before anything is read measures nothing.
    with pytest.raises(test_arm_adapter.types().SafetyViolation):
        sim.adapter.stream_to(LONG, 1.2)
    measures.close()
    (move,) = read_lines(tmp_path / launcher.MOVES_FILE)
    # No MOVE left: no move id, the checker's refusal, and what the wait and
    # the check took before it.
    assert "move_id" not in move and move["ended"] == "CheckRefused"
    assert move["fresh_reads"] == 1 and move["whole_check_s"] > 0.0
    assert "first_result_after_s" not in move and move["thor"] is None
    assert sim.of("move") == []


@apai
def test_a_full_queue_does_not_stop_a_move_a_jaw_command_or_the_camera(
    sim: SimRig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = Spy()
    spy.hold.clear()  # the disk stalls: the first write does not return
    monkeypatch.setattr(launcher, "_open_append", spy)
    measures = measured(sim, tmp_path, queue_size=4)
    assert spy.waiting.wait(5.0), "no ROBOT_STATE was written"
    began = time.monotonic()
    with sim.camera(0.1):
        assert sim.move(LONG, 1.2) is True
    sim.adapter.set_gripper(3.1, effort=0.4)
    for _ in range(100):
        measures.note_frame(SimpleNamespace(t=time.monotonic(), capture={}))
    took = time.monotonic() - began
    # The move took its own time and was sent whole; what the queue could
    # not take was dropped and counted.
    assert took < 3.0 and sim.adapter.last_move["sent"] == 60
    assert measures.dropped > 100
    spy.hold.set()
    lines = measures.close()
    (counts,) = (line for line in lines if line.startswith("ROBOT_STATEs "))
    assert counts.endswith(f"measurements dropped on a full queue {measures.dropped}")
    assert measures.states <= 5  # the one being written and the queue's four


@apai
def test_files_that_cannot_be_written_are_logged_once_each_and_stop_nothing(
    sim: SimRig, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    for name in (launcher.MOVES_FILE, launcher.STATES_FILE):
        (tmp_path / name).mkdir()  # a folder where the file would be
    with caplog.at_level(logging.WARNING):
        measures = measured(sim, tmp_path)
        with sim.camera(0.1):
            assert sim.move(LONG, 1.2) is True
            assert sim.move([0.0] * 6, 1.2) is True
        lines = measures.close()
    warned = [r.getMessage() for r in caplog.records if "cannot be written" in r.msg]
    assert len(warned) == 2
    for name in (launcher.MOVES_FILE, launcher.STATES_FILE):
        (line,) = (text for text in warned if str(tmp_path / name) in text)
        assert "IsADirectoryError" in line
        assert line.endswith("the run goes on without it")
    assert lines[-1] == f"not written: {launcher.MOVES_FILE}, {launcher.STATES_FILE}"
    # The others were written, and the two moves are in the summary.
    assert read_lines(tmp_path / launcher.LATENCY_FILE)
    assert summary_row(lines, "MOVE sent to its first RESULT")[0] == "2"


@apai
def test_a_measurement_that_fails_in_the_adapter_is_logged_once_and_stops_nothing(
    sim: SimRig, caplog: pytest.LogCaptureFixture
) -> None:
    calls: list[str] = []

    def broken(kind: str, facts: dict[str, Any]) -> None:
        calls.append(kind)
        raise RuntimeError("the measurements are broken")

    sim.adapter._measure = broken
    with caplog.at_level(logging.WARNING):
        with sim.camera(0.1):
            assert sim.move(LONG, 1.2) is True
        sim.adapter.set_gripper(3.1, effort=0.4)
    assert calls == [
        arm_adapter.MEASURED_MOVE,
        arm_adapter.MEASURED_JAW_BEGIN,
        arm_adapter.MEASURED_JAW,
    ]
    warned = [r for r in caplog.records if "a measurement of the arm adapter" in r.msg]
    assert len(warned) == 1
    assert sim.adapter.last_jaw["status"] == "completed"


# The whole launcher on APAI's mock scene


@apai
def test_a_run_of_the_launcher_leaves_every_file_and_says_the_summary_at_its_end(
    stage: Stage,
) -> None:
    stage.typed = [QUICK, PICK]
    stage.answers = ["yes", "yes"]
    code = stage.run("--interactive")
    assert code in (launcher.EXIT_OK, launcher.EXIT_FAILED), stage.lines
    run = stage.run_dir
    for name in (*launcher.MEASURE_FILES, launcher.SUMMARY_FILE):
        assert (run / name).is_file(), name
    # Said at the start, before the robot connection: where it goes and what
    # the workstation cannot see.
    (told,) = stage.said("measurements: ")
    assert stage.lines.index(told) < stage.lines.index(
        stage.said("robot connection up")[0]
    )
    for name in (*launcher.MEASURE_FILES, launcher.SUMMARY_FILE):
        assert name in told
    assert "how long the Thor takes to send one setpoint to the motors" in told
    assert "the camera's capture and encoding apart from the network" in told
    # One line per MOVE and per jaw command the agent's link sent.
    moves = read_lines(run / launcher.MOVES_FILE)
    sent = [move for move in moves if move.get("move_id")]
    assert len(sent) == stage.motion.count("send_move") > 3
    jaws = read_lines(run / launcher.JAWS_FILE)
    assert len(jaws) == stage.motion.count("gripper") >= 2
    # The pace of the moves long enough for two ROBOT_STATEs with progress.
    paces = [
        move["thor"]["setpoints_per_s"]
        for move in sent
        if move["thor"] and move["thor"]["setpoints_per_s"] is not None
    ]
    assert paces and all(40.0 < pace < 60.0 for pace in paces)
    # The camera frames APAI's mock camera returned, each with its times.
    frames = read_lines(run / launcher.CAMERA_FILE)
    assert len(frames) > 5
    assert all(frame["capture"].get("invented_times") is True for frame in frames)
    assert all(0.0 <= frame["age_at_return_s"] < 1.0 for frame in frames)
    # The summary: the last thing the program says, and the file.
    start = stage.lines.index(f"latency summary of {run}")
    assert start > stage.lines.index(told)
    summary = stage.lines[start:]
    assert (run / launcher.SUMMARY_FILE).read_text() == "\n".join(summary) + "\n"
    assert int(summary_row(summary, "MOVE sent to its first RESULT")[0]) == len(sent)
    assert int(summary_row(summary, "jaw command sent to its RESULT")[0]) == len(jaws)
    assert int(summary_row(summary, "camera frame: its time to its return")[0]) == (
        len(frames)
    )
    reads = summary_row(summary, "single joint read, on the Thor's clock")
    assert int(reads[0]) > 10 and math.isfinite(float(reads[1]))
    assert stage.order[-1] == "close:nobody"


@apai
def test_a_task_that_only_looks_is_measured_without_a_move(stage: Stage) -> None:
    assert stage.run("--task", LOOK) in (launcher.EXIT_OK, launcher.EXIT_FAILED)
    run = stage.run_dir
    assert (run / launcher.MOVES_FILE).read_text() == ""
    assert (run / launcher.JAWS_FILE).read_text() == ""
    assert read_lines(run / launcher.CAMERA_FILE)
    seconds = read_lines(run / launcher.LATENCY_FILE)
    assert seconds and seconds[-1]["clock_samples"] >= 1
    # While nothing moves robot-state.jsonl gets about one line a second.
    states = read_lines(run / launcher.STATES_FILE)
    assert 1 <= len(states) < sum(line["states"] for line in seconds)
    assert states[0]["q"] == pytest.approx([0.0, 0.1, 0.1, 0.0, 0.0, 0.0])
    assert stage.said("latency summary of")


@apai
@experiment
def test_the_experiments_timing_shows_the_parts_of_each_move(
    stage: Stage, experiment_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The experiment's own timing (thor_timing.py) on a pick and place."""
    import_experiment("thor_timing")
    monkeypatch.setenv(EXPERIMENT_DIR_ENV, str(experiment_dir))
    timed = RunOptions(timing=True)
    monkeypatch.setattr(launcher, "_run_options", lambda args, settings: timed)
    code = stage.run("--task", PICK)
    assert code in (launcher.EXIT_OK, launcher.EXIT_FAILED), stage.lines
    run = stage.run_dir
    rows = read_lines(run / "timing.jsonl")
    spans = [row for row in rows if row["event"] == "end"]
    by_id = {span["id"]: span for span in spans}

    def inside(span: dict[str, Any]) -> str:
        """The name of the span this one lies in."""
        return by_id[span["parent"]]["name"] if span["parent"] in by_id else ""

    def named(name: str) -> list[dict[str, Any]]:
        return [span for span in spans if span["name"] == name]

    lines = read_lines(run / launcher.MOVES_FILE)
    moves = [move for move in lines if "move_id" in move]
    assert len(moves) > 3 and len(named("motor.stream_to")) >= len(moves)
    # The stretches before the MOVE and behind it, each inside its move.
    for name in ("safety.wait_fresh_grid", "safety.check_whole_move"):
        assert len(named(name)) >= len(moves), name
        assert {inside(span) for span in named(name)} == {"motor.stream_to"}, name
    assert len(named("motor.fresh_joint_readings")) >= len(moves)
    armed, first = named("motor.move_to_armed"), named("motor.move_to_first_result")
    assert len(armed) == len(first) == len(moves)
    assert {inside(span) for span in armed} == {"motor.stream_to"}
    assert {inside(span) for span in first} == {"motor.move_to_armed"}
    permits = named("motor.permits")
    assert len(permits) == len(moves)
    for span, move in zip(permits, moves, strict=True):
        assert inside(span) == "motor.stream_to"
        # The passes of the check ahead, in bulk, are those the move's line has.
        bulk = span["hot_calls"]["safety.check_ahead_pass"]
        assert bulk["count"] == move["passes"] >= 1 and bulk["errors"] == 0
        assert bulk["max_ms"] <= bulk["total_ms"] <= span["duration_ms"]
        # From the span's start to its end: the first PERMIT to the RESULT.
        whole_ms = 1e3 * (move["result_after_s"] - move["armed_after_s"])
        assert span["duration_ms"] == pytest.approx(whole_ms, abs=20.0)
    # The points of the permit loop: spans without length inside it.
    for point in arm_adapter.MARKS[:2]:
        marks = named(f"motor.mark:{point}")
        assert len(marks) == len(moves), point
        assert {inside(span) for span in marks} == {"motor.permits"}
        assert all(span["duration_ms"] < 5.0 for span in marks)
    # Taking the arm and giving it back: once each for the one task.
    (take,) = named("task.take_control")
    assert inside(take) == "tool:pick_and_place"
    assert len(named("task.give_back_control")) == 1
    status = json.loads((run / "timing-status.json").read_text())
    ours = [hook.rsplit(".", 1)[1] for hook in status["hooks"] if "agent." in hook]
    assert {"_permit_loop", "_mark", "step", "vet_whole", "_start", "_end"} <= set(ours)


def test_the_hand_overs_only_queue() -> None:
    """The three calls other threads make write nothing and call no link."""
    import ast
    import inspect
    import textwrap

    for name in ("on_state", "note", "note_frame", "_put"):
        source = textwrap.dedent(inspect.getsource(getattr(Measures, name)))
        called = {
            node.func.attr
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert called <= {"_put", "put_nowait", "monotonic", "_log_trouble"}, name

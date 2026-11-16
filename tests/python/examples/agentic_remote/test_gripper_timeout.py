# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A jaw command the agent program gave up on does not run later.

The arm adapter waits jaws.result_timeout_s for the final RESULT of a GRIPPER,
then gives up: it sends a STOP of the agent's own, and the agent is told the
command failed. A GRIPPER can also carry an expiry, a time of the Thor's
monotonic clock at or after which the robot app starts nothing of it.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from isaaccapture_examples.agentic_remote.agent import arm_adapter
from isaaccapture_examples.agentic_remote.agent.arm_adapter import RobotLinkError
from isaaccapture_examples.agentic_remote.agent.link_client import AgentLink, Pinned
from isaaccapture_examples.agentic_remote.protocol.carrier import loopback_pair
from isaaccapture_examples.agentic_remote.protocol.messages import (
    Gripper,
    Stop,
    decode,
)
from isaaccapture_examples.agentic_remote.robot_app import sim as sim_arm
from isaaccapture_examples.agentic_remote.robot_app import stop as stop_module
from link_bench import until
from test_arm_adapter import JAWS, ScriptedRig, ScriptedThor, SimRig, types

apai = pytest.mark.apai

WAIT_S = 5.0  # real seconds a test gives another thread before it fails
# A short jaws.result_timeout_s: the adapter gives up after this, in s.
GIVE_UP_S = 0.3
EXPIRED = re.compile(r"the jaw command expired \d+\.\d{3} s ago")


def position(opening: float) -> float:
    """A jaw opening, 0 closed to 1 open, in the units of the arm profile."""
    return JAWS["closed_pos"] + opening * (JAWS["open_pos"] - JAWS["closed_pos"])


class Late:
    """A robot connection whose arriving link messages a test keeps back.

    While kept back they wait on the link, in order, as on a slow network.
    What the robot app sends goes out as ever.
    """

    def __init__(self, carrier: Any) -> None:
        self._carrier = carrier
        self._open = threading.Event()
        self._open.set()
        # Taken off the carrier while kept back; read by the one receiver.
        self._kept: deque[tuple[int, bytes]] = deque()

    def keep_back(self) -> None:
        self._open.clear()

    def deliver(self) -> None:
        self._open.set()

    def receive(self, timeout_s: float) -> tuple[int, bytes] | None:
        if not self._open.wait(timeout_s):
            return None
        if self._kept:
            return self._kept.popleft()
        item = self._carrier.receive(timeout_s)
        if item is not None and not self._open.is_set():
            self._kept.append(item)  # it came in after keep_back()
            return None
        return item

    def __getattr__(self, name: str) -> Any:
        return getattr(self._carrier, name)


class LateRig(SimRig):
    """SimRig on a robot connection that can keep link messages back."""

    def __init__(self, harness: Any, **settings: Any) -> None:
        self.h = harness
        robot_end, self.agent_end = loopback_pair()
        self.robot_end = Late(robot_end)
        self._build(AgentLink(self.agent_end), **settings)
        assert harness.app.attach_carrier(self.robot_end)
        self.ready()


@pytest.fixture
def make_rig(make_realtime_robot_app: Callable[..., Any]) -> Iterator[Callable]:
    """The real robot app on the simulated arm with the agent as control owner."""
    made: list[SimRig] = []

    def make(kind: type[SimRig] = SimRig, **settings: Any) -> SimRig:
        rig = kind(make_realtime_robot_app(connected=False), **settings)
        made.append(rig)
        rig.own()
        return rig

    yield make
    for rig in made:
        rig.close()


@pytest.fixture
def scripted() -> Iterator[ScriptedRig]:
    """A scripted robot side that never answers a GRIPPER; the task is open."""
    rig = ScriptedRig(result_timeout_s=GIVE_UP_S)
    rig.own()
    rig.thor._take = rig.thor.others.append
    yield rig
    rig.close()


def command_ids(link: AgentLink) -> list[str]:
    """Every command id the link makes from now on, in order."""
    made: list[str] = []
    real = link.new_id

    def new_id() -> str:
        made.append(real())
        return made[-1]

    link.new_id = new_id
    return made


def statuses(h: Any, command_id: str) -> list[str]:
    return [result.status for result in h.client.results(command_id)]


def final(h: Any, command_id: str) -> Any:
    return h.client.wait_result(command_id, timeout_s=WAIT_S)


# The arm adapter gives up


@apai
def test_a_jaw_command_that_arrives_after_the_adapter_gave_up_moves_nothing(
    make_rig: Callable[..., SimRig],
) -> None:
    rig = make_rig(LateRig, result_timeout_s=GIVE_UP_S)
    h, link = rig.h, rig.link
    ids = command_ids(link)
    before = h.arm.gripper_read()
    assert before is not None and before > 0.9
    rig.robot_end.keep_back()
    with pytest.raises(RobotLinkError, match="no RESULT for the jaw command within"):
        rig.adapter.set_gripper(position(0.0))
    assert rig.adapter._stopped is True
    assert h.arm.jaw_commands == []  # the GRIPPER is still on its way
    rig.robot_end.deliver()
    until(lambda: link.final_result(ids[-1]) is not None, "the RESULT of the GRIPPER")
    assert link.final_result(ids[-1]).status == "refused"
    h.idle()
    assert h.app._worker.wait_idle(WAIT_S)
    # Nothing of the command reached the jaws.
    assert h.arm.jaw_commands == []
    assert h.arm.gripper_read() == before
    assert h.arm.jaw_stiffness == sim_arm.LIGHT
    # The adapter's STOP ended the agent's ownership; the motors stay on.
    _, text, data = h.wait_local_event("stop")
    assert data["sender"] == stop_module.AGENT_SENDER
    assert "gave up on a jaw command" in text
    until(lambda: link.state().owner == "nobody", "nobody as control owner")
    assert h.arm.torque_on() is True


@apai
def test_a_jaw_command_running_when_the_adapter_gives_up_ends_in_the_light_hold(
    make_rig: Callable[..., SimRig],
) -> None:
    rig = make_rig(result_timeout_s=GIVE_UP_S)
    h = rig.h
    h.arm.jaw_speed = 0.1  # the close would go on until the arm's 3 s timeout
    with pytest.raises(RobotLinkError, match="no RESULT for the jaw command within"):
        rig.adapter.set_gripper(position(0.0))
    h.wait_local_event("stop")
    until(lambda: h.arm.jaw_commands[-1].status == "cancelled", "the cancelled close")
    assert h.app._worker.wait_idle(WAIT_S)
    # Held lightly where the STOP found the jaws: begun, and cut short early.
    assert h.arm.jaw_stiffness == sim_arm.LIGHT
    assert h.arm.light_holds
    at = h.arm.gripper_read()
    assert 0.8 < at < 1.0
    time.sleep(0.2)
    assert h.arm.gripper_read() == at
    # Never opened: the close is the only command the jaws were given.
    assert [command.opening for command in h.arm.jaw_commands] == [0.0]
    assert h.arm.gripper_last_target() == 0.0
    assert h.arm.torque_on() is True


@apai
def test_giving_up_sends_the_agents_own_stop_also_inside_a_skill(
    scripted: ScriptedRig,
) -> None:
    adapter, thor = scripted.adapter, scripted.thor
    with (
        adapter.skill_scope(),  # here stop() itself sends nothing
        pytest.raises(RobotLinkError) as raised,
    ):
        adapter.close_gripper_two_stage()
    assert str(raised.value) == (
        f"no RESULT for the jaw command within {GIVE_UP_S:g} s; a STOP was sent"
    )

    def stops() -> list[Stop]:
        return [m for m in list(thor.others) if isinstance(m, Stop)]

    until(lambda: len(stops()) == 1, "the STOP")
    assert arm_adapter.AGENT_SENDER == stop_module.AGENT_SENDER
    assert stops()[0].sender == arm_adapter.AGENT_SENDER
    assert stops()[0].reason == (
        f"the agent gave up on a jaw command: no RESULT within {GIVE_UP_S:g} s"
    )
    # The task is over for the adapter: the next jaw command sends nothing.
    sent = [m for m in list(thor.others) if isinstance(m, Gripper)]
    with pytest.raises(types().SafetyViolation, match="arm stop latched"):
        adapter.set_gripper(position(1.0))
    time.sleep(0.05)
    assert [m for m in list(thor.others) if isinstance(m, Gripper)] == sent
    assert len(sent) == 1 and len(stops()) == 1


@apai
def test_a_jaw_command_answered_in_time_sends_no_stop_and_keeps_the_arm(
    make_rig: Callable[..., SimRig],
) -> None:
    rig = make_rig()
    h = rig.h
    assert rig.adapter.set_gripper(position(0.5), effort=0.4) is None
    assert rig.adapter.close_gripper_two_stage(width_frac_stage2=0.85) is not None
    assert [event[1:5] for event in rig.of("gripper")] == [
        (pytest.approx(0.5), 0.4, "plain", 12.0),
        (pytest.approx(0.15), 0.6, "close_until_contact", 12.0),
    ]
    assert [command.status for command in h.arm.jaw_commands] == ["reached"] * 2
    assert rig.of("stop") == [] and h.heard("stop") == []
    assert rig.adapter._stopped is False
    state = rig.link.state()
    assert (state.owner, state.stop_number) == ("agent", 0)


# The expiry of a GRIPPER on the robot app


def test_a_jaw_command_past_its_expiry_is_refused_at_once(robot_app: Any) -> None:
    h = robot_app
    h.new_task()
    command_id = h.client.gripper(0.0, expires_mono_s=h.clock.monotonic() - 0.5)
    result = final(h, command_id)
    assert statuses(h, command_id) == ["refused"]
    assert EXPIRED.fullmatch(result.reason), result.reason
    assert h.arm.jaw_commands == []


def test_a_jaw_command_that_expires_in_the_queue_never_starts(robot_app: Any) -> None:
    h = robot_app
    h.new_task()
    starting, start = h.hold_call("gripper_command")
    first = h.client.gripper(0.5)
    assert starting.wait(WAIT_S)
    waiting = h.client.gripper(0.0, expires_mono_s=h.clock.monotonic() + 1.0)
    last = h.client.gripper(0.8)
    assert h.client.wait_result(waiting, final=False).status == "accepted"
    assert h.client.wait_result(last, final=False).status == "accepted"
    h.clock.advance(2.0)
    start.set()
    result = final(h, waiting)
    assert statuses(h, waiting) == ["accepted", "refused"]
    assert EXPIRED.fullmatch(result.reason), result.reason
    assert result.detail == {"outcome": "refused", "opening": None}
    # The commands around it run; the close that expired was never given.
    assert final(h, first).status == "completed"
    assert final(h, last).status == "completed"
    assert [command.opening for command in h.arm.jaw_commands] == [0.5, 0.8]


def test_a_jaw_command_inside_its_expiry_runs_as_one_without(robot_app: Any) -> None:
    h = robot_app
    h.new_task()
    starting, start = h.hold_call("gripper_command")
    first = h.client.gripper(0.5, expires_mono_s=h.clock.monotonic() + 12.0)
    assert starting.wait(WAIT_S)
    waiting = h.client.gripper(0.2, expires_mono_s=h.clock.monotonic() + 12.0)
    assert h.client.wait_result(waiting, final=False).status == "accepted"
    start.set()
    for command_id, opening in ((first, 0.5), (waiting, 0.2)):
        result = final(h, command_id)
        assert statuses(h, command_id) == ["accepted", "completed"]
        assert (result.reason, result.end_q) == ("", None)
        assert result.detail == {
            "outcome": "reached",
            "opening": pytest.approx(opening),
        }
    assert [command.opening for command in h.arm.jaw_commands] == [0.5, 0.2]


def test_a_gripper_without_an_expiry_is_read_and_never_expires(robot_app: Any) -> None:
    h = robot_app
    h.new_task()
    state = h.client.state
    old = {
        "command_id": "old-1",
        "session_id": state.session_id,
        "owner_number": state.owner_number,
        "stop_number": state.stop_number,
        "opening": 0.5,
        "max_effort": 0.5,
        "mode": "plain",
    }
    payload = json.dumps(old).encode()
    assert decode(Gripper.TYPE, payload).expires_mono_s is None
    h.clock.advance(1000.0)
    h.workstation_end.send_reliable(Gripper.TYPE, payload)
    assert final(h, "old-1").status == "completed"
    assert statuses(h, "old-1") == ["accepted", "completed"]
    assert [command.opening for command in h.arm.jaw_commands] == [0.5]


# The expiry the agent program gives a GRIPPER


def test_the_agent_link_gives_a_gripper_the_expiry_of_its_own_wait() -> None:
    thor = ScriptedThor()  # its clocks are this machine's
    link = AgentLink(thor.agent_end)
    try:
        link.wait_ready(WAIT_S)
        link.pin(Pinned("s1", 1, 0))
        before = time.monotonic()
        link.gripper(0.5, 0.5, "plain", 7.0)
        after = time.monotonic()
        expires = thor.grippers[0].expires_mono_s
        assert expires is not None
        # Never later than the moment the wait for the RESULT ends.
        assert before + 7.0 - 0.25 <= expires <= after + 7.0
    finally:
        link.close()
        thor.close()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Inbox in isaaccapture_examples.agentic_remote.robot_app.inbox."""

from __future__ import annotations

from isaaccapture_examples.agentic_remote.robot_app.inbox import (
    INBOX_BYTES,
    INBOX_MESSAGES,
    Inbox,
)

MOVE, GRIPPER, PERMIT, VIEW_STATUS = 65, 66, 97, 98


def taken(inbox: Inbox) -> list[tuple[int, bytes]]:
    """Everything that waits, oldest first; the inbox is empty afterwards."""
    out = []
    while (waiting := inbox.take()) is not None:
        out.append(waiting)
    return out


def test_the_bounds_of_the_robot_app():
    inbox = Inbox()
    assert (inbox.max_messages, inbox.max_bytes) == (INBOX_MESSAGES, INBOX_BYTES)
    assert (INBOX_MESSAGES, INBOX_BYTES) == (65536, 8 * 1024 * 1024)


def test_an_empty_inbox_gives_nothing():
    inbox = Inbox()
    assert len(inbox) == 0
    assert inbox.take() is None


def test_link_messages_are_taken_in_the_order_they_were_added():
    inbox = Inbox()
    added = [(MOVE, b"1"), (GRIPPER, b"2"), (MOVE, b"3"), (PERMIT, b"4")]
    assert all(inbox.add(*one) for one in added)
    assert len(inbox) == 4
    assert taken(inbox) == added
    assert len(inbox) == 0


def test_a_newest_wins_link_message_replaces_the_waiting_one_of_its_type():
    inbox = Inbox()
    for one in [(PERMIT, b"old"), (MOVE, b"m"), (VIEW_STATUS, b"v"), (PERMIT, b"new")]:
        assert inbox.add(*one)
    # The newer permit takes the newer place: it never passes the MOVE.
    assert taken(inbox) == [(MOVE, b"m"), (VIEW_STATUS, b"v"), (PERMIT, b"new")]


def test_a_reliable_link_message_is_never_replaced():
    inbox = Inbox()
    assert inbox.add(GRIPPER, b"1") and inbox.add(GRIPPER, b"2")
    assert taken(inbox) == [(GRIPPER, b"1"), (GRIPPER, b"2")]


def test_a_newest_wins_link_message_that_was_taken_is_not_replaced():
    inbox = Inbox()
    assert inbox.add(PERMIT, b"1")
    assert inbox.take() == (PERMIT, b"1")
    assert inbox.add(PERMIT, b"2") and inbox.add(PERMIT, b"3")
    assert taken(inbox) == [(PERMIT, b"3")]


def test_no_more_link_messages_are_kept_than_the_bound():
    inbox = Inbox(max_messages=3)
    assert [inbox.add(GRIPPER, b"%d" % n) for n in range(5)] == [True] * 3 + [False] * 2
    assert len(inbox) == 3
    # Taking one makes room for one.
    assert inbox.take() == (GRIPPER, b"0")
    assert inbox.add(GRIPPER, b"5") is True
    assert inbox.add(GRIPPER, b"6") is False
    assert taken(inbox) == [(GRIPPER, b"1"), (GRIPPER, b"2"), (GRIPPER, b"5")]


def test_no_more_bytes_are_kept_than_the_bound():
    inbox = Inbox(max_bytes=10)
    assert inbox.add(MOVE, b"x" * 6) is True
    assert inbox.add(MOVE, b"x" * 5) is False  # 11 bytes in all
    assert inbox.add(MOVE, b"x" * 4) is True  # exactly the bound
    assert inbox.add(GRIPPER, b"") is True  # an empty payload needs no room
    assert inbox.add(GRIPPER, b"x") is False
    assert inbox.take() == (MOVE, b"x" * 6)
    assert inbox.add(MOVE, b"x" * 6) is True


def test_a_full_inbox_still_takes_a_newest_wins_link_message_that_replaces_one():
    inbox = Inbox(max_messages=2, max_bytes=10)
    assert inbox.add(PERMIT, b"x" * 5) and inbox.add(MOVE, b"x" * 5)
    assert inbox.add(VIEW_STATUS, b"") is False  # it would be a third
    assert inbox.add(PERMIT, b"y" * 5) is True  # in the place of the old permit
    assert inbox.add(PERMIT, b"y" * 6) is False  # one byte too many
    assert taken(inbox) == [(MOVE, b"x" * 5), (PERMIT, b"y" * 5)]


def test_a_link_message_that_was_refused_changes_nothing():
    inbox = Inbox(max_messages=1)
    assert inbox.add(PERMIT, b"kept")
    assert inbox.add(MOVE, b"over") is False
    assert inbox.add(PERMIT, b"x" * (INBOX_BYTES + 1)) is False
    assert taken(inbox) == [(PERMIT, b"kept")]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reading robot state from the Teleop Session Server, or a robot app in a local session.

robot = RobotState.connect()
what = robot.description()            # None until a robot app sent one
what.series["arm"].joint_names
arm = robot.joints("arm")             # None without a fresh reading
arm["joint2"].position, arm.age_s
robot.gripper("gripper").opening
robot.status()["fault"]
"""

from __future__ import annotations

import logging
import math
import os
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

from . import records
from ._local import LocalClient, NotThisVersion
from ._robot_state import RobotStateStore
from .description import GripperDescription, RobotDescription, SeriesDescription
from .records import DESCRIPTION_TOPIC, Fault

logger = logging.getLogger(__name__)

__all__ = [
    "GripperReading",
    "JointReading",
    "JointsReading",
    "Reading",
    "RecordReading",
    "RobotState",
    "StatusReading",
]

# Requests one refresh() may make: a kept connection the server closed, declarations that changed.
_MAX_REQUESTS = 4


@dataclass(frozen=True)
class Reading:
    """The newest record of one data series and how old it is.

    `sample_time_ns` is on the robot machine's clock; compare ages, not times. `age_s` is None
    when no clock estimate came with the record, and too large, never too small, while
    `clock_two_way` is False. `fresh` is False for a record older than its series allows and for
    one of a robot app that left.
    """

    series: str
    sequence: int
    sample_time_ns: int
    age_s: float | None
    receive_age_s: float
    fresh: bool
    clock_two_way: bool


@dataclass(frozen=True)
class JointReading:
    """One joint. `velocity` and `effort` are None when the series has none; `valid` is False
    for a joint the robot could not read, and its numbers are then nan."""

    name: str
    position: float
    velocity: float | None
    effort: float | None
    valid: bool


@dataclass(frozen=True)
class JointsReading(Reading, Mapping[str, JointReading]):
    """The joints of one record, by name, in the record's order."""

    joints: tuple[JointReading, ...]
    _by_name: dict[str, JointReading] = field(
        init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        by_name: dict[str, JointReading] = {}
        for joint in self.joints:
            by_name.setdefault(joint.name, joint)
        object.__setattr__(self, "_by_name", by_name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._by_name)

    @property
    def positions(self) -> tuple[float, ...]:
        return tuple(joint.position for joint in self._by_name.values())

    def __getitem__(self, name: str) -> JointReading:
        return self._by_name[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._by_name)

    def __len__(self) -> int:
        return len(self._by_name)


@dataclass(frozen=True)
class StatusReading(Reading, Mapping[str, float | bool | str | None]):
    """The readings of one status record, by name, and the faults the robot reports.

    A reading the robot could not read is None.
    """

    readings: Mapping[str, float | bool | str | None] = field(hash=False)
    faults: tuple[Fault, ...]

    def __getitem__(self, name: str) -> float | bool | str | None:
        return self.readings[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.readings)

    def __len__(self) -> int:
        return len(self.readings)


@dataclass(frozen=True)
class GripperReading(Reading):
    """A gripper's opening: 0 closed to 1 open, from the position of its first joint.

    `position` and `opening` are nan when `valid` is False.
    """

    name: str
    position: float
    valid: bool
    opening: float


@dataclass(frozen=True)
class RecordReading(Reading):
    """A record in a shape the robot defines. `values` is its JSON object, None for another encoding."""

    shape: str
    schema_name: str
    message_encoding: str
    data: bytes
    values: dict | None = field(hash=False)


def _opening(gripper: GripperDescription, joint: JointReading) -> float:
    span = gripper.open - gripper.closed
    if not joint.valid or span == 0.0 or not math.isfinite(joint.position):
        return math.nan
    return min(1.0, max(0.0, (joint.position - gripper.closed) / span))


def _joint_reading(joint: records.JointValue) -> JointReading:
    if joint.valid:
        return JointReading(*joint)
    # The record holds 0.0 there, a believable angle. `math.nan` itself, so that two readings
    # of one record compare equal.
    return JointReading(
        name=joint.name,
        position=math.nan,
        velocity=None if joint.velocity is None else math.nan,
        effort=None if joint.effort is None else math.nan,
        valid=False,
    )


def _names(names) -> str:
    return ", ".join(names) or "none"


class RobotState:
    """What the robot is and its newest readings, read on the machine of the program that
    serves them: the Teleop Session Server, or a robot app in a local session.

    Make one with `connect()` (a program of its own) or `from_store()` (the server's program).
    A read is None when the series has no fresh record or no robot app is connected;
    `allow_stale=True` gives the newest record whatever its age. Thread-safe.
    """

    def __init__(
        self,
        store: RobotStateStore,
        *,
        client: LocalClient | None = None,
        auto_refresh: bool = True,
        robot: str | None = None,
    ) -> None:
        self._store = store
        self._client = client
        self._auto_refresh = auto_refresh
        # The robot a `connect()` reader expects; None reads whichever is served.
        self._robot = robot
        self._other_robot_reported: str | None = None
        self._lock = threading.RLock()
        self._closed = False
        self._reachable: bool | None = None if client is not None else True
        self._known_setup_generation = 0
        self._other_version_reported = False
        self._description_key: tuple[int, int] | None = None
        self._description: RobotDescription | None = None
        self._undecodable: set[tuple[int, str]] = set()

    @classmethod
    def connect(
        cls,
        socket_path: str | os.PathLike | None = None,
        *,
        robot: str | None = None,
        timeout_s: float = 1.0,
        auto_refresh: bool = True,
    ) -> RobotState:
        """Read from the Teleop Session Server, or a robot app in a local session, on this machine.

        Never raises for a server that is not running: `server_reachable` is then False and
        every refresh tries again. `robot` is the robot this reader expects, by
        `RobotDescription.robot`: while another robot is served, nothing is read. `timeout_s` is
        how long one refresh waits for the server. With `auto_refresh=False` the readings change
        only when `refresh()` is called.
        """
        state = cls(
            RobotStateStore(),
            client=LocalClient(socket_path, timeout_s),
            auto_refresh=auto_refresh,
            robot=robot,
        )
        state.refresh()
        return state

    @classmethod
    def from_store(cls, store: RobotStateStore) -> RobotState:
        """Read a store in this program, e.g. the one the Teleop Session Server hosts."""
        return cls(store)

    def refresh(self) -> bool:
        """Fetch the server's newest robot state; False when the server did not answer.

        Always True for `from_store()`, whose store is read as it is.
        """
        with self._lock:
            if self._client is None:
                return True
            if self._closed:
                return False
            return self._fetch(self._client)

    @property
    def server_reachable(self) -> bool:
        """True when the Teleop Session Server, or a robot app in a local session, answered
        the last refresh.

        With a server, False until a robot app agreed robot state with it: the server opens the
        connection for readers only then.
        """
        with self._lock:
            self._update()
            return bool(self._reachable)

    @property
    def connected(self) -> bool:
        """True while a robot app that sends robot state is connected to the server or, in a
        local session, is in its session.

        A reader that expects a robot (`connect(robot=...)`) also needs that robot's description.
        """
        with self._lock:
            self._update()
            if not self._store.connected:
                return False
            return self._robot is None or self._describe() is not None

    def description(self) -> RobotDescription | None:
        """What the robot is; None until a robot app sent its description.

        It stays readable after the robot app left, until the next one connects.
        """
        with self._lock:
            self._update()
            return self._describe()

    def series(self) -> dict[str, SeriesDescription]:
        """The robot's data series by name, in the description's order; {} without a description."""
        description = self.description()
        return {} if description is None else dict(description.series)

    def joints(self, series: str, *, allow_stale: bool = False) -> JointsReading | None:
        """The newest record of a joints series.

        KeyError for a series the robot does not have, ValueError for one of another shape.
        """
        with self._lock:
            self._update()
            sessions = self._store.session_count
            description = self._describe()
            if description is None:
                return None
            return self._joints(description, series, allow_stale, sessions)

    def gripper(self, name: str, *, allow_stale: bool = False) -> GripperReading | None:
        """The newest reading of a gripper. KeyError for a gripper the robot does not have."""
        with self._lock:
            self._update()
            sessions = self._store.session_count
            description = self._describe()
            if description is None:
                return None
            gripper = description.grippers.get(name)
            if gripper is None:
                raise KeyError(
                    f"robot '{description.robot}' has no gripper '{name}' "
                    f"(it has: {_names(description.grippers)})"
                )
            joints = self._joints(description, gripper.series, allow_stale, sessions)
            if joints is None or not gripper.joints:
                return None
            joint = joints.get(gripper.joints[0])
            if joint is None:
                return None
            return GripperReading(
                series=joints.series,
                sequence=joints.sequence,
                sample_time_ns=joints.sample_time_ns,
                age_s=joints.age_s,
                receive_age_s=joints.receive_age_s,
                fresh=joints.fresh,
                clock_two_way=joints.clock_two_way,
                name=name,
                position=joint.position,
                valid=joint.valid,
                opening=_opening(gripper, joint),
            )

    def status(
        self, series: str = "status", *, allow_stale: bool = False
    ) -> StatusReading | None:
        """The newest record of a status series.

        KeyError for a series the robot does not have, ValueError for one of another shape.
        """
        with self._lock:
            self._update()
            sessions = self._store.session_count
            description = self._describe()
            if description is None:
                return None
            self._entry(description, series, "status")
            reading = self._read(series, allow_stale, sessions)
            if reading is not None and not isinstance(reading, StatusReading):
                raise ValueError(f"series '{series}' does not carry status records")
            return reading

    def record(self, series: str, *, allow_stale: bool = False) -> Reading | None:
        """The newest record of any series: joints and status as their own readings, every
        other shape as a `RecordReading`. KeyError for a series the robot does not have."""
        with self._lock:
            self._update()
            sessions = self._store.session_count
            description = self._describe()
            if description is None:
                return None
            self._entry(description, series, None)
            return self._read(series, allow_stale, sessions)

    def close(self) -> None:
        """Leave the server. Reads of `connect()` give None afterwards; a store given to
        `from_store()` is not touched."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._client is not None:
                self._client.close()
                self._forget()
                self._reachable = False

    def __enter__(self) -> RobotState:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _update(self) -> None:
        if self._auto_refresh and self._client is not None and not self._closed:
            self._fetch(self._client)

    def _fetch(self, client: LocalClient) -> bool:
        retried = False
        for _ in range(_MAX_REQUESTS):
            new_connection = not client.is_open
            try:
                if new_connection:
                    # Nothing read from one server is shown as another's.
                    self._forget()
                    client.open()
                snapshot = client.request(self._known_setup_generation)
            except NotThisVersion as error:
                return self._lost(str(error), other_version=True)
            except OSError as error:
                # The server may have closed a connection kept since an earlier refresh: one
                # new connection is tried, but never after the wait for an answer ran out.
                if new_connection or retried or isinstance(error, TimeoutError):
                    return self._lost(str(error))
                retried = True
                continue
            sessions = self._store.session_count
            try:
                self._store.import_snapshot(snapshot)
            except RuntimeError as error:
                if self._known_setup_generation == 0:
                    return self._lost(str(error), other_version=True)
                # The declarations changed since they were fetched: ask for them.
                self._known_setup_generation = 0
                continue
            if self._store.session_count < sessions:
                self._forget()
                continue
            self._known_setup_generation = self._store.setup_generation
            self._other_version_reported = False
            if self._reachable is not True:
                self._reachable = True
                logger.info("Reading robot state at %s", client.path)
            return True
        return self._lost("the server's robot state kept changing")

    def _lost(self, reason: str, *, other_version: bool = False) -> bool:
        """No usable answer: nothing of an earlier answer stays readable."""
        client = self._client
        client.close()
        self._forget()
        if other_version and not self._other_version_reported:
            self._other_version_reported = True
            logger.warning(
                "The program that serves robot state at %s runs another version of the "
                "isaaccapture package than this program; no robot state is read (%s)",
                client.path,
                reason,
            )
        if self._reachable is not False:
            self._reachable = False
            logger.info("Nothing serves robot state at %s (%s)", client.path, reason)
        return False

    def _forget(self) -> None:
        self._store = RobotStateStore()
        self._known_setup_generation = 0
        self._description_key = None
        self._description = None
        self._undecodable.clear()

    def _describe(self) -> RobotDescription | None:
        """The served description; None without one, and while it names another robot than
        the one this reader expects."""
        description = self._decode_description()
        if description is None or self._robot in (None, description.robot):
            return description
        return None

    def _decode_description(self) -> RobotDescription | None:
        store = self._store
        # The session number is read first: a description that belongs to a newer session is
        # then decoded again at the next call instead of being kept under the newer number.
        sessions = store.session_count
        record = store.latest(DESCRIPTION_TOPIC)
        if record.data is None:
            return None
        key = (sessions, record.sequence)
        if key != self._description_key:
            try:
                description = records.decode_description(record.data)
            except ValueError as error:
                description = None
                logger.warning("The robot's description cannot be read: %s", error)
            self._description_key = key
            self._description = description
            if description is not None and self._client is not None:
                self._report_robot(self._client.path, description.robot)
        return self._description

    def _report_robot(self, path: str, robot: str) -> None:
        """Say which robot is read: once per description, and once per robot that is refused."""
        logger.info("Robot state at %s: robot '%s'", path, robot)
        if self._robot in (None, robot):
            self._other_robot_reported = None
        elif robot != self._other_robot_reported:
            self._other_robot_reported = robot
            logger.warning(
                "Robot state at %s is of robot '%s', not '%s'; nothing is read",
                path,
                robot,
                self._robot,
            )

    def _entry(
        self, description: RobotDescription, series: str, shape: str | None
    ) -> SeriesDescription:
        entry = description.series.get(series)
        if entry is None:
            raise KeyError(
                f"robot '{description.robot}' has no series '{series}' "
                f"(it has: {_names(description.series)})"
            )
        if shape is not None and entry.shape != shape:
            raise ValueError(
                f"series '{series}' has shape '{entry.shape_name}', not '{shape}'"
            )
        return entry

    def _joints(
        self,
        description: RobotDescription,
        series: str,
        allow_stale: bool,
        sessions: int,
    ) -> JointsReading | None:
        self._entry(description, series, "joints")
        reading = self._read(series, allow_stale, sessions)
        if reading is not None and not isinstance(reading, JointsReading):
            raise ValueError(f"series '{series}' does not carry joints records")
        return reading

    def _read(self, series: str, allow_stale: bool, sessions: int) -> Reading | None:
        """The series' newest record, decoded by the shape and encoding it was declared with.

        `sessions` is the store's session number from before the caller read the description.
        """
        store = self._store
        record = store.latest(series)
        data = record.data
        if data is None:
            return None
        fresh = record.fresh and store.connected
        if not fresh and not allow_stale:
            return None
        info = store.series_info(series)
        if info is None:
            return None
        # Another thread may fill the store. A robot app that connected since `sessions` was
        # read emptied it: the description, the record, `connected` and the declaration may
        # then belong to two robot apps.
        if store.session_count != sessions:
            return None
        common = {
            "series": series,
            "sequence": record.sequence,
            "sample_time_ns": record.sample_time_ns,
            "age_s": None if record.age_ns is None else record.age_ns / 1e9,
            "receive_age_s": record.receive_age_ns / 1e9,
            "fresh": fresh,
            "clock_two_way": record.clock_two_way,
        }
        flatbuffer = info.message_encoding == "flatbuffer"
        try:
            if flatbuffer and info.shape == "joints":
                joints = tuple(
                    _joint_reading(joint) for joint in records.decode_joints(data)
                )
                return JointsReading(joints=joints, **common)
            if flatbuffer and info.shape == "status":
                readings, faults = records.decode_status(data)
                return StatusReading(readings=readings, faults=tuple(faults), **common)
        except ValueError as error:
            self._report_undecodable(sessions, series, error)
            return None
        values = None
        if info.message_encoding == "json":
            try:
                values = records.decode_custom(data)
            except ValueError as error:
                self._report_undecodable(sessions, series, error)
        return RecordReading(
            shape=info.shape,
            schema_name=info.schema_name,
            message_encoding=info.message_encoding,
            data=data,
            values=values,
            **common,
        )

    def _report_undecodable(self, sessions: int, series: str, error: Exception) -> None:
        """One warning per series and robot session."""
        if (sessions, series) not in self._undecodable:
            self._undecodable.add((sessions, series))
            logger.warning(
                "A record of the robot's series '%s' cannot be read: %s", series, error
            )

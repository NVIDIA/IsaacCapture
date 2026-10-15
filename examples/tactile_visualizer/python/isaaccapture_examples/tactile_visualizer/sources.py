# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Latest-only mock, live, and MCAP inputs for the tactile visualizer."""

from __future__ import annotations

import logging
import math
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .model import (
    LatestMailbox,
    ShapeDocument,
    default_openxr_shape,
    matrix_to_quaternion_xyzw,
)

logger = logging.getLogger("isaaccapture.examples.tactile_visualizer.sources")

RAW_JOINT_NAMES = (
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
    "index_MCP_AA",
    "index_MCP_FE",
    "index_PIP",
    "index_DIP",
    "middle_MCP_AA",
    "middle_MCP_FE",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_AA",
    "ring_MCP_FE",
    "ring_PIP",
    "ring_DIP",
    "pinky_MCP_AA",
    "pinky_MCP_FE",
    "pinky_DIP1",
    "pinky_DIP2",
    "pinky_DIP3",
)
RAW_COLLECTIONS = {
    "left": "avatar_raw_left",
    "right": "avatar_raw_right",
}


@dataclass(frozen=True)
class HandSample:
    positions: np.ndarray
    orientations: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class SourceSnapshot:
    received_at: float
    hands: dict[str, HandSample] = field(default_factory=dict)
    raw_joints: dict[str, dict[str, float]] = field(default_factory=dict)
    shapes: dict[str, ShapeDocument] = field(default_factory=dict)
    forces: dict[str, np.ndarray] = field(default_factory=dict)


class SnapshotSource:
    def __init__(self) -> None:
        self.mailbox: LatestMailbox[SourceSnapshot] = LatestMailbox()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("source is already started")
        self._thread = threading.Thread(
            target=self._thread_main,
            name=f"{type(self).__name__}-producer",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.error("%s did not stop within five seconds", type(self).__name__)

    def _thread_main(self) -> None:
        try:
            self.run()
        except BaseException as error:
            logger.exception("%s stopped", type(self).__name__)
            self.mailbox.fail(error)

    def run(self) -> None:
        raise NotImplementedError


def _mock_hand(shape: ShapeDocument, curl: float) -> HandSample:
    positions = shape.joint_bind_positions.copy()
    orientations = shape.joint_bind_orientations.copy()
    for start, length in ((2, 4), (6, 5), (11, 5), (16, 5), (21, 5)):
        base = positions[start].copy()
        previous = base
        cumulative = 0.0
        for joint in range(start + 1, start + length):
            segment = float(
                np.linalg.norm(
                    shape.joint_bind_positions[joint]
                    - shape.joint_bind_positions[joint - 1]
                )
            )
            cumulative += curl * (0.35 if start == 2 else 0.48)
            direction = np.array([0.0, math.cos(cumulative), -math.sin(cumulative)])
            positions[joint] = previous + segment * direction
            c, s = math.cos(cumulative), math.sin(cumulative)
            orientations[joint] = matrix_to_quaternion_xyzw(
                np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])
            )
            previous = positions[joint]
    return HandSample(
        positions=positions,
        orientations=orientations,
        valid=np.ones(shape.num_joints, dtype=bool),
    )


class MockSource(SnapshotSource):
    """Alternates between open palm and fist while publishing tactile forces."""

    def __init__(self, num_taxels: int = 200, rate_hz: float = 30.0) -> None:
        super().__init__()
        if rate_hz <= 0:
            raise ValueError("mock rate must be positive")
        self.shape = default_openxr_shape(num_taxels)
        self.rate_hz = rate_hz

    def run(self) -> None:
        start = time.monotonic()
        frame = 0
        while not self._stop.is_set():
            now = time.monotonic()
            phase = 0.5 - 0.5 * math.cos(2 * math.pi * (now - start) / 4.0)
            hand = _mock_hand(self.shape, phase)
            raw = {
                name: phase * (1.1 if name.endswith(("FE", "IP", "PIP")) else 0.35)
                for name in RAW_JOINT_NAMES
            }
            force = np.zeros((self.shape.num_taxels, 3), dtype=np.float32)
            taxel_phase = np.arange(self.shape.num_taxels, dtype=np.float32) * 0.07
            force[:, 2] = np.maximum(
                0.0,
                phase * (1.4 + 0.6 * np.sin(2 * math.pi * (now - start) + taxel_phase)),
            )
            shapes = {"left": self.shape, "right": self.shape}
            self.mailbox.put(
                SourceSnapshot(
                    received_at=now,
                    hands={"left": hand, "right": hand},
                    raw_joints={"left": raw, "right": raw},
                    shapes=shapes,
                    forces={"left": force, "right": force},
                )
            )
            frame += 1
            self._stop.wait(max(0.0, start + frame / self.rate_hz - time.monotonic()))


@dataclass(frozen=True)
class SessionSourceConfig:
    tactile_collections: dict[str, str]
    tactile_channels: dict[str, str]
    launch_plugin: bool = False
    plugin_search_paths: tuple[Path, ...] = ()
    sdk_config: Path | None = None
    datasets: str = "human,raw,robot,haptic"
    replay_path: Path | None = None
    replay_rate_hz: float = 60.0


class SessionSource(SnapshotSource):
    """Uses the same TeleopSession sources and collection names as Sharpa Avatar."""

    def __init__(self, config: SessionSourceConfig, launcher_args: Any) -> None:
        super().__init__()
        self.config = config
        self.launcher_args = launcher_args
        self._known_shapes: dict[str, ShapeDocument] = {}

    def _plugins(self):
        if not self.config.launch_plugin:
            return []
        from isaaccapture.teleop_session_manager import PluginConfig

        plugin_args = [f"--datasets={self.config.datasets}"]
        if self.config.sdk_config is not None:
            plugin_args.insert(0, str(self.config.sdk_config))
        return [
            PluginConfig(
                plugin_name="avatar_hand_plugin",
                plugin_root_id="sharpa_avatar",
                search_paths=list(self.config.plugin_search_paths),
                plugin_args=plugin_args,
            )
        ]

    def _duration_s(self) -> float | None:
        if self.config.replay_path is None:
            return None
        from mcap.reader import make_reader

        with self.config.replay_path.open("rb") as stream:
            summary = make_reader(stream).get_summary()
        if summary is None or summary.statistics is None:
            return None
        stats = summary.statistics
        return max(0.0, (stats.message_end_time - stats.message_start_time) / 1e9)

    @staticmethod
    def _hand(group, index) -> HandSample | None:
        if group is None or group.is_none:
            return None
        return HandSample(
            positions=np.asarray(group[index.JOINT_POSITIONS], dtype=np.float32).copy(),
            orientations=np.asarray(
                group[index.JOINT_ORIENTATIONS], dtype=np.float32
            ).copy(),
            valid=np.asarray(group[index.JOINT_VALID], dtype=bool).copy(),
        )

    @staticmethod
    def _raw(group) -> dict[str, float] | None:
        if group is None or group.is_none:
            return None
        return {name: float(group[i]) for i, name in enumerate(RAW_JOINT_NAMES)}

    def run(self) -> None:
        from isaaccapture.cloudxr import CloudXRLauncher
        from isaaccapture.deviceio_session import McapReplayConfig
        from isaaccapture.deviceio_trackers import TactileTracker
        from isaaccapture.retargeting_engine.deviceio_source_nodes import (
            HandsSource,
            JointStateSource,
        )
        from isaaccapture.retargeting_engine.interface import OutputCombiner
        from isaaccapture.retargeting_engine.tensor_types import HandInputIndex
        from isaaccapture.teleop_session_manager import (
            SessionMode,
            TeleopSession,
            TeleopSessionConfig,
        )

        hands = HandsSource(name="hands")
        raw_sources = {
            side: JointStateSource(
                name=RAW_COLLECTIONS[side],
                collection_id=RAW_COLLECTIONS[side],
                joint_names=list(RAW_JOINT_NAMES),
            )
            for side in ("left", "right")
        }
        pipeline = OutputCombiner(
            {
                "hand_left": hands.output(HandsSource.LEFT),
                "hand_right": hands.output(HandsSource.RIGHT),
                "raw_left": raw_sources["left"].output(JointStateSource.JOINTS),
                "raw_right": raw_sources["right"].output(JointStateSource.JOINTS),
            }
        )
        tactile = {
            side: TactileTracker(self.config.tactile_collections[side])
            for side in ("left", "right")
        }
        replay = self.config.replay_path is not None
        replay_config = None
        if replay:
            replay_config = McapReplayConfig(
                str(self.config.replay_path),
                [
                    (tactile[side], self.config.tactile_channels[side])
                    for side in ("left", "right")
                ],
            )
        session_config = TeleopSessionConfig(
            app_name="TactileShapeVisualizer",
            pipeline=pipeline,
            mode=SessionMode.REPLAY if replay else SessionMode.LIVE,
            trackers=[] if replay else list(tactile.values()),
            plugins=[] if replay else self._plugins(),
            mcap_config=replay_config,
            verbose=False,
        )
        duration = self._duration_s()
        started = time.monotonic()
        launcher = (
            nullcontext()
            if replay
            else CloudXRLauncher.launch_context(self.launcher_args)
        )
        with launcher, TeleopSession(session_config) as session:
            frame = 0
            while not self._stop.is_set():
                if duration is not None and time.monotonic() - started > duration:
                    return
                try:
                    result = session.step()
                    now = time.monotonic()
                    hand_values: dict[str, HandSample] = {}
                    raw_values: dict[str, dict[str, float]] = {}
                    shapes: dict[str, ShapeDocument] = {}
                    forces: dict[str, np.ndarray] = {}
                    for side in ("left", "right"):
                        hand = self._hand(result.get(f"hand_{side}"), HandInputIndex)
                        raw = self._raw(result.get(f"raw_{side}"))
                        if hand is not None:
                            hand_values[side] = hand
                        if raw is not None:
                            raw_values[side] = raw
                        if side not in self._known_shapes:
                            shape = tactile[side].get_metadata(session.deviceio_session)
                            if shape is not None:
                                self._known_shapes[side] = ShapeDocument.from_schema(
                                    shape
                                )
                        if side in self._known_shapes:
                            shapes[side] = self._known_shapes[side]
                        data = tactile[side].get_data(session.deviceio_session)
                        if data is not None:
                            forces[side] = np.asarray(
                                data.taxel_force, dtype=np.float32
                            ).copy()
                    # Cached metadata alone is not a live sample and must not
                    # keep the connection state alive after producers stop.
                    if hand_values or raw_values or forces:
                        self.mailbox.put(
                            SourceSnapshot(
                                received_at=now,
                                hands=hand_values,
                                raw_joints=raw_values,
                                shapes=shapes,
                                forces=forces,
                            )
                        )
                except Exception as error:
                    logger.exception("Rejected malformed live message")
                    self.mailbox.fail(error)
                frame += 1
                target = started + frame / self.config.replay_rate_hz
                self._stop.wait(max(0.0, target - time.monotonic()))

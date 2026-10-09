# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Configuration dataclasses for TeleopSession.

These classes provide a clean, declarative way to configure teleop sessions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union

import isaaccapture
from isaaccapture.retargeting_engine.deviceio_source_nodes import IDeviceIOSink
from isaaccapture.retargeting_engine.interface.retargeter_core_types import (
    GraphExecutable,
)
from isaaccapture.retargeting_engine.interface.retargeter_subgraph import (
    RetargeterSubgraph,
)
from isaaccapture.retargeting_engine.tensor_types import BoolType

from .teleop_state_manager_types import teleop_control_states

if TYPE_CHECKING:
    from isaaccapture.deviceio_session import (
        McapRecordingConfig,
        McapReplayConfig,
    )
    from isaaccapture.oxr import OpenXRSessionHandles

    # Under TYPE_CHECKING only: isaaccapture.viz is not shipped by a BUILD_VIZ=OFF
    # build, and importing it here would make this module unimportable there.
    from isaaccapture.viz.robot import RobotTwinPublisher


class SessionMode(Enum):
    """Determines whether the teleop session runs live or replays from an MCAP file."""

    LIVE = "live"
    REPLAY = "replay"


class RetargetingExecutionMode(str, Enum):
    """How :class:`TeleopSession` executes the main retargeting graph."""

    SYNC = "sync"
    PIPELINED = "pipelined"


class RetargetingPacingMode(str, Enum):
    """How the retarget worker schedules pipelined requests."""

    IMMEDIATE = "immediate"
    DEADLINE = "deadline"


@dataclass(frozen=True)
class PacingStartupState:
    """Initial estimates the worker keeps for paced scheduling."""

    submit_period_s: float = 0.0
    compute_duration_s: float = 0.0
    compute_sample_window: int = 1


class _PacingBehavior:
    """Default no-delay behavior shared by pacing configs."""

    def startup_state(self) -> PacingStartupState:
        return PacingStartupState()

    def update_submit_period_s(
        self,
        current_estimate_s: float,
        observed_period_s: float,
    ) -> float:
        return current_estimate_s

    def update_compute_duration_s(
        self,
        current_estimate_s: float,
        observed_duration_s: float,
    ) -> float:
        return current_estimate_s

    def compute_delay_s(
        self,
        *,
        submitted_time_s: float,
        now_s: float,
        submission_count: int,
        submit_period_s: float,
        compute_duration_s: float,
        compute_duration_samples: Sequence[float],
    ) -> float:
        return 0.0


@dataclass
class ImmediatePacingConfig(_PacingBehavior):
    """Start retarget work as soon as the worker receives a request."""

    mode: RetargetingPacingMode | str = RetargetingPacingMode.IMMEDIATE

    def __post_init__(self) -> None:
        self.mode = RetargetingPacingMode(self.mode)
        if self.mode != RetargetingPacingMode.IMMEDIATE:
            raise ValueError("ImmediatePacingConfig mode must be 'immediate'")


@dataclass
class DeadlinePacingConfig(_PacingBehavior):
    """Delay retarget work toward the predicted next application frame.

    Tune :attr:`safety_margin_s` first. The adaptation and spike estimate fields
    are for workloads whose frame cadence or retarget cost changes noticeably.
    """

    mode: RetargetingPacingMode | str = RetargetingPacingMode.DEADLINE
    safety_margin_s: float = 0.015
    frame_period_adaptation: float = 0.2
    compute_cost_adaptation: float = 0.25
    spike_guard_window: int = 60
    spike_guard_percentile: float = 0.90
    startup_frame_period_s: float = 0.022
    startup_compute_cost_s: float = 0.005

    def __post_init__(self) -> None:
        self.mode = RetargetingPacingMode(self.mode)
        self._validate()

    def startup_state(self) -> PacingStartupState:
        """Seed worker-side pacing estimates before runtime samples exist."""
        return PacingStartupState(
            submit_period_s=self.startup_frame_period_s,
            compute_duration_s=self.startup_compute_cost_s,
            compute_sample_window=self.spike_guard_window,
        )

    def update_submit_period_s(
        self,
        current_estimate_s: float,
        observed_period_s: float,
    ) -> float:
        """Fold one observed app-frame period into the deadline estimate."""
        if observed_period_s <= 0.0:
            return current_estimate_s
        return (
            self.frame_period_adaptation * observed_period_s
            + (1.0 - self.frame_period_adaptation) * current_estimate_s
        )

    def update_compute_duration_s(
        self,
        current_estimate_s: float,
        observed_duration_s: float,
    ) -> float:
        """Fold one retarget duration into the cost estimate."""
        return (
            self.compute_cost_adaptation * observed_duration_s
            + (1.0 - self.compute_cost_adaptation) * current_estimate_s
        )

    def compute_delay_s(
        self,
        *,
        submitted_time_s: float,
        now_s: float,
        submission_count: int,
        submit_period_s: float,
        compute_duration_s: float,
        compute_duration_samples: Sequence[float],
    ) -> float:
        """Delay work so DeviceIO polling happens near next use.

        Deadline pacing is optional; the immediate default returns zero. This
        mode starts later than immediate pacing to use more recent inputs, but
        still reserves estimated compute time plus ``safety_margin_s`` so the
        next application frame is less likely to miss the finished result.
        """
        if submission_count < 2:
            return 0.0

        target_start_s = (
            submitted_time_s
            + submit_period_s
            - self._estimated_compute_duration_s(
                compute_duration_s,
                compute_duration_samples,
            )
            - self.safety_margin_s
        )
        return max(0.0, target_start_s - now_s)

    def _estimated_compute_duration_s(
        self,
        compute_duration_s: float,
        compute_duration_samples: Sequence[float],
    ) -> float:
        """Return the cost estimate used for spike-aware scheduling."""
        if not compute_duration_samples:
            return compute_duration_s

        samples = sorted(compute_duration_samples)
        index = min(
            len(samples) - 1,
            max(0, math.ceil(self.spike_guard_percentile * len(samples)) - 1),
        )
        return max(compute_duration_s, samples[index])

    def _validate(self) -> None:
        if self.mode != RetargetingPacingMode.DEADLINE:
            raise ValueError("DeadlinePacingConfig mode must be 'deadline'")
        if not (0.0 < self.frame_period_adaptation <= 1.0):
            raise ValueError("frame_period_adaptation must be in (0, 1]")
        if not (0.0 < self.compute_cost_adaptation <= 1.0):
            raise ValueError("compute_cost_adaptation must be in (0, 1]")
        if self.spike_guard_window <= 0:
            raise ValueError("spike_guard_window must be positive")
        if not (0.0 < self.spike_guard_percentile <= 1.0):
            raise ValueError("spike_guard_percentile must be in (0, 1]")
        if self.safety_margin_s < 0:
            raise ValueError("safety_margin_s must be non-negative")
        if self.startup_frame_period_s <= 0:
            raise ValueError("startup_frame_period_s must be positive")
        if self.startup_compute_cost_s < 0:
            raise ValueError("startup_compute_cost_s must be non-negative")


_RetargetingPacingConfig = ImmediatePacingConfig | DeadlinePacingConfig


def _coerce_pacing_config(
    pacing: _RetargetingPacingConfig | RetargetingPacingMode | str,
) -> _RetargetingPacingConfig:
    """Return a concrete pacing config from a config object or mode string."""
    if isinstance(
        pacing,
        (ImmediatePacingConfig, DeadlinePacingConfig),
    ):
        return pacing

    mode = RetargetingPacingMode(pacing)
    if mode == RetargetingPacingMode.IMMEDIATE:
        return ImmediatePacingConfig()
    if mode == RetargetingPacingMode.DEADLINE:
        return DeadlinePacingConfig()

    raise ValueError(f"Unsupported pacing mode: {pacing}")


@dataclass
class RetargetingExecutionConfig:
    """Configuration for synchronous vs. pipelined retarget step execution.

    Pipelined mode keeps the public ``TeleopSession.step()`` return type as a
    normal ``RetargeterIO`` dict, but returns the latest completed retarget
    frame while submitting the current step request to a background worker.
    """

    mode: RetargetingExecutionMode | str = RetargetingExecutionMode.SYNC
    pacing: _RetargetingPacingConfig | RetargetingPacingMode | str = field(
        default_factory=ImmediatePacingConfig
    )

    def __post_init__(self) -> None:
        self.mode = RetargetingExecutionMode(self.mode)
        self.pacing = _coerce_pacing_config(self.pacing)


def _bundled_plugins_dir(package_roots: Sequence[str]) -> Path:
    """The first ``plugins/`` under the package roots, else the first root's."""
    # An editable install splits isaaccapture between the source tree (authored .py, this
    # module) and the install tree (build products, the plugins), so search every root.
    dirs = [Path(root) / "plugins" for root in package_roots]
    return next((d for d in dirs if d.is_dir()), dirs[0])


# Plugins bundled with the isaaccapture package (isaaccapture/plugins/<name>/), so a wheel install
# can start them: pass it as a PluginConfig search path.
BUNDLED_PLUGINS_DIR = _bundled_plugins_dir(isaaccapture.__path__)


@dataclass
class PluginConfig:
    """Configuration for a plugin.

    Attributes:
        plugin_name: Name of the plugin to load
        plugin_root_id: Root ID for the plugin instance
        search_paths: List of directories to search for plugins
        enabled: Whether to load and use this plugin
        plugin_args: Optional list of arguments passed to the plugin process
        required: Whether session startup should fail if the plugin cannot be found
    """

    plugin_name: str
    plugin_root_id: str
    search_paths: List[Path]
    enabled: bool = True
    plugin_args: List[str] = field(default_factory=list)
    required: bool = False


@dataclass
class TwinRenderConfig:
    """How a session-owned robot twin is rendered.

    Only reaches a twin when :attr:`TeleopSessionConfig.joint_publisher` is set.

    Attributes:
        near_z: Near clip plane [m]. Handed to the compositor **and** to the twin, from
            here, because a twin projecting against a different pair renders geometry
            the runtime then reprojects wrongly.
        far_z: Far clip plane [m].
        layer_name: Name of the projection layer the twin is drawn into.
        join_timeout_s: How long teardown waits for the render thread at each step
            before declaring the teardown unclean and leaving the session alive.
        system_wait_seconds: How long entering the session waits for the headset before
            raising ``TimeoutError``. 0 fails fast, as a headless test would.
    """

    near_z: float = 0.05
    far_z: float = 50.0
    layer_name: str = "robot_twin"
    join_timeout_s: float = 5.0
    system_wait_seconds: int = 30

    def __post_init__(self) -> None:
        """Reject clip planes no projection could be built from.

        Raises:
            ValueError: If the planes are not ``0 < near_z < far_z``.
        """
        if not 0.0 < self.near_z < self.far_z:
            raise ValueError(
                f"require 0 < near_z < far_z, got {self.near_z} and {self.far_z}"
            )


@dataclass
class TeleopSessionConfig:
    """Complete configuration for a teleop session.

    Encapsulates all components needed to run a teleop session:
    - Retargeting pipeline (trackers auto-discovered from sources!)
    - Optional teleop control pipeline (state/reset events for ComputeContext)
    - OpenXR application settings
    - Plugin configuration (optional)
    - Manual trackers (optional, for advanced use)
    - Pre-existing OpenXR handles (optional, for external runtime integration)

    Loop control is handled externally by the user.

    Attributes:
        app_name: Name of the OpenXR application
        pipeline: Main retargeting pipeline.
        mode: Whether to run a live OpenXR session or replay from an MCAP file.
            Defaults to ``SessionMode.LIVE``. When ``REPLAY``, ``mcap_config`` is required
            and OpenXR/plugin initialization is skipped.
        teleop_control_pipeline: Optional control pipeline whose outputs are
            decoded into ComputeContext.execution_events before running ``pipeline``.
            Expected outputs:
              - "teleop_state": one-hot bool group [stopped, paused, running]
              - "reset_event": single bool pulse
        sinks: Optional list of device-output sinks. Each entry is an
            :class:`~isaaccapture.retargeting_engine.deviceio_source_nodes.IDeviceIOSink`
            (typically the subgraph returned by ``sink.connect({...})``).
            ``TeleopSession`` discovers each sink subgraph's own leaves
            (sources / external inputs feeding it), executes the sink each frame
            *after* ``pipeline``, then calls ``flush_to_device(session)`` so the
            device write happens in the same frame. Unlike ``pipeline``, sinks do
            not contribute to the value returned by ``step()`` -- they are
            side-effecting terminals (e.g. a haptic ``HapticSink``).
        trackers: Optional list of manual trackers (usually not needed - auto-discovered!)
        plugins: List of plugin configurations
        verbose: Whether to print detailed progress information during setup
        oxr_handles: Optional pre-existing OpenXRSessionHandles from an external runtime
            (e.g. Kit's XR system). When provided, TeleopSession will use these handles
            instead of creating its own OpenXR session via OpenXRSession.create().
            Construct with ``OpenXRSessionHandles(instance, session, space, proc_addr)``
            where each argument is a ``uint64`` handle value.
        mcap_config: MCAP configuration — ``McapRecordingConfig`` for live
            recording, ``McapReplayConfig`` for replay.  **Required** when
            ``mode`` is ``SessionMode.REPLAY``; optional in ``LIVE`` mode.
            TeleopSession always auto-populates tracker names from the
            pipeline's discovered DeviceIO sources (using each source's
            ``name`` as the MCAP channel name).  Any ``tracker_names``
            explicitly provided in the config are **appended** after the
            auto-discovered sources.
        retargeting_execution: Synchronous vs. pipelined execution settings for
            the main retargeting pipeline. Defaults to synchronous exact-current-frame
            behavior; set ``mode="pipelined"`` to opt into background execution.
        joint_publisher: Optional
            :class:`~isaaccapture.viz.robot.RobotTwinPublisher` -- a digital twin of the
            robot being teleoperated, drawn into the operator's headset. Setting it
            makes the session create its own compositor session (so ``oxr_handles``
            must be left unset), own a render thread, and tear both down; no viewer
            type becomes public. Publish joints onto the object you passed in, from
            the control thread; the render thread draws the latest.
        twin_render: Clip planes, layer name and teardown budget for that twin. Ignored
            when ``joint_publisher`` is None.

    Example (auto-discovery):
        # Source creates its own tracker automatically!
        controllers = ControllersSource(name="controllers")

        # Build retargeting pipeline (one GripperRetargeter per side)
        gripper_left = GripperRetargeter(
            GripperRetargeterConfig(hand_side="left"),
            name="gripper_left",
        )
        connected_left = gripper_left.connect({
            ControllersSource.LEFT: controllers.output(ControllersSource.LEFT),
        })
        gripper_right = GripperRetargeter(
            GripperRetargeterConfig(hand_side="right"),
            name="gripper_right",
        )
        connected_right = gripper_right.connect({
            ControllersSource.RIGHT: controllers.output(ControllersSource.RIGHT),
        })
        pipeline = OutputCombiner({
            "gripper_left": connected_left.output("gripper_command"),
            "gripper_right": connected_right.output("gripper_command"),
        })

        # Configure session - NO TRACKERS NEEDED!
        config = TeleopSessionConfig(
            app_name="MyApp",
            pipeline=pipeline,  # Trackers auto-discovered from pipeline
        )

        # Run session
        with TeleopSession(config) as session:
            while True:
                result = session.run()
                left_gripper = result["gripper_left"][0]

    Example (external OpenXR handles from Kit):
        from isaaccapture.oxr import OpenXRSessionHandles

        handles = OpenXRSessionHandles(
            instance_handle, session_handle, space_handle, proc_addr
        )
        config = TeleopSessionConfig(
            app_name="MyApp",
            pipeline=pipeline,
            oxr_handles=handles,  # Skip internal OpenXR session creation
        )
    """

    app_name: str
    pipeline: GraphExecutable
    mode: SessionMode = SessionMode.LIVE
    teleop_control_pipeline: Optional[GraphExecutable] = None
    sinks: List[GraphExecutable] = field(default_factory=list)
    trackers: List[Any] = field(default_factory=list)
    plugins: List[PluginConfig] = field(default_factory=list)
    verbose: bool = True
    oxr_handles: Optional[OpenXRSessionHandles] = None
    mcap_config: Optional[Union[McapRecordingConfig, McapReplayConfig]] = None
    retargeting_execution: RetargetingExecutionConfig = field(
        default_factory=RetargetingExecutionConfig
    )
    joint_publisher: Optional["RobotTwinPublisher"] = None
    twin_render: TwinRenderConfig = field(default_factory=TwinRenderConfig)

    def __post_init__(self) -> None:
        """Validate configuration consistency."""
        if self.mode == SessionMode.REPLAY and self.mcap_config is None:
            raise ValueError("mcap_config is required when mode is SessionMode.REPLAY")

        if self.joint_publisher is not None:
            # The twin's compositor session IS the OpenXR session, so there is nothing
            # for caller-supplied handles to mean here: one of the two would have to be
            # ignored, and either choice is a silently different session.
            if self.oxr_handles is not None:
                raise ValueError(
                    "joint_publisher and oxr_handles are mutually exclusive: the twin "
                    "creates the OpenXR session that the trackers then share."
                )
            if self.mode == SessionMode.REPLAY:
                raise ValueError(
                    "joint_publisher requires a live session; REPLAY has no headset "
                    "to render into."
                )

        self._validate_sinks()

        if self.teleop_control_pipeline is None:
            return

        output_types = self.teleop_control_pipeline.output_types()
        if "teleop_state" not in output_types:
            raise ValueError(
                "teleop_control_pipeline must output 'teleop_state' "
                "(one-hot stopped/paused/running)"
            )
        if "reset_event" not in output_types:
            raise ValueError(
                "teleop_control_pipeline must output 'reset_event' (single bool pulse)"
            )

        teleop_state_type = output_types["teleop_state"]
        if teleop_state_type.is_optional:
            raise ValueError(
                "teleop_control_pipeline output 'teleop_state' channels must be "
                "plain BoolType (OptionalType is not allowed)"
            )
        expected_channels = [state.value for state in teleop_control_states()]
        actual_channels = [tensor_type.name for tensor_type in teleop_state_type.types]
        if set(actual_channels) != set(expected_channels):
            raise ValueError(
                "teleop_control_pipeline output 'teleop_state' must expose one bool "
                f"channel per execution state {expected_channels} "
                f"(got channels: {actual_channels})"
            )
        if len(actual_channels) != len(set(actual_channels)):
            raise ValueError(
                "teleop_control_pipeline output 'teleop_state' contains duplicate "
                f"channels: {actual_channels}"
            )
        for tensor_type in teleop_state_type.types:
            if not isinstance(tensor_type, BoolType):
                raise ValueError(
                    "teleop_control_pipeline output 'teleop_state' channels must be "
                    f"BoolType, got {type(tensor_type).__name__}"
                )

        reset_event_type = output_types["reset_event"]
        if reset_event_type.is_optional:
            raise ValueError(
                "teleop_control_pipeline output 'reset_event' must be a plain "
                "BoolType scalar (OptionalType is not allowed)"
            )
        if len(reset_event_type.types) != 1:
            raise ValueError(
                "teleop_control_pipeline output 'reset_event' must have exactly 1 channel"
            )
        if not isinstance(reset_event_type.types[0], BoolType):
            raise ValueError(
                "teleop_control_pipeline output 'reset_event' must be a bool scalar "
                f"(got {type(reset_event_type.types[0]).__name__})"
            )

    def _validate_sinks(self) -> None:
        """Ensure every ``sinks`` entry resolves to an ``IDeviceIOSink``.

        Accepts either a bare sink node or the subgraph returned by
        ``sink.connect({...})`` (whose ``target_module`` is the sink).
        """
        for entry in self.sinks:
            target = (
                entry.target_module if isinstance(entry, RetargeterSubgraph) else entry
            )
            if not isinstance(target, IDeviceIOSink):
                raise ValueError(
                    "TeleopSessionConfig.sinks entries must be an IDeviceIOSink "
                    "(or a subgraph wrapping one, e.g. sink.connect({...})); got "
                    f"{type(target).__name__}"
                )

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TeleopSession - A high-level wrapper for complete teleop pipelines.

This class encapsulates all the boilerplate for setting up DeviceIO sessions,
plugins, and retargeting engines, allowing users to focus on configuration
rather than initialization code.
"""

import logging
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import isaaccapture.plugin_manager as pm
from isaaccapture import deviceio, deviceio_trackers, oxr
from isaaccapture.logging_config._native_api import capture_native_output
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    IDeviceIOSink,
    IDeviceIOSource,
)
from isaaccapture.retargeting_engine.interface import BaseRetargeter
from isaaccapture.retargeting_engine.interface.execution_events import (
    ExecutionEvents,
    ExecutionState,
)
from isaaccapture.retargeting_engine.interface.retargeter_core_types import (
    ComputeContext,
    ExecutionCache,
    GraphExecutable,
    GraphTime,
    RetargeterIO,
    RetargeterIOType,
)
from isaaccapture.retargeting_engine.interface.retargeter_subgraph import (
    RetargeterSubgraph,
)

from .async_retarget_runner import (
    AsyncRetargetRunner,
    AsyncRetargetRunnerStopped,
    AsyncRetargetWorkerError,
    RetargetFrame,
    StepRequest,
    snapshot_compute_context,
    snapshot_pipeline_inputs,
    snapshot_retargeter_io,
)
from .config import (
    RetargetingExecutionMode,
    SessionMode,
    TeleopSessionConfig,
)
from .helpers import build_vendor_config_from_sources
from .status import DeviceStatus, ProviderStatus, StatusSnapshot
from .status_monitor import PluginProviderSpec, StatusMonitor
from .teleop_state_manager_types import teleop_control_states

logger = logging.getLogger(__name__)

# Plugins that publish device status (see PluginDeviceStatusPublisher); others launch unmonitored.
_MONITORED_PLUGIN_NAMES = frozenset({"manus_hand_plugin", "spacemouse"})


def _resolve_sink(entry: GraphExecutable) -> tuple[GraphExecutable, IDeviceIOSink]:
    """Resolve a configured ``sinks`` entry into ``(executable, sink_node)``.

    ``entry`` is either a bare :class:`IDeviceIOSink` or the
    :class:`RetargeterSubgraph` returned by ``sink.connect({...})`` (whose
    ``target_module`` is the sink). The executable is what the session runs each
    frame; the sink node is what it flushes to the device afterwards.
    """
    target = entry.target_module if isinstance(entry, RetargeterSubgraph) else entry
    if not isinstance(target, IDeviceIOSink):
        raise TypeError(
            "TeleopSession sinks must be an IDeviceIOSink (or a subgraph wrapping "
            f"one, e.g. sink.connect({{...}})); got {type(target).__name__}"
        )
    return entry, target


@dataclass
class RetargetingStepInfo:
    """Age and timing metadata for the most recent ``step()`` return.

    ``returned_frame_id`` and ``submitted_frame_id`` identify which completed
    frame was returned and which request this call submitted. Pipelined callers
    can use ``returned_age_frames`` to observe result age without implying any
    current-frame wait. Counts such as ``dropped_submissions`` and
    ``frame_deadline_miss`` are per-step so apps can accumulate them for
    run-wide debug summaries.
    """

    returned_frame_id: int | None = None
    submitted_frame_id: int | None = None
    returned_age_frames: int | None = None
    returned_age_s: float | None = None
    compute_duration_s: float | None = None
    dropped_submissions: int = 0
    ran_synchronously: bool = False
    frame_deadline_miss: bool = False
    worker_exception: BaseException | None = None


class TeleopSession:
    """High-level teleop session manager with RAII pattern.

    This class manages the complete lifecycle of a teleop session using
    Python's context manager protocol.

    The session handles:
    1. Creating OpenXR session with required extensions
    2. Creating DeviceIO session with trackers
    3. Initializing plugins
    4. Running retargeting pipeline via step() method
    5. Cleanup on exit

    Pipelines may contain leaf nodes that are DeviceIO sources (auto-polled from
    hardware trackers) and/or leaf nodes that are regular retargeters requiring
    external inputs. External inputs are provided by the caller when calling step().

    Usage with DeviceIO-only pipeline:
        controllers = ControllersSource(name="controllers")
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

        config = TeleopSessionConfig(app_name="MyApp", pipeline=pipeline)
        with TeleopSession(config) as session:
            while True:
                result = session.step()
                left = result["gripper_left"][0]

    Usage with external (non-DeviceIO) inputs via ValueInput:
        controllers = ControllersSource(name="controllers")
        ee_state = ValueInput("ee_state", RobotEEState())  # external leaf
        pipeline = combiner.connect({
            "controller_left": controllers.output("controller_left"),
            "ee_pos": ee_state.output("value"),
        })

        config = TeleopSessionConfig(app_name="MyApp", pipeline=pipeline)
        with TeleopSession(config) as session:
            ext_specs = session.get_external_input_specs()
            # ext_specs == {"ee_state": {"value": TensorGroupType(...)}}

            while True:
                ee_tg = TensorGroup(RobotEEState())
                ee_tg[0] = get_ee_position()

                result = session.step(external_inputs={
                    "ee_state": {"value": ee_tg}
                })

    Any BaseRetargeter that is a leaf (not connected to a DeviceIO source)
    automatically becomes an external input -- ValueInput is just a shortcut
    for the common single-value passthrough case. See external_inputs_example.py.
    """

    def __init__(self, config: TeleopSessionConfig):
        """Initialize the teleop session.

        Discovers sources and trackers from the pipeline and prepares for session creation.
        Actual resource creation happens in __enter__.

        Args:
            config: Complete configuration including pipeline (trackers auto-discovered)
        """
        self.config = config
        self.pipeline: GraphExecutable = config.pipeline
        self.teleop_control_pipeline: GraphExecutable | None = (
            config.teleop_control_pipeline
        )

        # Core components (will be created in __enter__)
        self._oxr_session: oxr.OpenXRSession | None = None
        self.deviceio_session: Any | None = None
        self.plugin_managers: list[pm.PluginManager] = []
        self.plugin_contexts: list[Any] = []
        self._resolved_plugins: list[tuple[Any, Any, PluginProviderSpec | None]] = []
        self._status_monitor = StatusMonitor()

        # The robot twin's render thread, when config.joint_publisher is set.
        self._twin_runner: Any | None = None
        self._twin_teardown_clean: bool | None = None

        # Exit stack for RAII resource management
        self._exit_stack = ExitStack()

        # Auto-discovered sources
        self._sources: list[IDeviceIOSource] = []

        # External (non-DeviceIO) leaf nodes that require caller-provided inputs
        self._external_leaves: list[BaseRetargeter] = []

        # Cached leaf name sets for filtering pipeline inputs
        self._main_leaf_names: set[str] = set()
        self._control_leaf_names: set[str] = set()
        self._sink_leaf_names: set[str] = set()

        # Output sinks discovered from config: (executable, sink node) pairs.
        # The executable is run each frame (after the main pipeline); the sink
        # node is then flushed to its device with the active session.
        self._sinks: list[tuple[GraphExecutable, IDeviceIOSink]] = []

        # Runtime state
        self.frame_count: int = 0
        self.start_time: float = 0.0
        self._last_context: ComputeContext | None = None
        self._last_step_info = RetargetingStepInfo()
        self._last_execution_state: ExecutionState | None = None
        self._async_runner: AsyncRetargetRunner | None = None
        self._active_retargeting_execution_mode: RetargetingExecutionMode | None = None
        # True from the start of __enter__ until __exit__ or rollback finishes.
        self._in_context: bool = False
        # Discover sources and external leaves from pipeline
        self._discover_sources()
        # Vendor selection is a live-only concern; reject it up front in replay.
        self._reject_vendor_selection_in_replay()

    @property
    def oxr_session(self) -> oxr.OpenXRSession | None:
        """The internal OpenXR session, or ``None`` when using external handles or after the context manager exits (read-only)."""
        return self._oxr_session

    @property
    def twin_teardown_clean(self) -> bool | None:
        """Did the robot twin's render thread shut down cleanly?

        ``None`` when no twin was configured or the session has not exited yet. **False
        means the compositor session was not destroyed and is still live**: a thread is
        somewhere inside the OpenXR runtime, and only that thread may destroy what it
        made. A caller that owns the runtime -- one that launched CloudXR itself --
        must NOT stop it on that path; the non-daemon thread keeps the process alive
        and the OS reaps everything at exit.

        ``__exit__`` does not raise on False. There is nothing the caller can do about
        it that the caller is not already better placed to decide.
        """
        return self._twin_teardown_clean

    @property
    def twin_resolution(self) -> Any | None:
        """Per-view resolution the twin was built at, or ``None`` without one."""
        return None if self._twin_runner is None else self._twin_runner.resolution

    def take_twin_recentered(self) -> bool:
        """True once per runtime recentre (guardian recenter, room rebound).

        False without a twin. Clears on read. A caller holding a pose latched in the XR
        reference space must re-anchor it and disengage for one frame; carrying it across
        is the XR analogue of carrying a pose across a ``map -> odom`` jump.
        """
        runner = self._twin_runner
        return runner.take_recentered() if runner is not None else False

    @property
    def twin_head_pose(self) -> Any | None:
        """The last rendered frame's 7-D head pose ``[x, y, z, qx, qy, qz, qw]``.

        ``None`` without a twin, and before its first rendered frame. Read from the
        control thread; it is whatever the render thread last saw, not a pose belonging
        to this step. Anchoring content to where the operator stood is what it is for --
        an app deriving per-frame motion from it is reading the wrong clock.
        """
        return None if self._twin_runner is None else self._twin_runner.head_pose

    @property
    def twin_rendering(self) -> bool:
        """Whether the twin's frame loop is still running.

        Goes False when the runtime asks the session to close, which is the operator
        taking the headset off or the runtime going away -- an app that wants to stop
        on that has to poll it, because nothing else reports it.
        """
        return self._twin_runner is not None and self._twin_runner.rendering

    @property
    def last_context(self) -> ComputeContext | None:
        """Most recent ComputeContext produced by ``step()``, or ``None`` before first step."""
        return self._last_context

    @property
    def last_step_info(self) -> RetargetingStepInfo:
        """Age and timing metadata for the most recent ``step()`` return."""
        return self._last_step_info

    def get_status(self) -> StatusSnapshot:
        """Return the current cached provider/device snapshot without doing I/O."""
        return self._status_monitor.get_status()

    def get_provider_status(self, provider_id: str) -> ProviderStatus | None:
        """Return one cached provider status, or ``None`` for an unknown ID."""
        return self._status_monitor.get_provider_status(provider_id)

    def get_device_status(self, device_id: str) -> DeviceStatus | None:
        """Return one cached device status, or ``None`` for an unknown ID."""
        return self._status_monitor.get_device_status(device_id)

    def _discover_sources(self) -> None:
        """Discover DeviceIO sources, output sinks, and external leaf nodes.

        Traverses the main pipeline, the teleop control pipeline, and every
        registered output sink subgraph to find all leaf nodes, partitioning
        them into:
        - IDeviceIOSource instances (auto-polled from hardware trackers)
        - External leaves (non-DeviceIO leaves with non-empty input_spec() that
          require caller-provided inputs in step()). Leaves with empty input_spec()
          (e.g. fixed-command retargeters) are not treated as external and do not
          require external_inputs.

        Sinks are resolved into ``self._sinks`` and their own upstream leaves
        (the sources / external inputs feeding them) are folded into the same
        discovery, so a heatmap/force that feeds *only* a sink is still polled
        or requested via ``step(external_inputs=...)``.
        """
        main_leaf_nodes = self.pipeline.get_leaf_nodes()
        control_leaf_nodes: list[BaseRetargeter] = []
        if self.teleop_control_pipeline is not None:
            control_leaf_nodes = self.teleop_control_pipeline.get_leaf_nodes()

        # Resolve registered sinks and gather the leaves feeding each sink
        # subgraph (excluding the sink node itself, which is a consumer, not a
        # leaf input).
        self._sinks = [_resolve_sink(entry) for entry in self.config.sinks]
        sink_leaf_nodes: list[BaseRetargeter] = []
        for executable, sink_node in self._sinks:
            for leaf in executable.get_leaf_nodes():
                if leaf is sink_node:
                    continue
                sink_leaf_nodes.append(leaf)

        leaf_nodes = main_leaf_nodes + control_leaf_nodes + sink_leaf_nodes
        main_leaf_ids = {id(node) for node in main_leaf_nodes}
        control_leaf_ids = {id(node) for node in control_leaf_nodes}

        # Cache leaf name sets for filtering pipeline inputs
        self._main_leaf_names = {node.name for node in main_leaf_nodes}
        self._control_leaf_names = {node.name for node in control_leaf_nodes}
        self._sink_leaf_names = {node.name for node in sink_leaf_nodes}

        # Leaf names must be unique across the pipeline, control pipeline, and
        # sinks because they are used as top-level keys in collected inputs.
        seen_name_to_id: dict[str, int] = {}
        for node in leaf_nodes:
            existing_id = seen_name_to_id.get(node.name)
            if existing_id is not None and existing_id != id(node):
                raise ValueError(
                    "Duplicate leaf node name detected across pipeline, "
                    "teleop_control_pipeline, and sinks: "
                    f"'{node.name}' (node ids: {existing_id}, {id(node)})"
                )
            seen_name_to_id[node.name] = id(node)

        self._sources = []
        self._external_leaves = []
        visited_nodes = set()
        for node in leaf_nodes:
            if id(node) in visited_nodes:
                continue
            visited_nodes.add(id(node))
            if isinstance(node, IDeviceIOSource):
                self._sources.append(node)
            elif node.input_spec():
                if id(node) in control_leaf_ids and id(node) not in main_leaf_ids:
                    raise ValueError(
                        "teleop_control_pipeline contains an external-input leaf "
                        f"'{node.name}', which is not supported. Control pipeline "
                        "inputs must come from DeviceIO sources (or be no-input nodes)."
                    )
                self._external_leaves.append(node)

        # Create tracker-to-source mapping for efficient lookup
        self._tracker_to_source: dict[Any, Any] = {}
        for source in self._sources:
            tracker = source.get_tracker()
            self._tracker_to_source[id(tracker)] = source

    def _reject_vendor_selection_in_replay(self) -> None:
        """Reject source-carried vendor selections when mode is REPLAY.

        Vendor selection is a live-session concern: the live path routes each
        source's ``get_vendor()`` into a ``VendorConfig``, but replay reads the
        recorded channel regardless of vendor. Without this guard a vendor set on
        a source would be silently ignored in REPLAY. Fail fast instead, matching
        the fail-fast convention on the live path (unknown vendor ids and
        non-vendored trackers both raise).
        """
        if self.config.mode != SessionMode.REPLAY:
            return
        vendored = [
            source for source in self._sources if source.get_vendor() is not None
        ]
        if vendored:
            names = ", ".join(sorted(source.name for source in vendored))
            raise ValueError(
                f"Vendor selection is only valid in SessionMode.LIVE, but "
                f"source(s) {names} carry a vendor and mode is SessionMode.REPLAY. "
                f"Replay reads the recorded channel regardless of vendor; remove "
                f"the vendor selection or run in LIVE mode."
            )

    def get_external_input_specs(self) -> dict[str, RetargeterIOType]:
        """Get the input specifications for all external leaf nodes that need inputs.

        Only includes non-DeviceIO leaves with non-empty input_spec(). Leaves with
        empty input_spec() (e.g. fixed-command retargeters) are not included.

        Returns:
            Dict mapping leaf node name to its input_spec (Dict[str, TensorGroupType]).
            Empty dict if no leaves require caller-provided inputs.

        Example:
            specs = session.get_external_input_specs()
            # specs == {"sim_state": {"joint_positions": TensorGroupType(...)}}
        """
        return {leaf.name: leaf.input_spec() for leaf in self._external_leaves}

    def has_external_inputs(self) -> bool:
        """Check whether this pipeline requires caller-provided external inputs.

        Returns True only if there are leaf nodes that are not DeviceIO sources
        and have a non-empty input_spec() (i.e. they need inputs in step()).
        """
        return len(self._external_leaves) > 0

    def step(
        self,
        *,
        external_inputs: dict[str, RetargeterIO] | None = None,
        graph_time: GraphTime | None = None,
        execution_events: ExecutionEvents | None = None,
    ):
        """Execute a single step of the teleop session.

        In sync mode, updates DeviceIO session, polls tracker data, merges any
        caller-provided external inputs, and executes the retargeting pipeline
        before returning. In pipelined mode after the seed frame, this call
        submits that work to the retarget worker and returns the latest
        completed output; an unstarted pending request may be replaced by a
        newer submission.

        ``TeleopSession.step()`` is single-caller application-loop API. The
        async runner serializes retarget work internally, but session fields
        such as ``frame_count`` and ``last_step_info`` are intentionally owned
        by the application thread that calls ``step()``.

        Args:
            external_inputs: Optional dict mapping external leaf node names to their
                input data (Dict[str, TensorGroup]). Required when the pipeline has
                leaf nodes that require caller-provided inputs. Use get_external_input_specs()
                to discover what external inputs are expected. Keys that do not correspond
                to an external leaf node or a DeviceIO source name are silently ignored.
                Within each external leaf, keys not declared by that leaf's input_spec()
                are silently ignored.
                Keys that collide with a DeviceIO source name are invalid and cause
                validation to raise.
            graph_time: Optional ``GraphTime`` for this step. When omitted,
                both sim/real time are initialized from the current monotonic clock.
            execution_events: Optional externally-specified execution control events.
                When provided, these values are injected into ``ComputeContext`` and
                the session will skip running ``teleop_control_pipeline`` for this step.

        Returns:
            Dict[str, TensorGroup] - Output from the retargeting pipeline. The
            per-step context is available via ``session.last_context``. In
            pipelined mode this is the latest completed retarget result, which
            can be older than the current submitted frame; age/timing metadata is
            available via ``session.last_step_info``.

        Raises:
            ValueError: If external leaves exist but external_inputs is missing or
                incomplete, or if external_inputs contains keys that collide with
                DeviceIO source names.
            RuntimeError: If a critical DeviceIO/tracker/runtime failure occurs
                while updating the session. This is a fatal condition; the
                application is expected to terminate rather than continue.
        """
        if self.frame_count % 60 == 0:
            self._check_plugin_health()

        execution_mode = (
            self._active_retargeting_execution_mode
            or self.config.retargeting_execution.mode
        )
        # Latch sync vs. pipelined for the active context-manager run. The
        # config object is mutable, but switching modes while a worker is live
        # could otherwise execute the same graph concurrently.
        if execution_mode == RetargetingExecutionMode.PIPELINED:
            return self._step_pipelined(
                external_inputs=external_inputs,
                graph_time=graph_time,
                execution_events=execution_events,
            )

        return self._step_sync(
            external_inputs=external_inputs,
            graph_time=graph_time,
            execution_events=execution_events,
        )

    def _build_step_request(
        self,
        *,
        external_inputs: dict[str, RetargeterIO] | None,
        graph_time: GraphTime | None,
        execution_events: ExecutionEvents | None,
        snapshot_external_inputs: bool = True,
    ) -> StepRequest:
        """Snapshot caller-owned step arguments into a worker request.

        The whole-step worker polls DeviceIO itself, so only explicit
        ``external_inputs`` cross the thread boundary. Those inputs, along with
        optional graph time and explicit execution events, are filtered to the
        external leaf specs before optionally being copied here so the
        application can safely reuse or mutate its objects after ``step()``
        returns. Sync mode disables the snapshot and uses this same request
        shape only to avoid duplicating the step execution path.
        """
        self._validate_external_inputs(external_inputs)
        external_inputs = self._filter_external_inputs(external_inputs)
        request_external_inputs = None
        if external_inputs:
            request_external_inputs = (
                snapshot_pipeline_inputs(external_inputs)
                if snapshot_external_inputs
                else external_inputs
            )

        return StepRequest(
            frame_id=self.frame_count,
            external_inputs=request_external_inputs,
            graph_time=GraphTime(
                sim_time_ns=graph_time.sim_time_ns,
                real_time_ns=graph_time.real_time_ns,
            )
            if graph_time is not None
            else None,
            execution_events=ExecutionEvents(
                reset=bool(execution_events.reset),
                execution_state=ExecutionState(execution_events.execution_state),
            )
            if execution_events is not None
            else None,
            submitted_time_s=time.monotonic(),
        )

    def _execute_step_request(
        self,
        request: StepRequest,
    ) -> tuple[RetargeterIO, ComputeContext]:
        """Execute one normal synchronous step for ``request``.

        Pipelined mode deliberately moves this whole method to the worker
        thread. Keeping DeviceIO polling, control decoding, and graph execution
        together preserves the old synchronous ordering and avoids passing raw
        DeviceIO source state across threads. Sync mode calls the same method
        directly so the behavioral core stays in one place.
        """
        try:
            self.deviceio_session.update()
        except BaseException as error:
            self._status_monitor.mark_runtime_failed(error)
            raise
        self._status_monitor.refresh(self.deviceio_session, self._oxr_session)
        pipeline_inputs = self._collect_tracker_data()

        if request.external_inputs:
            pipeline_inputs.update(request.external_inputs)

        now_ns = time.monotonic_ns()
        graph_time = request.graph_time
        if graph_time is None:
            graph_time = GraphTime(sim_time_ns=now_ns, real_time_ns=now_ns)

        execution_events = request.execution_events
        if execution_events is None and self.teleop_control_pipeline is not None:
            control_inputs = {
                k: v
                for k, v in pipeline_inputs.items()
                if k in self._control_leaf_names
            }
            control_outputs = self.teleop_control_pipeline.execute_pipeline(
                control_inputs
            )
            execution_events = self._decode_teleop_control_events(control_outputs)
        elif execution_events is None:
            # Auto-fire START on the first step when no control pipeline is configured
            execution_events = ExecutionEvents(
                reset=False, execution_state=ExecutionState.RUNNING
            )

        context = ComputeContext(
            graph_time=GraphTime(
                sim_time_ns=graph_time.sim_time_ns,
                real_time_ns=graph_time.real_time_ns,
            ),
            execution_events=ExecutionEvents(
                reset=bool(execution_events.reset),
                execution_state=ExecutionState(execution_events.execution_state),
            ),
        )

        # Fast path with no output sinks: run the main pipeline exactly as
        # before (preserves the GraphExecutable.execute_pipeline contract).
        if not self._sinks:
            main_inputs = {
                k: v for k, v in pipeline_inputs.items() if k in self._main_leaf_names
            }
            return self.pipeline.execute_pipeline(main_inputs, context), context

        # Sinks present: drive the main pipeline and every sink subgraph through
        # one shared ExecutionCache so nodes shared between them (e.g. a stateful
        # smoothing retargeter feeding both a returned output and a sink) compute
        # exactly once per frame rather than advancing their state twice.
        graph_leaf_names = self._main_leaf_names | self._sink_leaf_names
        graph_inputs = {
            k: v for k, v in pipeline_inputs.items() if k in graph_leaf_names
        }
        cache = ExecutionCache(graph_inputs, context)
        main_outputs = self.pipeline.execute_pipeline_with_cache(cache)

        # Output phase: run each sink subgraph (its _compute_fn stores this
        # frame's per-endpoint values on the device), then flush every sink to
        # hardware with the active session. The IDeviceIOSink/IHapticDevice
        # contract is non-throwing, so a device hiccup cannot tear down the loop.
        for executable, _sink_node in self._sinks:
            executable.execute_pipeline_with_cache(cache)
        for _executable, sink_node in self._sinks:
            sink_node.flush_to_device(self.deviceio_session)

        return main_outputs, context

    def _make_retarget_frame(
        self,
        request: StepRequest,
        outputs: RetargeterIO,
        context: ComputeContext,
        *,
        started_time_s: float,
        completed_time_s: float,
    ) -> RetargetFrame:
        """Package one completed request as an owned cached frame.

        The first pipelined call runs on the application thread as a seed frame
        before the worker exists. It still enters the same latest-frame cache,
        so it must follow the worker invariant: cached outputs/context are
        owned by IsaacTeleop and can be returned later without aliasing
        retargeter-owned reusable buffers.
        """
        return RetargetFrame(
            frame_id=request.frame_id,
            outputs=snapshot_retargeter_io(outputs),
            context=snapshot_compute_context(context),
            submitted_time_s=request.submitted_time_s,
            started_time_s=started_time_s,
            completed_time_s=completed_time_s,
            compute_duration_s=completed_time_s - started_time_s,
        )

    def _step_sync(
        self,
        *,
        external_inputs: dict[str, RetargeterIO] | None,
        graph_time: GraphTime | None,
        execution_events: ExecutionEvents | None,
    ) -> RetargeterIO:
        """Execute retargeting synchronously for exact current-frame behavior.

        This is both the escape hatch for users that cannot accept older-frame
        actions and the reference behavior that pipelined mode overlaps with
        the application loop.
        """
        request = self._build_step_request(
            external_inputs=external_inputs,
            graph_time=graph_time,
            execution_events=execution_events,
            snapshot_external_inputs=False,
        )
        started = time.monotonic()
        result, context = self._execute_step_request(request)
        completed = time.monotonic()

        self._last_context = context
        self._last_execution_state = context.execution_events.execution_state
        self._last_step_info = RetargetingStepInfo(
            returned_frame_id=request.frame_id,
            submitted_frame_id=request.frame_id,
            returned_age_frames=0,
            returned_age_s=completed - request.submitted_time_s,
            compute_duration_s=completed - started,
            ran_synchronously=True,
        )

        self.frame_count += 1
        return result

    def _step_pipelined(
        self,
        *,
        external_inputs: dict[str, RetargeterIO] | None,
        graph_time: GraphTime | None,
        execution_events: ExecutionEvents | None,
    ) -> RetargeterIO:
        """Submit a full sync step and return the latest completed output.

        Public ``step()`` remains a normal function returning ``RetargeterIO``.
        In pipelined mode it acts as a small scheduler: submit the current
        application-frame request, then return the latest completed frame.
        """
        if self._async_runner is None:
            return self._step_pipelined_seed(
                external_inputs=external_inputs,
                graph_time=graph_time,
                execution_events=execution_events,
            )

        runner = self._async_runner
        try:
            runner.raise_if_failed()
            request = self._build_step_request(
                external_inputs=external_inputs,
                graph_time=graph_time,
                execution_events=execution_events,
            )

            dropped = runner.submit(request)
            frame = runner.latest()
            if frame is None:
                raise AsyncRetargetRunnerStopped(
                    "Async retarget runner has no completed frame to return"
                )
            return self._return_pipelined_frame(
                frame,
                submitted_frame_id=request.frame_id,
                dropped_submissions=dropped,
                ran_synchronously=False,
            )
        except AsyncRetargetWorkerError as exc:
            self._last_step_info = RetargetingStepInfo(worker_exception=exc)
            raise

    def _step_pipelined_seed(
        self,
        *,
        external_inputs: dict[str, RetargeterIO] | None,
        graph_time: GraphTime | None,
        execution_events: ExecutionEvents | None,
    ) -> RetargeterIO:
        """Run the first pipelined frame synchronously, then start the worker.

        The first call has no previous action to return, so it runs exactly
        once on the application thread. Publishing that seed frame lets later
        calls use the latest-completed path without changing the public return
        type.
        """
        request = self._build_step_request(
            external_inputs=external_inputs,
            graph_time=graph_time,
            execution_events=execution_events,
        )
        started = time.monotonic()
        outputs, context = self._execute_step_request(request)
        completed = time.monotonic()
        frame = self._make_retarget_frame(
            request,
            outputs,
            context,
            started_time_s=started,
            completed_time_s=completed,
        )

        runner = self._ensure_async_runner()
        runner.publish_seed(frame)
        return self._return_pipelined_frame(
            frame,
            submitted_frame_id=request.frame_id,
            dropped_submissions=0,
            ran_synchronously=True,
        )

    def _ensure_async_runner(self) -> AsyncRetargetRunner:
        """Create and start the pipelined retarget worker lazily.

        The worker is session-scoped rather than construction-scoped because
        DeviceIO/OpenXR resources only exist inside the context manager.
        """
        if self._async_runner is None:
            self._async_runner = AsyncRetargetRunner(
                self._execute_step_request,
                self.config.retargeting_execution,
            )
            self._async_runner.start()
        return self._async_runner

    def _return_pipelined_frame(
        self,
        frame: RetargetFrame,
        *,
        submitted_frame_id: int,
        dropped_submissions: int,
        ran_synchronously: bool,
    ) -> RetargeterIO:
        """Publish metadata/context for a pipelined result and finish the step.

        ``last_context`` must always describe the output returned from this
        call, not the request just submitted. The returned output is copied so
        user mutation cannot corrupt the cached latest frame; uncopyable
        outputs should implement ``create_snapshot()`` or use sync mode.
        """
        returned_outputs = snapshot_retargeter_io(frame.outputs)
        returned_context = snapshot_compute_context(frame.context)
        now = time.monotonic()
        returned_age_frames = submitted_frame_id - frame.frame_id
        frame_deadline_miss = returned_age_frames > 1
        self._last_context = returned_context
        self._last_execution_state = returned_context.execution_events.execution_state
        self._last_step_info = RetargetingStepInfo(
            returned_frame_id=frame.frame_id,
            submitted_frame_id=submitted_frame_id,
            returned_age_frames=returned_age_frames,
            returned_age_s=max(0.0, now - frame.submitted_time_s),
            compute_duration_s=frame.compute_duration_s,
            dropped_submissions=dropped_submissions,
            ran_synchronously=ran_synchronously,
            frame_deadline_miss=frame_deadline_miss,
        )
        self.frame_count += 1
        return returned_outputs

    def _decode_teleop_control_events(
        self, control_outputs: RetargeterIO
    ) -> ExecutionEvents:
        """Decode teleop control pipeline outputs into ``ExecutionEvents``."""
        if "teleop_state" not in control_outputs:
            raise ValueError(
                "teleop_control_pipeline must output 'teleop_state' "
                "(one-hot stopped/paused/running)"
            )
        if "reset_event" not in control_outputs:
            raise ValueError(
                "teleop_control_pipeline must output 'reset_event' (single bool pulse)"
            )

        state_group = control_outputs["teleop_state"]
        reset_group = control_outputs["reset_event"]

        expected_states: set[ExecutionState] = set(teleop_control_states())
        if len(reset_group) != 1:
            raise ValueError(
                "teleop_control_pipeline output 'reset_event' must have 1 bool slot"
            )

        state_flags: dict[ExecutionState, bool] = {}
        for idx, tensor_type in enumerate(state_group.group_type.types):
            try:
                channel_state = ExecutionState(tensor_type.name)
            except ValueError as exc:
                raise ValueError(
                    "teleop_control_pipeline output 'teleop_state' contains unknown "
                    f"channel '{tensor_type.name}'. Channels must match ExecutionState."
                ) from exc
            if channel_state in state_flags:
                raise ValueError(
                    "teleop_control_pipeline output 'teleop_state' contains duplicate "
                    f"channel '{channel_state.value}'"
                )
            state_flags[channel_state] = bool(state_group[idx])

        if set(state_flags.keys()) != expected_states:
            missing = sorted(
                state.value for state in (expected_states - set(state_flags))
            )
            raise ValueError(
                "teleop_control_pipeline output 'teleop_state' missing required "
                f"ExecutionState channels: {missing}"
            )

        active_states = [state for state, is_active in state_flags.items() if is_active]
        if len(active_states) != 1:
            raise ValueError(
                "teleop_control_pipeline output 'teleop_state' must be one-hot "
                f"(got {state_flags})"
            )
        return ExecutionEvents(
            execution_state=active_states[0],
            reset=bool(reset_group[0]),
        )

    def _validate_external_inputs(
        self,
        external_inputs: dict[str, RetargeterIO] | None,
    ) -> None:
        """Validate that all required external inputs are provided.

        Checks that:
        1. No external input name collides with a DeviceIO source name (always).
        2. If external leaves exist: all required leaf names and keys are present.

        Args:
            external_inputs: The external inputs provided by the caller.

        Raises:
            ValueError: If external input names collide with source names, or if
                external leaves exist but inputs are missing or incomplete.
        """
        if external_inputs:
            source_names = {source.name for source in self._sources}
            provided_names = set(external_inputs.keys())
            collisions = provided_names & source_names
            if collisions:
                raise ValueError(
                    f"External input names collide with DeviceIO source names: {collisions}. "
                    f"Do not provide external inputs for source nodes; they are polled from hardware."
                )

        if not self._external_leaves:
            return

        expected_names = {leaf.name for leaf in self._external_leaves}

        if external_inputs is None:
            raise ValueError(
                f"Pipeline has external (non-DeviceIO) leaf nodes that require inputs: "
                f"{expected_names}. Pass external_inputs to step(). "
                f"Use get_external_input_specs() to discover required inputs."
            )

        provided_names = set(external_inputs.keys())
        missing = expected_names - provided_names
        if missing:
            raise ValueError(
                f"Missing external inputs for leaf nodes: {missing}. "
                f"Expected inputs for: {expected_names}. "
                f"Use get_external_input_specs() to discover required inputs."
            )

        # Validate per-leaf input keys
        for leaf in self._external_leaves:
            leaf_data = external_inputs[leaf.name]
            expected_keys = set(leaf.input_spec().keys())
            provided_keys = set(leaf_data.keys())
            missing_keys = expected_keys - provided_keys
            if missing_keys:
                raise ValueError(
                    f"External input '{leaf.name}' is missing input keys: {missing_keys}. "
                    f"Expected keys: {expected_keys}. "
                    f"Use get_external_input_specs() to discover required inputs."
                )

    def _filter_external_inputs(
        self,
        external_inputs: dict[str, RetargeterIO] | None,
    ) -> dict[str, RetargeterIO] | None:
        """Drop allowed-but-unused external leaf names and per-leaf input keys.

        The public API has historically ignored extra external leaf names. This
        filtering also keeps sync and pipelined mode aligned when callers pass
        extra per-leaf values. Keep required input names, ignore extras.
        """
        if not external_inputs:
            return None
        leaves_by_name = {leaf.name: leaf for leaf in self._external_leaves}
        filtered_inputs: dict[str, RetargeterIO] = {}
        for name, values in external_inputs.items():
            leaf = leaves_by_name.get(name)
            if leaf is None:
                continue
            expected_keys = set(leaf.input_spec().keys())
            filtered_inputs[name] = {
                input_name: value
                for input_name, value in values.items()
                if input_name in expected_keys
            }
        return filtered_inputs or None

    def _collect_tracker_data(self) -> dict[str, Any]:
        """Collect raw tracking data from all sources and map to module names.

        Each source polls its own tracker via poll_tracker() and returns
        a RetargeterIO dict matching its input_spec().

        Returns:
            Dict mapping source module names to their complete input dictionaries.
            Each input dictionary maps input names to TensorGroups containing raw data.
        """
        return {
            source.name: source.poll_tracker(self.deviceio_session)
            for source in self._sources
        }

    def _check_plugin_health(self):
        """Check health of all running plugins."""
        for plugin_context in self.plugin_contexts:
            try:
                plugin_context.check_health()
            except BaseException:
                self._status_monitor.refresh_plugin_processes()
                raise

    def get_elapsed_time(self) -> float:
        """Get elapsed time since session started."""
        return time.time() - self.start_time

    # ========================================================================
    # Context manager protocol
    # ========================================================================

    def __enter__(self):
        """Enter the context - create sessions and resources.

        Creates OpenXR session (unless external handles were provided),
        DeviceIO session, plugins, and UI. All preparation was done in __init__.

        When ``config.mode`` is ``SessionMode.REPLAY``, an OpenXR session is **not**
        created; instead a replay DeviceIO session is opened from ``config.mcap_config``.

        When ``config.oxr_handles`` is set (live mode), the provided handles are passed
        directly to ``DeviceIOSession.run()`` and no internal OpenXR session
        is created.  The caller is responsible for the external session lifetime.

        Returns:
            self for context manager protocol

        Raises:
            RuntimeError: If the session is already active.
        """
        # Detect nested context-manager use; this does not make the session thread-safe.
        if self._in_context:
            raise RuntimeError("TeleopSession is already active")

        self._in_context = True
        stack = ExitStack()
        # Everything acquired below reaches native code that writes diagnostics
        # straight to a descriptor and exports no log hook: xrCreateInstance /
        # xrCreateSession, the ~50 xrCreateHandTrackerEXT probes inside
        # DeviceIOSession.run(), and each plugin's startup. The scope covers the
        # rollback below too, which tears the same objects down again. See
        # capture_native_output() for what a scope costs while it is open.
        with capture_native_output():
            try:
                self._enter_resources(stack)
                self._exit_stack = stack.pop_all()
            except BaseException as error:
                try:
                    # Give resources the setup error, but do not let one suppress it.
                    stack.__exit__(type(error), error, error.__traceback__)
                except BaseException:
                    logger.exception("Failed to roll back TeleopSession setup")
                finally:
                    self._status_monitor.mark_startup_failed(error)
                    self._oxr_session = None
                    self._in_context = False
                raise

        return self

    def _enter_resources(self, stack: ExitStack) -> None:
        """Acquire and initialize resources for one context-manager run."""
        # config is mutable between construction and entry, so revalidate the
        # replay-only vendor guard here: a mode flipped to REPLAY after __init__
        # must still fail fast rather than silently ignore a source's vendor.
        self._reject_vendor_selection_in_replay()

        # Reset run-scoped plugin containers on each context entry.
        self.plugin_managers = []
        self.plugin_contexts = []
        self._twin_teardown_clean = None
        self._resolved_plugins = []

        live_mode = self.config.mode == SessionMode.LIVE
        self._status_monitor.rebuild(
            (),
            include_openxr=live_mode,
            external_openxr=live_mode and self.config.oxr_handles is not None,
        )
        if live_mode:
            self._resolve_monitored_plugins()
            self._status_monitor.rebuild(
                tuple(
                    spec
                    for _config, _manager, spec in self._resolved_plugins
                    if spec is not None
                ),
                include_openxr=True,
                external_openxr=self.config.oxr_handles is not None,
            )

        # Auto-populate mcap_config from pipeline sources if recording or replaying.
        mcap_config = None
        if self.config.mcap_config is not None:
            mcap_tracker_names = [
                (source.get_tracker(), source.name) for source in self._sources
            ]
            mcap_tracker_names.extend(self.config.mcap_config.get_tracker_names())
            if self.config.mode == SessionMode.REPLAY:
                mcap_config = deviceio.McapReplayConfig(
                    self.config.mcap_config.filename,
                    mcap_tracker_names,
                )
            else:
                mcap_config = deviceio.McapRecordingConfig(
                    self.config.mcap_config.filename,
                    mcap_tracker_names,
                )

        if self.config.mode == SessionMode.REPLAY:
            self.deviceio_session = stack.enter_context(
                deviceio.ReplaySession.run(mcap_config)
            )
        else:
            # Collect trackers from input sources, output sinks, and config,
            # deduplicating by object identity so a tracker shared between an
            # input source and an output sink (e.g. the ControllerTracker used
            # by both ControllersSource and ControllerHapticDevice) is
            # registered exactly once. Sink trackers must be included so their
            # OpenXR extensions (e.g. XR_NVX1_push_tensor for a cross-process
            # device) are aggregated into the session.
            trackers: list[Any] = []
            seen_tracker_ids: set[int] = set()

            def _add_tracker(tracker: Any) -> None:
                if tracker is None or id(tracker) in seen_tracker_ids:
                    return
                seen_tracker_ids.add(id(tracker))
                trackers.append(tracker)

            for source in self._sources:
                _add_tracker(source.get_tracker())
            for _executable, sink_node in self._sinks:
                _add_tracker(sink_node.get_tracker())
            for tracker in self.config.trackers:
                _add_tracker(tracker)
            for _plugin_config, _manager, plugin_spec in self._resolved_plugins:
                if plugin_spec is not None:
                    _add_tracker(plugin_spec.tracker)

            # Vendored trackers carry their vendor on the source, so it travels
            # with the pipeline into the VendorConfig (see the builder's docstring).
            vendor_config = build_vendor_config_from_sources(self._sources)

            # Get required extensions from all trackers
            required_extensions = deviceio.DeviceIOSession.get_required_extensions(
                trackers, vendor_config
            )

            # Resolve OpenXR handles
            if self.config.oxr_handles is not None:
                handles = self.config.oxr_handles
            elif self.config.joint_publisher is not None:
                handles = oxr.OpenXRSessionHandles(
                    *self._start_twin(stack, required_extensions)
                )
            else:
                self._oxr_session = stack.enter_context(
                    oxr.OpenXRSession(self.config.app_name, required_extensions)
                )
                handles = self._oxr_session.get_handles()

            # Create DeviceIO session with all trackers
            self.deviceio_session = stack.enter_context(
                deviceio.DeviceIOSession.run(
                    trackers, handles, mcap_config, vendor_config
                )
            )

        # Plugins retain their historical launch behavior in both live and replay
        # sessions. In live mode, Manus launches only after DeviceIO has created
        # its status reader.
        self._start_configured_plugins(stack)
        self._status_monitor.mark_available()

        if self._twin_runner is not None:
            # After DeviceIOSession, so the first rendered frame cannot precede a
            # tracker that answers on the same handles. Registered here rather than
            # inside _start_twin so the ExitStack unwinds it BEFORE DeviceIOSession:
            # the loop must be out of the runtime before the trackers sharing its
            # handles go away.
            stack.callback(self._stop_twin_rendering)
            self._twin_runner.begin_rendering()

        # Initialize runtime state
        self.frame_count = 0
        self.start_time = time.time()
        self._last_context = None
        self._last_step_info = RetargetingStepInfo()
        self._last_execution_state = None
        self._async_runner = None
        self._active_retargeting_execution_mode = self.config.retargeting_execution.mode

    def _start_twin(
        self, stack: ExitStack, required_extensions: list[str]
    ) -> tuple[int, int, int, int]:
        """Bring up the robot twin's render thread and return its OpenXR handles.

        The same aggregated ``required_extensions`` every tracker asked for: this is
        the ``xrCreateInstance`` call, so an extension discovered afterwards cannot be
        added and the tracker needing it would be silently dead rather than an error.
        ``get_required_oxr_extensions_from_pipeline`` exists so an *external* viz owner
        can precompute that list; this path already has it.
        """
        # Imported here, not at module scope: it reaches isaaccapture.viz, which a build
        # with BUILD_VIZ=OFF does not ship.
        from .twin_runner import TwinRunner

        view = self.config.twin_render
        runner = TwinRunner(
            self.config.joint_publisher,
            app_name=self.config.app_name,
            required_extensions=required_extensions,
            near_z=view.near_z,
            far_z=view.far_z,
            layer_name=view.layer_name,
            join_timeout_s=view.join_timeout_s,
            system_wait_seconds=view.system_wait_seconds,
        )
        # Registered before start(), so a failure part-way through creating the session
        # still joins the thread; and first of everything in _enter_resources, so it
        # unwinds LAST -- the compositor session outlives the trackers borrowing its
        # handles.
        stack.callback(self._destroy_twin)
        self._twin_runner = runner
        runner.start()
        return runner.oxr_handles

    def _stop_twin_rendering(self) -> None:
        """Leave the frame loop; record whether it got out."""
        if self._twin_runner is not None:
            self._twin_teardown_clean = self._twin_runner.stop_rendering()

    def _destroy_twin(self) -> None:
        """Tear the twin's session down on its own thread, and drop the runner."""
        runner, self._twin_runner = self._twin_runner, None
        if runner is None:
            return
        joined = runner.destroy()
        # None means _stop_twin_rendering never ran -- setup failed between _start_twin
        # and its registration, so the loop was never entered and there was nothing to
        # leave. That is clean; bool(None) would report a joined thread as unclean and
        # strand a self-owned runtime.
        stopped = (
            True if self._twin_teardown_clean is None else self._twin_teardown_clean
        )
        self._twin_teardown_clean = stopped and joined

    def _resolve_monitored_plugins(self) -> None:
        """Resolve the initially supported Manus inventory and status reader."""
        for plugin_config in self.config.plugins:
            if (
                not plugin_config.enabled
                or plugin_config.plugin_name not in _MONITORED_PLUGIN_NAMES
            ):
                continue

            manager = self._resolve_plugin(plugin_config)
            if manager is None:
                continue
            if plugin_config.plugin_name not in manager.get_plugin_names():
                self._resolved_plugins.append((plugin_config, manager, None))
                continue

            info = manager.get_plugin_info(plugin_config.plugin_name)
            tracker = deviceio_trackers.PluginDeviceStatusTracker(
                f"{plugin_config.plugin_root_id}/device_status"
            )
            self._resolved_plugins.append(
                (
                    plugin_config,
                    manager,
                    PluginProviderSpec(
                        plugin_root_id=plugin_config.plugin_root_id,
                        name=info.name,
                        devices=tuple(info.devices),
                        tracker=tracker,
                    ),
                )
            )

    def _resolve_plugin(self, plugin_config: Any) -> pm.PluginManager | None:
        """Apply the existing enabled/path/discovery policy to one plugin."""
        valid_paths = [p for p in plugin_config.search_paths if p.is_dir()]
        if not valid_paths:
            if plugin_config.required:
                configured_paths = [str(path) for path in plugin_config.search_paths]
                raise RuntimeError(
                    f"Required plugin {plugin_config.plugin_name!r} has no existing "
                    f"search directories. Configured search paths: {configured_paths!r}. "
                    "Build or install the plugin, or correct PluginConfig.search_paths."
                )
            return None

        manager = pm.PluginManager([str(p) for p in valid_paths])
        plugins = sorted(manager.get_plugin_names())
        if plugin_config.plugin_name not in plugins:
            if plugin_config.required:
                raise RuntimeError(
                    f"Required plugin {plugin_config.plugin_name!r} was not discovered "
                    f"in search paths {[str(path) for path in valid_paths]!r}. "
                    f"Discovered plugins: {plugins!r}. Build or install the plugin, "
                    "or correct PluginConfig.plugin_name or PluginConfig.search_paths."
                )
            return manager
        return manager

    def _start_configured_plugins(self, stack: ExitStack) -> None:
        """Discover and start enabled plugins in configuration order."""
        monitored_by_config = {
            id(plugin_config): (manager, spec)
            for plugin_config, manager, spec in self._resolved_plugins
        }
        for plugin_config in self.config.plugins:
            if not plugin_config.enabled:
                continue

            monitored = monitored_by_config.get(id(plugin_config))
            if monitored is None:
                manager = self._resolve_plugin(plugin_config)
                plugin_spec = None
            else:
                manager, plugin_spec = monitored

            if manager is None:
                continue
            self.plugin_managers.append(manager)
            if plugin_config.plugin_name not in manager.get_plugin_names():
                continue

            context = manager.start(
                plugin_config.plugin_name,
                plugin_config.plugin_root_id,
                plugin_config.plugin_args,
            )
            stack.enter_context(context)
            self.plugin_contexts.append(context)
            if plugin_spec is not None:
                self._status_monitor.bind_plugin_context(
                    plugin_spec.provider_id, context
                )

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context - cleanup resources."""
        if not self._in_context:
            return False

        runner_error = None
        if self._async_runner is not None:
            runner = self._async_runner
            runner.stop()
            try:
                runner.raise_if_failed(only_unreported=True)
            except BaseException as err:
                if exc_type is None:
                    runner_error = err
                else:
                    logger.error(
                        "Async retarget worker failed during TeleopSession cleanup",
                        exc_info=(type(err), err, err.__traceback__),
                    )
            self._async_runner = None
        self._active_retargeting_execution_mode = None

        # ExitStack automatically cleans up all managed contexts in reverse order.
        # Preserve TeleopSession's historical behavior of not suppressing
        # exceptions from the user body, even if a child context manager would.
        try:
            # Teardown is as noisy as construction: destroying the OpenXR
            # session and stopping each plugin both reach the same native code.
            with capture_native_output():
                self._exit_stack.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._status_monitor.mark_stopped()
            # The ExitStack above closes the OpenXR session; drop our reference so the
            # public `oxr_session` property honors its documented None contract post-exit
            # rather than surfacing a torn-down session -- even if a managed context's
            # cleanup raised. (deviceio_session has no such property/contract.)
            self._oxr_session = None
            self._in_context = False

        if runner_error is not None:
            raise runner_error

        return False

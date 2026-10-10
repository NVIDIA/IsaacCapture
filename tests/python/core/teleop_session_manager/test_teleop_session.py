# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for TeleopSession - core logic unit tests.

Tests the TeleopSession class without requiring OpenXR hardware by patching
OpenXR, DeviceIO, and PluginManager (unittest.mock.patch). Covers source
discovery, external input validation, step execution, session lifecycle,
and plugin management.
"""

import logging
import threading
import time
import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from contextlib import contextmanager

from isaaccapture.retargeting_engine.interface import (
    BaseRetargeter,
    ComputeContext,
    ExecutionEvents,
    ExecutionState,
    GraphTime,
    OutputCombiner,
    TensorGroupType,
    TensorGroup,
    TensorType,
)
from isaaccapture.retargeting_engine.interface.retargeter_core_types import (
    RetargeterIO,
    RetargeterIOType,
)
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    IDeviceIOSink,
    IDeviceIOSource,
)
from isaaccapture.retargeting_engine.tensor_types import FloatType

import isaaccapture.teleop_session_manager as teleop_session_manager
from isaaccapture.teleop_session_manager.config import (
    DeadlinePacingConfig,
    ImmediatePacingConfig,
    PluginConfig,
    RetargetingExecutionConfig,
    RetargetingExecutionMode,
    SessionMode,
    TeleopSessionConfig,
    _bundled_plugins_dir,
)
from isaaccapture.teleop_session_manager.async_retarget_runner import (
    AsyncRetargetRunner,
    AsyncRetargetRunnerStopped,
    AsyncRetargetWorkerError,
    RetargetFrame,
    StepRequest,
)
from isaaccapture.teleop_session_manager.teleop_session import TeleopSession
from isaaccapture.teleop_session_manager.teleop_state_manager_types import (
    teleop_state_manager_output_spec,
)


# ============================================================================
# Mock Tracker Classes
# ============================================================================
# These mock trackers replicate the polling APIs of real DeviceIO trackers.


class HeadTracker:
    """Mock head tracker for testing."""

    def __init__(self):
        self._head_data = 42.0

    def get_head(self, session):
        return self._head_data


class HandTracker:
    """Mock hand tracker for testing."""

    def __init__(self):
        self._left_hand = 1.0
        self._right_hand = 2.0

    def get_left_hand(self, session):
        return self._left_hand

    def get_right_hand(self, session):
        return self._right_hand


class ControllerTracker:
    """Mock controller tracker for testing."""

    def __init__(self):
        self._left_controller = 3.0
        self._right_controller = 4.0

    def get_left_controller(self, session):
        return self._left_controller

    def get_right_controller(self, session):
        return self._right_controller


# ============================================================================
# Mock DeviceIO Source Nodes
# ============================================================================


class MockDeviceIOSource(IDeviceIOSource):
    """A mock DeviceIO source that acts as both a retargeter and a source."""

    def __init__(self, source_name: str, tracker, input_names=None):
        self._tracker = tracker
        self._input_names = input_names or ["input_0"]
        super().__init__(source_name)

    def get_tracker(self):
        return self._tracker

    def poll_tracker(self, deviceio_session):
        """Default poll_tracker: creates a TensorGroup per input with value 0.0."""
        source_inputs = self.input_spec()
        result = {}
        for input_name, group_type in source_inputs.items():
            tg = TensorGroup(group_type)
            tg[0] = 0.0
            result[input_name] = tg
        return result

    def input_spec(self) -> RetargeterIOType:
        return {
            name: TensorGroupType(f"type_{name}", [FloatType("value")])
            for name in self._input_names
        }

    def output_spec(self) -> RetargeterIOType:
        return {
            "output_0": TensorGroupType("type_output", [FloatType("value")]),
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        outputs["output_0"][0] = 0.0


class MockHeadSource(MockDeviceIOSource):
    """Mock head source for testing."""

    def __init__(self, name: str = "head"):
        super().__init__(name, HeadTracker(), input_names=["head_pose"])

    def poll_tracker(self, deviceio_session):
        source_inputs = self.input_spec()
        head_data = self._tracker.get_head(deviceio_session)
        result = {}
        for input_name, group_type in source_inputs.items():
            tg = TensorGroup(group_type)
            tg[0] = head_data
            result[input_name] = tg
        return result


class MockHandsSource(MockDeviceIOSource):
    """Mock hands source for testing."""

    def __init__(self, name: str = "hands"):
        super().__init__(name, HandTracker(), input_names=["hand_left", "hand_right"])

    def poll_tracker(self, deviceio_session):
        source_inputs = self.input_spec()
        result = {}
        for input_name, group_type in source_inputs.items():
            tg = TensorGroup(group_type)
            if "left" in input_name:
                tg[0] = self._tracker.get_left_hand(deviceio_session)
            elif "right" in input_name:
                tg[0] = self._tracker.get_right_hand(deviceio_session)
            result[input_name] = tg
        return result


class MockControllersSource(MockDeviceIOSource):
    """Mock controllers source for testing."""

    def __init__(self, name: str = "controllers"):
        super().__init__(
            name,
            ControllerTracker(),
            input_names=["controller_left", "controller_right"],
        )

    def poll_tracker(self, deviceio_session):
        source_inputs = self.input_spec()
        left_controller = self._tracker.get_left_controller(deviceio_session)
        right_controller = self._tracker.get_right_controller(deviceio_session)
        result = {}
        for input_name, group_type in source_inputs.items():
            tg = TensorGroup(group_type)
            if "left" in input_name:
                tg[0] = left_controller
            elif "right" in input_name:
                tg[0] = right_controller
            result[input_name] = tg
        return result


# ============================================================================
# Mock External Retargeter (non-DeviceIO leaf)
# ============================================================================


class MockExternalRetargeter(BaseRetargeter):
    """A mock retargeter that is NOT a DeviceIO source (requires external inputs)."""

    def __init__(self, name: str):
        super().__init__(name)

    def input_spec(self) -> RetargeterIOType:
        return {
            "external_data": TensorGroupType("type_ext", [FloatType("value")]),
        }

    def output_spec(self) -> RetargeterIOType:
        return {
            "result": TensorGroupType("type_result", [FloatType("value")]),
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        outputs["result"][0] = inputs["external_data"][0]


# ============================================================================
# Mock empty input_spec retargeter (generator/constant source)
# ============================================================================


class MockEmptyInputRetargeter(BaseRetargeter):
    """A mock retargeter that is NOT a DeviceIO source and has empty input_spec().

    Models a generator or constant source that does not require caller-provided
    inputs. Should not be treated as an external leaf.
    """

    def __init__(self, name: str):
        super().__init__(name)

    def input_spec(self) -> RetargeterIOType:
        return {}

    def output_spec(self) -> RetargeterIOType:
        return {
            "value": TensorGroupType("type_value", [FloatType("value")]),
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        outputs["value"][0] = 1.0


# ============================================================================
# Mock Pipeline
# ============================================================================


class MockPipeline:
    """Mock pipeline that returns configurable leaf nodes and accepts inputs."""

    def __init__(self, leaf_nodes=None, call_result=None):
        self._leaf_nodes = leaf_nodes or []
        self._call_result = call_result or {}
        self.last_inputs = None

    def get_leaf_nodes(self):
        return self._leaf_nodes

    def execute_pipeline(self, inputs, context=None):
        self.last_inputs = inputs
        return self._call_result

    def __call__(self, inputs, context=None):
        return self.execute_pipeline(inputs, context)


ASYNC_RESULT_TYPE = TensorGroupType("async_result", [FloatType("value")])


class AnyValueType(TensorType):
    """Tensor type for tests that need an opaque Python value."""

    def _check_instance_compatibility(self, other: TensorType) -> bool:
        return self.name == other.name

    def validate_value(self, value) -> None:
        pass


OPAQUE_EXTERNAL_TYPE = TensorGroupType("opaque_external", [AnyValueType("value")])


class UncopyableValue:
    def __deepcopy__(self, memo):
        raise TypeError("cannot pickle opaque value")


def make_async_result(value: float) -> RetargeterIO:
    tg = TensorGroup(ASYNC_RESULT_TYPE)
    tg[0] = float(value)
    return {"result": tg}


def async_result_value(result: RetargeterIO) -> float:
    return result["result"][0]


class AnyExternalRetargeter(BaseRetargeter):
    """External leaf that accepts an opaque test value."""

    def __init__(self, name: str):
        super().__init__(name)

    def input_spec(self) -> RetargeterIOType:
        return {"value": OPAQUE_EXTERNAL_TYPE}

    def output_spec(self) -> RetargeterIOType:
        return {"result": ASYNC_RESULT_TYPE}

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        outputs["result"][0] = (
            1.0 if isinstance(inputs["value"][0], UncopyableValue) else 0.0
        )


class CountingPipeline(MockPipeline):
    """Pipeline that returns its call index and can sleep or fail per call."""

    def __init__(self, *, sleep_s=0.0, fail_on_call=None):
        super().__init__(leaf_nodes=[])
        self.sleep_s = sleep_s
        self.fail_on_call = fail_on_call
        self.call_count = 0
        self.max_active = 0
        self.failed = threading.Event()
        self._active = 0
        self._lock = threading.Lock()

    def execute_pipeline(self, inputs, context=None):
        with self._lock:
            call_idx = self.call_count
            self.call_count += 1
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            if self.sleep_s:
                time.sleep(self.sleep_s)
            if self.fail_on_call == call_idx:
                self.failed.set()
                raise RuntimeError("pipeline boom")
            return make_async_result(call_idx)
        finally:
            with self._lock:
                self._active -= 1


class ExternalEchoPipeline(MockPipeline):
    """Pipeline that echoes an external scalar after blocking its second call."""

    def __init__(self):
        self.leaf = MockExternalRetargeter("sim_state")
        super().__init__(leaf_nodes=[self.leaf])
        self.call_count = 0
        self.second_started = threading.Event()
        self.release_second = threading.Event()
        self.second_done = threading.Event()

    def execute_pipeline(self, inputs, context=None):
        call_idx = self.call_count
        self.call_count += 1
        if call_idx == 1:
            self.second_started.set()
            self.release_second.wait(timeout=2.0)
        value = inputs["sim_state"]["external_data"][0]
        if call_idx == 1:
            self.second_done.set()
        return make_async_result(value)


class ContextEchoPipeline(MockPipeline):
    """Pipeline that blocks its second call, then echoes execution events."""

    def __init__(self):
        super().__init__(leaf_nodes=[])
        self.call_count = 0
        self.second_started = threading.Event()
        self.release_second = threading.Event()
        self.second_done = threading.Event()

    def execute_pipeline(self, inputs, context=None):
        call_idx = self.call_count
        self.call_count += 1
        if call_idx == 1:
            self.second_started.set()
            self.release_second.wait(timeout=2.0)
        if context.execution_events.reset:
            value = 1.0
        elif context.execution_events.execution_state == ExecutionState.PAUSED:
            value = 2.0
        else:
            value = 0.0
        if call_idx == 1:
            self.second_done.set()
        return make_async_result(value)


class FrameIdPipeline(MockPipeline):
    """Pipeline that records and returns the frame id encoded in GraphTime."""

    def __init__(self):
        super().__init__(leaf_nodes=[])
        self.executed_frame_ids: list[int] = []

    def execute_pipeline(self, inputs, context=None):
        frame_id = int(context.graph_time.sim_time_ns)
        self.executed_frame_ids.append(frame_id)
        return make_async_result(frame_id)


class BlockingSecondCountingPipeline(CountingPipeline):
    """Counting pipeline that holds its second call until released."""

    def __init__(self):
        super().__init__()
        self.second_started = threading.Event()
        self.release_second = threading.Event()

    def execute_pipeline(self, inputs, context=None):
        with self._lock:
            call_idx = self.call_count
            self.call_count += 1
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            if call_idx == 1:
                self.second_started.set()
                self.release_second.wait(timeout=2.0)
            return make_async_result(call_idx)
        finally:
            with self._lock:
                self._active -= 1


class ReusingOutputPipeline(MockPipeline):
    """Pipeline that reuses one output object across calls.

    This mirrors stateful retargeters that keep output buffers internally. The
    async worker must snapshot before publishing, otherwise a later call can
    mutate an older cached frame while the application is still returning it as
    the latest completed output.
    """

    def __init__(self):
        super().__init__(leaf_nodes=[])
        self.outputs = make_async_result(-1.0)
        self.second_started = threading.Event()
        self.release_second = threading.Event()

    def execute_pipeline(self, inputs, context=None):
        frame_id = int(context.graph_time.sim_time_ns)
        self.outputs["result"][0] = float(frame_id)
        if frame_id == 2:
            self.second_started.set()
            self.release_second.wait(timeout=2.0)
        return self.outputs


class UncopyableOutputPipeline(MockPipeline):
    """Pipeline whose output cannot be snapshot-copied."""

    def __init__(self):
        super().__init__(leaf_nodes=[])
        self.group_type = TensorGroupType("opaque_result", [AnyValueType("value")])

    def execute_pipeline(self, inputs, context=None):
        group = TensorGroup(self.group_type)
        group[0] = UncopyableValue()
        return {"result": group}


class BlockingControlPipeline(MockPipeline):
    """Control pipeline that blocks its second automatic event decode."""

    def __init__(
        self,
        *,
        second_state: ExecutionState = ExecutionState.RUNNING,
        second_reset: bool = False,
    ):
        super().__init__(leaf_nodes=[])
        self.call_count = 0
        self.second_started = threading.Event()
        self.release_second = threading.Event()
        self.second_state = second_state
        self.second_reset = second_reset

    def output_types(self) -> RetargeterIOType:
        return teleop_state_manager_output_spec()

    def execute_pipeline(self, inputs, context=None):
        call_idx = self.call_count
        self.call_count += 1
        if call_idx == 1:
            self.second_started.set()
            self.release_second.wait(timeout=2.0)
            return make_control_outputs(self.second_state, reset=self.second_reset)
        return make_control_outputs(ExecutionState.RUNNING, reset=False)


class ControlEventEchoPipeline(MockPipeline):
    """Pipeline that encodes worker-decoded control events in its output."""

    def execute_pipeline(self, inputs, context=None):
        if context.execution_events.reset:
            return make_async_result(1.0)
        if context.execution_events.execution_state == ExecutionState.PAUSED:
            return make_async_result(2.0)
        return make_async_result(0.0)


def make_external_scalar(value: float) -> RetargeterIO:
    tg = TensorGroup(TensorGroupType("external_scalar", [FloatType("value")]))
    tg[0] = float(value)
    return {"external_data": tg}


def make_opaque_external() -> RetargeterIO:
    tg = TensorGroup(OPAQUE_EXTERNAL_TYPE)
    tg[0] = UncopyableValue()
    return {"value": tg}


def make_control_outputs(state: ExecutionState, *, reset: bool) -> RetargeterIO:
    outputs = {}
    for name, group_type in teleop_state_manager_output_spec().items():
        group = TensorGroup(group_type)
        if name == "teleop_state":
            for index, tensor_type in enumerate(group_type.types):
                group[index] = tensor_type.name == state.value
        else:
            group[0] = reset
        outputs[name] = group
    return outputs


def make_step_request(
    frame_id: int,
    *,
    submitted_time_s: float | None = None,
) -> StepRequest:
    return StepRequest(
        frame_id=frame_id,
        external_inputs={},
        graph_time=GraphTime(sim_time_ns=frame_id, real_time_ns=frame_id),
        execution_events=ExecutionEvents(
            reset=False,
            execution_state=ExecutionState.RUNNING,
        ),
        submitted_time_s=(
            time.monotonic() if submitted_time_s is None else submitted_time_s
        ),
    )


def execute_pipeline_request(pipeline, request: StepRequest):
    context = ComputeContext(
        graph_time=request.graph_time
        or GraphTime(sim_time_ns=request.frame_id, real_time_ns=request.frame_id),
        execution_events=request.execution_events or ExecutionEvents(),
    )
    return pipeline.execute_pipeline(request.external_inputs or {}, context), context


def make_async_runner(pipeline, config) -> AsyncRetargetRunner:
    return AsyncRetargetRunner(
        lambda request: execute_pipeline_request(pipeline, request),
        config.retargeting_execution,
    )


class FailingPollSource(MockDeviceIOSource):
    """Source whose first seed-frame poll fails before worker submission."""

    def poll_tracker(self, deviceio_session):
        raise RuntimeError("poll failed")


# ============================================================================
# Mock OpenXR Session
# ============================================================================


class MockOpenXRHandles:
    """Mock OpenXR session handles."""


class MockOpenXRSession:
    """Mock OpenXR session that supports context manager protocol."""

    def __init__(self, app_name="test", extensions=None):
        self.app_name = app_name
        self.extensions = extensions or []
        self._handles = MockOpenXRHandles()
        self.provider_poll_count = 0

    def get_handles(self):
        return self._handles

    def get_provider_snapshot(self):
        self.provider_poll_count += 1
        return SimpleNamespace(
            state="AVAILABLE",
            headset_state="CONNECTED",
            reason="NONE",
            result_code=None,
            error="",
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ============================================================================
# Mock DeviceIO Session
# ============================================================================


class MockDeviceIOSession:
    """Mock DeviceIO session that supports context manager and update."""

    def __init__(self):
        self.update_count = 0

    def update(self):
        self.update_count += 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ============================================================================
# Mock Plugin Infrastructure
# ============================================================================


class MockPluginContext:
    """Mock plugin context that supports context manager and health checks."""

    def __init__(self):
        self.health_check_count = 0
        self.entered = False
        self.exited = False

    def check_health(self):
        self.health_check_count += 1

    def get_process_snapshot(self):
        return SimpleNamespace(
            state="RUNNING",
            reason="NONE",
            pid=123,
            exit_code=None,
            term_signal=None,
            error_code=None,
            error="",
        )

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *args):
        self.exited = True


class MockPluginManager:
    """Mock plugin manager for testing."""

    def __init__(self, plugin_names=None):
        self._plugin_names = plugin_names or []
        self._contexts = {}
        for name in self._plugin_names:
            self._contexts[name] = MockPluginContext()
        self.start_calls: list = []

    def get_plugin_names(self):
        return self._plugin_names

    def get_plugin_info(self, plugin_name):
        return SimpleNamespace(name=plugin_name, devices=())

    def start(self, plugin_name, plugin_root_id, plugin_args=None):
        self.start_calls.append(
            {
                "plugin_name": plugin_name,
                "plugin_root_id": plugin_root_id,
                "plugin_args": plugin_args,
            }
        )
        return self._contexts[plugin_name]


# ============================================================================
# Patch-based session dependencies (no hardware)
# ============================================================================


@contextmanager
def mock_session_dependencies(
    mock_oxr=None,
    mock_dio=None,
    mock_pm=None,
    get_required_extensions_return=None,
    collected_trackers=None,
):
    """Patch OpenXR, DeviceIO, and PluginManager so TeleopSession.__enter__ runs without hardware.

    Use this when a test needs to create a TeleopSession and call __enter__ (e.g. with session:).
    Yields nothing; create TeleopSession(config) inside the block.

    Args:
        mock_oxr: Optional mock OpenXR session (default: MockOpenXRSession()).
        mock_dio: Optional mock DeviceIO session (default: MockDeviceIOSession()).
        mock_pm: Optional mock PluginManager (default: MockPluginManager()).
        get_required_extensions_return: List returned by get_required_extensions (default: []).
        collected_trackers: If provided, trackers passed to get_required_extensions are appended here.
    """
    mock_oxr = mock_oxr or MockOpenXRSession()
    mock_dio = mock_dio or MockDeviceIOSession()
    mock_pm = mock_pm or MockPluginManager()

    if collected_trackers is not None:

        def get_ext_side_effect(trackers, vendor_config=None):
            collected_trackers.extend(trackers)
            return []

        patch_get_ext = patch(
            "isaaccapture.deviceio.DeviceIOSession.get_required_extensions",
            side_effect=get_ext_side_effect,
        )
    else:
        get_ext_return = (
            get_required_extensions_return
            if get_required_extensions_return is not None
            else []
        )
        patch_get_ext = patch(
            "isaaccapture.deviceio.DeviceIOSession.get_required_extensions",
            return_value=get_ext_return,
        )

    with (
        patch("isaaccapture.oxr.OpenXRSession", return_value=mock_oxr),
        patch("isaaccapture.deviceio.DeviceIOSession.run", return_value=mock_dio),
        patch("isaaccapture.plugin_manager.PluginManager", return_value=mock_pm),
        patch_get_ext,
    ):
        yield


# ============================================================================
# Helper to build a TeleopSessionConfig quickly
# ============================================================================


def make_config(
    pipeline,
    plugins=None,
    trackers=None,
    app_name="TestApp",
    teleop_control_pipeline=None,
    retargeting_execution=None,
):
    """Create a TeleopSessionConfig with sensible test defaults."""
    return TeleopSessionConfig(
        app_name=app_name,
        pipeline=pipeline,
        teleop_control_pipeline=teleop_control_pipeline,
        trackers=trackers or [],
        plugins=plugins or [],
        verbose=False,
        retargeting_execution=retargeting_execution
        or RetargetingExecutionConfig(mode=RetargetingExecutionMode.SYNC),
    )


# ============================================================================
# Test Classes
# ============================================================================


class TestSourceDiscovery:
    """Test that _discover_sources correctly partitions leaf nodes."""

    def test_all_deviceio_sources(self):
        """All leaf nodes are DeviceIO sources - no external leaves."""
        head = MockHeadSource()
        controllers = MockControllersSource()
        pipeline = MockPipeline(leaf_nodes=[head, controllers])

        config = make_config(pipeline)
        session = TeleopSession(config)

        assert len(session._sources) == 2
        assert head in session._sources
        assert controllers in session._sources
        assert len(session._external_leaves) == 0

    def test_all_external_leaves(self):
        """All leaf nodes are external retargeters - no DeviceIO sources."""
        ext1 = MockExternalRetargeter("ext1")
        ext2 = MockExternalRetargeter("ext2")
        pipeline = MockPipeline(leaf_nodes=[ext1, ext2])

        config = make_config(pipeline)
        session = TeleopSession(config)

        assert len(session._sources) == 0
        assert len(session._external_leaves) == 2
        assert ext1 in session._external_leaves
        assert ext2 in session._external_leaves

    def test_mixed_sources_and_external(self):
        """Pipeline has both DeviceIO sources and external leaves."""
        head = MockHeadSource()
        ext = MockExternalRetargeter("sim_state")
        pipeline = MockPipeline(leaf_nodes=[head, ext])

        config = make_config(pipeline)
        session = TeleopSession(config)

        assert len(session._sources) == 1
        assert head in session._sources
        assert len(session._external_leaves) == 1
        assert ext in session._external_leaves

    def test_empty_pipeline(self):
        """Pipeline with no leaf nodes."""
        pipeline = MockPipeline(leaf_nodes=[])

        config = make_config(pipeline)
        session = TeleopSession(config)

        assert len(session._sources) == 0
        assert len(session._external_leaves) == 0

    def test_empty_input_spec_leaf_not_external(self):
        """Non-DeviceIO leaf with empty input_spec() is not treated as external."""
        constant_source = MockEmptyInputRetargeter("constant")
        pipeline = MockPipeline(leaf_nodes=[constant_source])

        config = make_config(pipeline)
        session = TeleopSession(config)

        assert len(session._sources) == 0
        assert len(session._external_leaves) == 0

    def test_tracker_to_source_mapping(self):
        """Tracker-to-source mapping is built correctly."""
        head = MockHeadSource()
        controllers = MockControllersSource()
        pipeline = MockPipeline(leaf_nodes=[head, controllers])

        config = make_config(pipeline)
        session = TeleopSession(config)

        # Each source's tracker id should map back to that source
        head_tracker = head.get_tracker()
        controller_tracker = controllers.get_tracker()
        assert id(head_tracker) in session._tracker_to_source
        assert id(controller_tracker) in session._tracker_to_source
        assert session._tracker_to_source[id(head_tracker)] is head
        assert session._tracker_to_source[id(controller_tracker)] is controllers


class TestExternalInputSpecs:
    """Test external input specification discovery."""

    def test_get_specs_with_external_leaves(self):
        """External leaves should report their input specs."""
        ext1 = MockExternalRetargeter("sim_state")
        ext2 = MockExternalRetargeter("robot_state")
        pipeline = MockPipeline(leaf_nodes=[ext1, ext2])

        config = make_config(pipeline)
        session = TeleopSession(config)

        specs = session.get_external_input_specs()

        assert "sim_state" in specs
        assert "robot_state" in specs
        # Each spec should contain the input_spec of that retargeter
        assert "external_data" in specs["sim_state"]

    def test_get_specs_empty_when_all_deviceio(self):
        """No external specs when all leaves are DeviceIO sources."""
        head = MockHeadSource()
        pipeline = MockPipeline(leaf_nodes=[head])

        config = make_config(pipeline)
        session = TeleopSession(config)

        specs = session.get_external_input_specs()
        assert specs == {}

    def test_has_external_inputs_true(self):
        """has_external_inputs returns True when external leaves exist."""
        ext = MockExternalRetargeter("ext")
        pipeline = MockPipeline(leaf_nodes=[ext])

        config = make_config(pipeline)
        session = TeleopSession(config)

        assert session.has_external_inputs() is True

    def test_has_external_inputs_false(self):
        """has_external_inputs returns False when no external leaves."""
        head = MockHeadSource()
        pipeline = MockPipeline(leaf_nodes=[head])

        config = make_config(pipeline)
        session = TeleopSession(config)

        assert session.has_external_inputs() is False

    def test_has_external_inputs_empty(self):
        """has_external_inputs returns False when pipeline has no leaves."""
        pipeline = MockPipeline(leaf_nodes=[])

        config = make_config(pipeline)
        session = TeleopSession(config)

        assert session.has_external_inputs() is False

    def test_has_external_inputs_false_when_only_empty_input_spec_leaves(self):
        """has_external_inputs returns False when only non-DeviceIO leaves have empty input_spec()."""
        constant_source = MockEmptyInputRetargeter("constant")
        pipeline = MockPipeline(leaf_nodes=[constant_source])

        config = make_config(pipeline)
        session = TeleopSession(config)

        assert session.has_external_inputs() is False
        assert session.get_external_input_specs() == {}

    def test_step_succeeds_without_external_inputs_for_empty_input_spec_leaf(self):
        """step() succeeds without external_inputs when pipeline has only empty input_spec() leaves."""
        constant_source = MockEmptyInputRetargeter("constant")
        pipeline = MockPipeline(
            leaf_nodes=[constant_source], call_result={"constant": None}
        )

        config = make_config(pipeline)
        with mock_session_dependencies():
            session = TeleopSession(config)
            with session:
                result = session.step()
                assert result == {"constant": None}


class TestValidateExternalInputs:
    """Test the _validate_external_inputs method."""

    def test_no_external_leaves_no_inputs(self):
        """No validation error when there are no external leaves."""
        head = MockHeadSource()
        pipeline = MockPipeline(leaf_nodes=[head])

        config = make_config(pipeline)
        session = TeleopSession(config)

        # Should not raise
        session._validate_external_inputs(None)

    def test_no_external_leaves_with_inputs(self):
        """No validation error even if inputs provided when none needed."""
        head = MockHeadSource()
        pipeline = MockPipeline(leaf_nodes=[head])

        config = make_config(pipeline)
        session = TeleopSession(config)

        # Should not raise (extra inputs are silently ignored by validation)
        session._validate_external_inputs({"extra": {}})

    def test_external_leaves_missing_all_inputs(self):
        """ValueError when external leaves exist but no inputs provided."""
        ext = MockExternalRetargeter("sim_state")
        pipeline = MockPipeline(leaf_nodes=[ext])

        config = make_config(pipeline)
        session = TeleopSession(config)

        with pytest.raises(ValueError, match=r"external.*non-DeviceIO"):
            session._validate_external_inputs(None)

    def test_external_leaves_missing_some_inputs(self):
        """ValueError when some external inputs are missing."""
        ext1 = MockExternalRetargeter("sim_state")
        ext2 = MockExternalRetargeter("robot_state")
        pipeline = MockPipeline(leaf_nodes=[ext1, ext2])

        config = make_config(pipeline)
        session = TeleopSession(config)

        # Only provide one of the two required inputs
        with pytest.raises(ValueError, match="Missing external inputs"):
            session._validate_external_inputs({"sim_state": {}})

    def test_external_leaves_all_inputs_provided(self):
        """No error when all required external inputs are provided."""
        ext1 = MockExternalRetargeter("sim_state")
        ext2 = MockExternalRetargeter("robot_state")
        pipeline = MockPipeline(leaf_nodes=[ext1, ext2])

        config = make_config(pipeline)
        session = TeleopSession(config)

        # Should not raise
        session._validate_external_inputs(
            {
                "sim_state": {"external_data": MagicMock()},
                "robot_state": {"external_data": MagicMock()},
            }
        )

    def test_external_leaves_extra_inputs_allowed(self):
        """Extra inputs beyond what's required should not cause errors."""
        ext = MockExternalRetargeter("sim_state")
        pipeline = MockPipeline(leaf_nodes=[ext])

        config = make_config(pipeline)
        session = TeleopSession(config)

        # Provide required plus extra
        session._validate_external_inputs(
            {
                "sim_state": {"external_data": MagicMock()},
                "bonus_data": {"something": MagicMock()},
            }
        )


class TestSessionLifecycle:
    """Test __enter__ and __exit__ session lifecycle."""

    def test_enter_creates_sessions(self):
        """__enter__ should create OpenXR and DeviceIO sessions."""
        head = MockHeadSource()
        pipeline = MockPipeline(leaf_nodes=[head])
        mock_oxr = MockOpenXRSession()
        mock_dio = MockDeviceIOSession()

        config = make_config(pipeline)
        with mock_session_dependencies(mock_oxr=mock_oxr, mock_dio=mock_dio):
            session = TeleopSession(config)
            with session as s:
                assert s is session
                assert s.oxr_session is mock_oxr
                assert s.deviceio_session is mock_dio
                assert s.frame_count == 0

    def test_enter_rejects_nested_active_session(self):
        """A session should be reusable after exit but not re-enterable while active."""
        config = make_config(MockPipeline(leaf_nodes=[]))

        with mock_session_dependencies():
            session = TeleopSession(config)
            with session:
                with pytest.raises(RuntimeError, match="already active"):
                    session.__enter__()

            with session:
                pass

    @pytest.mark.parametrize(
        "failure_stage", ["openxr", "deviceio", "plugin_1", "plugin_2"]
    )
    def test_enter_failure_exits_previously_acquired_resources_once(
        self, tmp_path, failure_stage
    ):
        """A failed __enter__ should propagate after unwinding resources in reverse order."""
        events = []
        failures_remaining = {failure_stage}

        class TrackedContext:
            def __init__(self, name):
                self.name = name

            def get_handles(self):
                return MockOpenXRHandles()

            def __enter__(self):
                events.append(("enter", self.name))
                if self.name in failures_remaining:
                    failures_remaining.remove(self.name)
                    raise RuntimeError(f"{self.name} failed")
                return self

            def __exit__(self, *args):
                events.append(("exit", self.name))
                return True

        plugin_contexts = {
            name: TrackedContext(name) for name in ("plugin_1", "plugin_2")
        }

        class TrackedPluginManager:
            def get_plugin_names(self):
                return list(plugin_contexts)

            def start(self, plugin_name, *args):
                return plugin_contexts[plugin_name]

        plugins = [
            PluginConfig(
                plugin_name=name,
                plugin_root_id="/root",
                search_paths=[tmp_path],
            )
            for name in plugin_contexts
        ]
        config = make_config(MockPipeline(leaf_nodes=[]), plugins=plugins)

        with mock_session_dependencies(
            mock_oxr=TrackedContext("openxr"),
            mock_dio=TrackedContext("deviceio"),
            mock_pm=TrackedPluginManager(),
        ):
            session = TeleopSession(config)
            with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
                session.__enter__()

        resource_order = ["openxr", "deviceio", "plugin_1", "plugin_2"]
        failure_index = resource_order.index(failure_stage)
        expected_events = [
            ("enter", name) for name in resource_order[: failure_index + 1]
        ]
        expected_events.extend(
            ("exit", name) for name in reversed(resource_order[:failure_index])
        )
        assert events == expected_events
        assert session.oxr_session is None

        events.clear()
        with mock_session_dependencies(
            mock_oxr=TrackedContext("openxr"),
            mock_dio=TrackedContext("deviceio"),
            mock_pm=TrackedPluginManager(),
        ):
            with session:
                pass

        expected_events = [("enter", name) for name in resource_order]
        expected_events.extend(("exit", name) for name in reversed(resource_order))
        assert events == expected_events

    def test_enter_preserves_setup_error_if_rollback_fails(self, caplog):
        """A resource cleanup error should not replace the setup error."""
        session = TeleopSession(make_config(MockPipeline(leaf_nodes=[])))

        class FailingExit:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                raise RuntimeError("rollback failed")

        def fail_setup(stack):
            stack.enter_context(FailingExit())
            raise ValueError("setup failed")

        with (
            caplog.at_level(logging.ERROR),
            patch.object(session, "_enter_resources", side_effect=fail_setup),
            pytest.raises(ValueError, match="setup failed"),
        ):
            session.__enter__()

        assert "Failed to roll back TeleopSession setup" in caplog.text

        with mock_session_dependencies():
            with session:
                pass

    def test_enter_initializes_runtime_state(self):
        """__enter__ should reset frame count and record start time."""
        pipeline = MockPipeline(leaf_nodes=[])

        config = make_config(pipeline)
        with mock_session_dependencies():
            session = TeleopSession(config)
            before = time.time()
            with session as s:
                after = time.time()
                assert s.frame_count == 0
                assert before <= s.start_time <= after

    def test_exit_is_noop_when_session_is_not_active(self):
        """A direct __exit__ after context exit should not repeat resource cleanup."""
        pipeline = MockPipeline(leaf_nodes=[])

        class CountingOpenXRSession(MockOpenXRSession):
            def __init__(self):
                super().__init__()
                self.exit_count = 0

            def __exit__(self, *args):
                self.exit_count += 1

        mock_oxr = CountingOpenXRSession()
        config = make_config(pipeline)
        with mock_session_dependencies(mock_oxr=mock_oxr):
            session = TeleopSession(config)
            with session:
                pass

            assert mock_oxr.exit_count == 1
            assert session.__exit__(None, None, None) is False
            assert mock_oxr.exit_count == 1

    def test_exit_clears_oxr_session(self):
        """After the with-block exits, the public oxr_session property honors its None contract."""
        pipeline = MockPipeline(leaf_nodes=[])

        config = make_config(pipeline)
        with mock_session_dependencies():
            session = TeleopSession(config)
            with session as s:
                assert s.oxr_session is not None

        assert session.oxr_session is None

    def test_exit_clears_oxr_session_even_if_cleanup_raises(self):
        """oxr_session returns None post-exit even when a managed context's cleanup raises."""

        class RaisingDeviceIOSession(MockDeviceIOSession):
            def __exit__(self, *args):
                raise RuntimeError("cleanup boom")

        pipeline = MockPipeline(leaf_nodes=[])
        config = make_config(pipeline)
        with mock_session_dependencies(mock_dio=RaisingDeviceIOSession()):
            session = TeleopSession(config)
            session.__enter__()
            assert session.oxr_session is not None

            # The ExitStack unwind raises, but exception propagation is preserved
            # and the reference is still cleared via the finally block.
            with pytest.raises(RuntimeError, match="cleanup boom"):
                session.__exit__(None, None, None)

            assert session.oxr_session is None

    def test_context_manager_protocol(self):
        """TeleopSession works correctly as a context manager."""
        head = MockHeadSource()
        pipeline = MockPipeline(leaf_nodes=[head])

        config = make_config(pipeline)
        entered = False
        with mock_session_dependencies():
            session = TeleopSession(config)
            with session as s:
                entered = True
                assert s is session

        assert entered is True

    def test_enter_collects_trackers_from_sources(self):
        """__enter__ should gather trackers from all discovered sources."""
        head = MockHeadSource()
        controllers = MockControllersSource()
        pipeline = MockPipeline(leaf_nodes=[head, controllers])

        collected_trackers = []
        config = make_config(pipeline)
        with mock_session_dependencies(collected_trackers=collected_trackers):
            session = TeleopSession(config)
            with session:
                pass

        # Should have collected trackers from both sources
        assert len(collected_trackers) == 2

    def test_enter_includes_manual_trackers(self):
        """__enter__ should include manual trackers of new types, dedup same types."""
        head = MockHeadSource()
        pipeline = MockPipeline(leaf_nodes=[head])
        # Manual tracker of a *different* type should be included
        manual_tracker = ControllerTracker()

        collected_trackers = []
        config = make_config(pipeline, trackers=[manual_tracker])
        with mock_session_dependencies(collected_trackers=collected_trackers):
            session = TeleopSession(config)
            with session:
                pass

        # 1 HeadTracker from source + 1 ControllerTracker manual = 2 unique types
        assert len(collected_trackers) == 2
        assert manual_tracker in collected_trackers

    def test_enter_does_not_deduplicate_same_type_trackers(self):
        """__enter__ should pass all trackers including duplicates of the same type."""
        head = MockHeadSource()
        pipeline = MockPipeline(leaf_nodes=[head])
        duplicate_tracker = HeadTracker()

        collected_trackers = []
        config = make_config(pipeline, trackers=[duplicate_tracker])
        with mock_session_dependencies(collected_trackers=collected_trackers):
            session = TeleopSession(config)
            with session:
                pass

        # Both trackers are passed through: source tracker + manual tracker
        assert len(collected_trackers) == 2
        assert collected_trackers[0] is head.get_tracker()
        assert collected_trackers[1] is duplicate_tracker


class TestPluginInitialization:
    """Test plugin initialization in __enter__."""

    def test_plugins_initialized(self, tmp_path):
        """Required plugins with valid paths are started."""
        pipeline = MockPipeline(leaf_nodes=[])

        mock_pm = MockPluginManager(plugin_names=["test_plugin"])

        plugin_config = PluginConfig(
            plugin_name="test_plugin",
            plugin_root_id="/root",
            search_paths=[tmp_path],
            enabled=True,
            required=True,
        )

        config = make_config(pipeline, plugins=[plugin_config])
        with mock_session_dependencies(mock_pm=mock_pm):
            session = TeleopSession(config)
            with session:
                assert len(session.plugin_managers) == 1
                assert len(session.plugin_contexts) == 1

    def test_disabled_plugin_skipped(self, tmp_path):
        """Disabled plugins are not started even when marked required."""
        pipeline = MockPipeline(leaf_nodes=[])

        mock_pm = MockPluginManager(plugin_names=["test_plugin"])

        plugin_config = PluginConfig(
            plugin_name="test_plugin",
            plugin_root_id="/root",
            search_paths=[tmp_path],
            enabled=False,
            required=True,
        )

        config = make_config(pipeline, plugins=[plugin_config])
        with mock_session_dependencies(mock_pm=mock_pm):
            session = TeleopSession(config)
            with session:
                assert len(session.plugin_managers) == 0
                assert len(session.plugin_contexts) == 0

    def test_missing_plugin_skipped(self, tmp_path):
        """Plugins not found in search paths are skipped."""
        pipeline = MockPipeline(leaf_nodes=[])

        # Plugin manager doesn't know about "nonexistent_plugin"
        mock_pm = MockPluginManager(plugin_names=["other_plugin"])

        plugin_config = PluginConfig(
            plugin_name="nonexistent_plugin",
            plugin_root_id="/root",
            search_paths=[tmp_path],
            enabled=True,
        )

        config = make_config(pipeline, plugins=[plugin_config])
        with mock_session_dependencies(mock_pm=mock_pm):
            session = TeleopSession(config)
            with session:
                # Manager was created but plugin not found, so no context started
                assert len(session.plugin_managers) == 1
                assert len(session.plugin_contexts) == 0

    def test_required_missing_plugin_raises(self, tmp_path):
        """Required plugins must be discovered in an existing search path."""
        pipeline = MockPipeline(leaf_nodes=[])
        mock_pm = MockPluginManager(plugin_names=["z_plugin", "a_plugin"])
        plugin_config = PluginConfig(
            plugin_name="required_plugin",
            plugin_root_id="/root",
            search_paths=[tmp_path],
            required=True,
        )

        config = make_config(pipeline, plugins=[plugin_config])
        with mock_session_dependencies(mock_pm=mock_pm):
            session = TeleopSession(config)
            with pytest.raises(RuntimeError) as exc_info:
                session.__enter__()

        message = str(exc_info.value)
        assert "Required plugin 'required_plugin' was not discovered" in message
        assert f"search paths ['{tmp_path}']" in message
        assert "Discovered plugins: ['a_plugin', 'z_plugin']" in message
        assert session.oxr_session is None
        assert session._in_context is False

    def test_invalid_search_paths_skipped(self):
        """Plugins with no valid search paths are skipped entirely."""
        pipeline = MockPipeline(leaf_nodes=[])

        mock_pm = MockPluginManager(plugin_names=["test_plugin"])

        plugin_config = PluginConfig(
            plugin_name="test_plugin",
            plugin_root_id="/root",
            search_paths=[Path("/nonexistent/path/that/does/not/exist")],
            enabled=True,
        )

        config = make_config(pipeline, plugins=[plugin_config])
        with mock_session_dependencies(mock_pm=mock_pm):
            session = TeleopSession(config)
            with session:
                # Skipped because no valid search paths
                assert len(session.plugin_managers) == 0
                assert len(session.plugin_contexts) == 0

    def test_required_invalid_search_paths_raise(self, tmp_path):
        """Required plugins must have at least one existing search path."""
        pipeline = MockPipeline(leaf_nodes=[])
        missing_path = tmp_path / "missing"
        plugin_config = PluginConfig(
            plugin_name="required_plugin",
            plugin_root_id="/root",
            search_paths=[missing_path],
            required=True,
        )

        config = make_config(pipeline, plugins=[plugin_config])
        with mock_session_dependencies():
            session = TeleopSession(config)
            with pytest.raises(RuntimeError) as exc_info:
                session.__enter__()

        message = str(exc_info.value)
        assert (
            "Required plugin 'required_plugin' has no existing search directories"
            in message
        )
        assert f"Configured search paths: ['{missing_path}']" in message
        assert session.plugin_managers == []
        assert session.oxr_session is None
        assert session._in_context is False

    def test_required_regular_file_search_path_raises(self, tmp_path):
        """Required plugin search paths must be directories, not regular files."""
        pipeline = MockPipeline(leaf_nodes=[])
        file_path = tmp_path / "not-a-directory"
        file_path.write_text("")
        plugin_config = PluginConfig(
            plugin_name="required_plugin",
            plugin_root_id="/root",
            search_paths=[file_path],
            required=True,
        )

        config = make_config(pipeline, plugins=[plugin_config])
        with mock_session_dependencies():
            session = TeleopSession(config)
            with pytest.raises(RuntimeError) as exc_info:
                session.__enter__()

        message = str(exc_info.value)
        assert (
            "Required plugin 'required_plugin' has no existing search directories"
            in message
        )
        assert f"Configured search paths: ['{file_path}']" in message
        assert session.plugin_managers == []

    def test_no_plugins_configured(self):
        """Session works fine with no plugins configured."""
        pipeline = MockPipeline(leaf_nodes=[])

        config = make_config(pipeline, plugins=[])
        with mock_session_dependencies():
            session = TeleopSession(config)
            with session:
                assert len(session.plugin_managers) == 0
                assert len(session.plugin_contexts) == 0

    def test_plugin_args_passed_through(self, tmp_path):
        """Plugin args from config are forwarded to manager.start()."""
        pipeline = MockPipeline(leaf_nodes=[])

        mock_pm = MockPluginManager(plugin_names=["test_plugin"])

        plugin_config = PluginConfig(
            plugin_name="test_plugin",
            plugin_root_id="/root",
            search_paths=[tmp_path],
            enabled=True,
            plugin_args=["--flag", "value"],
        )

        config = make_config(pipeline, plugins=[plugin_config])
        with mock_session_dependencies(mock_pm=mock_pm):
            session = TeleopSession(config)
            with session:
                assert len(mock_pm.start_calls) == 1
                call = mock_pm.start_calls[0]
                assert call["plugin_name"] == "test_plugin"
                assert call["plugin_root_id"] == "/root"
                assert call["plugin_args"] == ["--flag", "value"]

    def test_plugin_args_default_empty(self, tmp_path):
        """Plugin args default to an empty list when not specified."""
        pipeline = MockPipeline(leaf_nodes=[])

        mock_pm = MockPluginManager(plugin_names=["test_plugin"])

        plugin_config = PluginConfig(
            plugin_name="test_plugin",
            plugin_root_id="/root",
            search_paths=[tmp_path],
            enabled=True,
        )

        config = make_config(pipeline, plugins=[plugin_config])
        with mock_session_dependencies(mock_pm=mock_pm):
            session = TeleopSession(config)
            with session:
                assert len(mock_pm.start_calls) == 1
                assert mock_pm.start_calls[0]["plugin_args"] == []


class TestStatusMonitoringIntegration:
    """Test monitored-plugin inventory, refresh, and provider failure integration."""

    def test_spacemouse_plugin_is_monitored(self, tmp_path):
        manager = MockPluginManager(plugin_names=["spacemouse"])
        manager.get_plugin_info = MagicMock(
            return_value=SimpleNamespace(
                name="spacemouse",
                devices=(
                    SimpleNamespace(
                        path="/spacemouse", type="spacemouse", description="SpaceMouse"
                    ),
                ),
            )
        )
        config = make_config(
            MockPipeline(leaf_nodes=[]),
            plugins=[
                PluginConfig(
                    plugin_name="spacemouse",
                    plugin_root_id="spacemouse",
                    search_paths=[tmp_path],
                )
            ],
        )

        with (
            patch(
                "isaaccapture.deviceio_trackers.PluginDeviceStatusTracker"
            ) as tracker_cls,
            mock_session_dependencies(mock_pm=manager, collected_trackers=[]),
        ):
            with TeleopSession(config) as session:
                tracker_cls.assert_called_once_with("spacemouse/device_status")
                assert session.get_device_status("spacemouse/spacemouse") is not None

    def test_only_manus_is_monitored_and_plugin_launch_order_is_preserved(
        self, tmp_path
    ):
        class StatusTracker:
            def __init__(self):
                self.poll_count = 0

            def get_device_status_snapshot(self, _session):
                self.poll_count += 1

        class CountingContext(MockPluginContext):
            def __init__(self):
                super().__init__()
                self.process_poll_count = 0

            def get_process_snapshot(self):
                self.process_poll_count += 1
                return super().get_process_snapshot()

        manager = MockPluginManager(plugin_names=["legacy_plugin", "manus_hand_plugin"])
        legacy_context = CountingContext()
        manus_context = CountingContext()
        manager._contexts["legacy_plugin"] = legacy_context
        manager._contexts["manus_hand_plugin"] = manus_context
        manager.get_plugin_info = MagicMock(
            return_value=SimpleNamespace(
                name="Manus",
                devices=(
                    SimpleNamespace(
                        path="/hand/left", type="hand", description="Left hand"
                    ),
                ),
            )
        )
        tracker = StatusTracker()
        collected_trackers = []
        config = make_config(
            MockPipeline(leaf_nodes=[]),
            plugins=[
                PluginConfig(
                    plugin_name="legacy_plugin",
                    plugin_root_id="/legacy",
                    search_paths=[tmp_path],
                ),
                PluginConfig(
                    plugin_name="manus_hand_plugin",
                    plugin_root_id="/manus",
                    search_paths=[tmp_path],
                ),
            ],
        )

        with (
            patch(
                "isaaccapture.deviceio_trackers.PluginDeviceStatusTracker",
                return_value=tracker,
            ) as tracker_cls,
            mock_session_dependencies(
                mock_pm=manager,
                collected_trackers=collected_trackers,
            ),
        ):
            session = TeleopSession(config)
            with session:
                tracker_cls.assert_called_once_with("/manus/device_status")
                manager.get_plugin_info.assert_called_once_with("manus_hand_plugin")
                assert collected_trackers == [tracker]
                assert [call["plugin_name"] for call in manager.start_calls] == [
                    "legacy_plugin",
                    "manus_hand_plugin",
                ]
                assert len(session.plugin_managers) == 2
                assert len(session.plugin_contexts) == 2

                snapshot = session.get_status()
                assert {provider.id for provider in snapshot.providers} == {
                    "openxr/runtime",
                    "plugin//manus",
                }
                assert {device.id for device in snapshot.devices} == {
                    "openxr/headset",
                    "/manus/hand/left",
                }
                assert session.get_provider_status("plugin//legacy") is None
                assert session.get_device_status("/legacy/device") is None

                cached = session.get_status()
                assert manus_context.process_poll_count == 0
                assert legacy_context.process_poll_count == 0
                assert tracker.poll_count == 0

                session.step()
                assert session.get_status() is not cached
                assert manus_context.process_poll_count == 1
                assert legacy_context.process_poll_count == 0
                assert tracker.poll_count == 1

    def test_manus_process_failure_is_cached_before_health_error(self, tmp_path):
        class CrashedContext(MockPluginContext):
            def check_health(self):
                raise RuntimeError("plugin crashed")

            def get_process_snapshot(self):
                return SimpleNamespace(
                    state="EXITED",
                    reason="NONZERO_EXIT",
                    pid=123,
                    exit_code=7,
                    term_signal=None,
                    error_code=None,
                    error="unexpected exit",
                )

        context = CrashedContext()
        manager = MockPluginManager(plugin_names=["manus_hand_plugin"])
        manager._contexts["manus_hand_plugin"] = context
        manager.get_plugin_info = MagicMock(
            return_value=SimpleNamespace(
                name="Manus",
                devices=(
                    SimpleNamespace(
                        path="/hand/left", type="hand", description="Left hand"
                    ),
                ),
            )
        )
        config = make_config(
            MockPipeline(leaf_nodes=[]),
            plugins=[
                PluginConfig(
                    plugin_name="manus_hand_plugin",
                    plugin_root_id="/manus",
                    search_paths=[tmp_path],
                )
            ],
        )

        with (
            patch("isaaccapture.deviceio_trackers.PluginDeviceStatusTracker"),
            mock_session_dependencies(mock_pm=manager),
        ):
            session = TeleopSession(config)
            with session:
                with pytest.raises(RuntimeError, match="plugin crashed"):
                    session.step()

                provider = session.get_provider_status("plugin//manus")
                assert provider.status == teleop_session_manager.ProviderState.FAILED
                assert (
                    provider.reason
                    == teleop_session_manager.StatusReason.PROCESS_EXITED
                )
                assert provider.exit_code == 7

    def test_deviceio_failure_marks_openxr_and_manus_failed(self, tmp_path):
        class FailingDeviceIOSession(MockDeviceIOSession):
            def update(self):
                raise RuntimeError("runtime update failed")

        manager = MockPluginManager(plugin_names=["manus_hand_plugin"])
        manager.get_plugin_info = MagicMock(
            return_value=SimpleNamespace(name="Manus", devices=())
        )
        config = make_config(
            MockPipeline(leaf_nodes=[]),
            plugins=[
                PluginConfig(
                    plugin_name="manus_hand_plugin",
                    plugin_root_id="/manus",
                    search_paths=[tmp_path],
                )
            ],
        )

        with (
            patch("isaaccapture.deviceio_trackers.PluginDeviceStatusTracker"),
            mock_session_dependencies(
                mock_dio=FailingDeviceIOSession(),
                mock_pm=manager,
            ),
        ):
            session = TeleopSession(config)
            with session:
                with pytest.raises(RuntimeError, match="runtime update failed"):
                    session.step()

                assert {
                    provider.status for provider in session.get_status().providers
                } == {teleop_session_manager.ProviderState.FAILED}
                assert {
                    provider.reason for provider in session.get_status().providers
                } == {teleop_session_manager.StatusReason.RUNTIME_UPDATE_FAILED}


class TestStep:
    """Test the step() method execution flow."""

    def test_step_calls_deviceio_update(self):
        """step() should call deviceio_session.update()."""
        head = MockHeadSource()
        mock_dio = MockDeviceIOSession()
        pipeline = MockPipeline(leaf_nodes=[head])

        config = make_config(pipeline)
        with mock_session_dependencies(mock_dio=mock_dio):
            session = TeleopSession(config)
            with session:
                session.step()
                assert mock_dio.update_count == 1

                session.step()
                assert mock_dio.update_count == 2

    def test_step_calls_pipeline(self):
        """step() should call the pipeline with collected inputs."""
        head = MockHeadSource()
        expected_result = {"output": "data"}
        pipeline = MockPipeline(leaf_nodes=[head], call_result=expected_result)

        config = make_config(pipeline)
        with mock_session_dependencies():
            session = TeleopSession(config)
            with session:
                result = session.step()
                assert result == expected_result
                assert pipeline.last_inputs is not None

    def test_step_increments_frame_count(self):
        """step() should increment frame_count each call."""
        pipeline = MockPipeline(leaf_nodes=[])

        config = make_config(pipeline)
        with mock_session_dependencies():
            session = TeleopSession(config)
            with session:
                assert session.frame_count == 0
                session.step()
                assert session.frame_count == 1
                session.step()
                assert session.frame_count == 2
                session.step()
                assert session.frame_count == 3

    def test_step_with_external_inputs(self):
        """step() should merge external inputs into pipeline inputs."""
        ext = MockExternalRetargeter("sim_state")
        pipeline = MockPipeline(leaf_nodes=[ext])

        config = make_config(pipeline)
        external_data = {"sim_state": {"external_data": MagicMock()}}
        with mock_session_dependencies():
            session = TeleopSession(config)
            with session:
                session.step(external_inputs=external_data)
                # The pipeline should receive the external inputs
                assert "sim_state" in pipeline.last_inputs

    def test_default_sync_step_filters_unused_external_inputs(self):
        """Default sync mode should ignore unused external leaves and per-leaf keys."""
        ext = MockExternalRetargeter("sim_state")
        pipeline = MockPipeline(leaf_nodes=[ext])

        config = TeleopSessionConfig(
            app_name="TestApp",
            pipeline=pipeline,
            verbose=False,
        )
        external_value = MagicMock()
        external_data = {
            "sim_state": {
                "external_data": external_value,
                "ignored_data": MagicMock(),
            },
            "bonus_data": {"external_data": MagicMock()},
        }
        with mock_session_dependencies():
            session = TeleopSession(config)
            with session:
                session.step(external_inputs=external_data)

                assert set(pipeline.last_inputs) == {"sim_state"}
                assert set(pipeline.last_inputs["sim_state"]) == {"external_data"}
                assert (
                    pipeline.last_inputs["sim_state"]["external_data"] is external_value
                )

    def test_step_with_mixed_sources(self):
        """step() with both DeviceIO sources and external inputs."""
        head = MockHeadSource()
        ext = MockExternalRetargeter("sim_state")
        pipeline = MockPipeline(leaf_nodes=[head, ext])

        config = make_config(pipeline)
        external_data = {"sim_state": {"external_data": MagicMock()}}
        with mock_session_dependencies():
            session = TeleopSession(config)
            with session:
                session.step(external_inputs=external_data)
                # Pipeline should have both DeviceIO and external inputs
                assert "head" in pipeline.last_inputs
                assert "sim_state" in pipeline.last_inputs

    def test_step_raises_on_missing_external_inputs(self):
        """step() should raise ValueError when external inputs are required but missing."""
        ext = MockExternalRetargeter("sim_state")
        pipeline = MockPipeline(leaf_nodes=[ext])

        config = make_config(pipeline)
        with mock_session_dependencies():
            session = TeleopSession(config)
            with session:
                with pytest.raises(ValueError, match=r"external.*non-DeviceIO"):
                    session.step()

    def test_step_checks_plugin_health_every_60_frames(self):
        """step() should check plugin health every 60 frames."""
        pipeline = MockPipeline(leaf_nodes=[])
        mock_ctx = MockPluginContext()

        config = make_config(pipeline)
        with mock_session_dependencies():
            session = TeleopSession(config)
            with session:
                # Manually add a mock plugin context
                session.plugin_contexts.append(mock_ctx)

                # Frame 0: should check (0 % 60 == 0)
                session.step()
                assert mock_ctx.health_check_count == 1

                # Frames 1-59: should not check
                for _ in range(59):
                    session.step()
                assert mock_ctx.health_check_count == 1

                # Frame 60: should check again
                session.step()
                assert mock_ctx.health_check_count == 2

    def test_step_no_external_inputs_when_none_needed(self):
        """step() without external_inputs works when no external leaves exist."""
        head = MockHeadSource()
        pipeline = MockPipeline(leaf_nodes=[head])

        config = make_config(pipeline)
        with mock_session_dependencies():
            session = TeleopSession(config)
            with session:
                # Should not raise
                session.step()
                assert session.frame_count == 1


class TestPipelinedRetargeting:
    """Tests for TeleopSession-owned pipelined async retarget execution."""

    def _pipelined_config(self, pipeline, **execution_kwargs):
        return make_config(
            pipeline,
            retargeting_execution=RetargetingExecutionConfig(
                mode=RetargetingExecutionMode.PIPELINED,
                **execution_kwargs,
            ),
        )

    def test_first_frame_runs_synchronously_and_returns_seed(self):
        pipeline = CountingPipeline()
        config = self._pipelined_config(pipeline)

        with mock_session_dependencies():
            with TeleopSession(config) as session:
                result = session.step()

                assert async_result_value(result) == 0.0
                assert session.last_step_info.ran_synchronously is True
                assert session.last_step_info.returned_frame_id == 0
                assert session.last_step_info.submitted_frame_id == 0
                assert session.last_step_info.returned_age_frames == 0

    def test_default_pipelined_returns_latest_completed_result(self):
        pipeline = BlockingSecondCountingPipeline()
        config = self._pipelined_config(pipeline)

        with mock_session_dependencies():
            with TeleopSession(config) as session:
                first = session.step()
                second = session.step()

                assert async_result_value(first) == 0.0
                assert async_result_value(second) == 0.0
                assert session.last_step_info.ran_synchronously is False
                assert session.last_step_info.returned_frame_id == 0
                assert session.last_step_info.submitted_frame_id == 1
                assert session.last_step_info.returned_age_frames == 1
                pipeline.release_second.set()

    def test_reset_frame_is_not_forced_current_but_returned_frame_is_consistent(self):
        pipeline = ContextEchoPipeline()
        config = self._pipelined_config(
            pipeline,
            pacing=DeadlinePacingConfig(
                frame_period_adaptation=0.01,
                safety_margin_s=0.0,
                startup_frame_period_s=0.25,
                startup_compute_cost_s=0.0,
            ),
        )

        with mock_session_dependencies():
            with TeleopSession(config) as session:
                assert (
                    async_result_value(
                        session.step(
                            execution_events=ExecutionEvents(
                                reset=False,
                                execution_state=ExecutionState.RUNNING,
                            )
                        )
                    )
                    == 0.0
                )
                result = session.step(
                    execution_events=ExecutionEvents(
                        reset=True,
                        execution_state=ExecutionState.RUNNING,
                    )
                )

                assert async_result_value(result) == 0.0
                assert session.last_step_info.ran_synchronously is False
                assert session.last_context.execution_events.reset is False

                assert pipeline.second_started.wait(timeout=1.0)
                pipeline.release_second.set()
                assert pipeline.second_done.wait(timeout=1.0)
                assert session._async_runner is not None
                assert session._async_runner.wait_for_frame(1, timeout_s=1.0)

                result = session.step(
                    execution_events=ExecutionEvents(
                        reset=False,
                        execution_state=ExecutionState.RUNNING,
                    )
                )

                assert async_result_value(result) == 1.0
                assert session.last_step_info.ran_synchronously is False
                assert session.last_step_info.returned_frame_id == 1
                assert session.last_step_info.submitted_frame_id == 2
                assert session.last_context.execution_events.reset is True

    def test_control_transition_frame_is_returned_with_its_context(self):
        pipeline = ContextEchoPipeline()
        config = self._pipelined_config(
            pipeline,
            pacing=DeadlinePacingConfig(
                frame_period_adaptation=0.01,
                safety_margin_s=0.0,
                startup_frame_period_s=0.25,
                startup_compute_cost_s=0.0,
            ),
        )

        with mock_session_dependencies():
            with TeleopSession(config) as session:
                assert (
                    async_result_value(
                        session.step(
                            execution_events=ExecutionEvents(
                                reset=False,
                                execution_state=ExecutionState.RUNNING,
                            )
                        )
                    )
                    == 0.0
                )
                result = session.step(
                    execution_events=ExecutionEvents(
                        reset=False,
                        execution_state=ExecutionState.PAUSED,
                    )
                )

                assert async_result_value(result) == 0.0
                assert session.last_step_info.ran_synchronously is False
                assert pipeline.second_started.wait(timeout=1.0)
                pipeline.release_second.set()
                assert pipeline.second_done.wait(timeout=1.0)
                assert session._async_runner is not None
                assert session._async_runner.wait_for_frame(1, timeout_s=1.0)

                result = session.step(
                    execution_events=ExecutionEvents(
                        reset=False,
                        execution_state=ExecutionState.RUNNING,
                    )
                )

                assert async_result_value(result) == 2.0
                assert session.last_step_info.ran_synchronously is False
                assert session.last_step_info.returned_frame_id == 1
                assert session.last_step_info.submitted_frame_id == 2
                assert (
                    session.last_context.execution_events.execution_state
                    == ExecutionState.PAUSED
                )

    def test_automatic_control_pipeline_frame_is_not_forced_current(self):
        pipeline = ControlEventEchoPipeline()
        control_pipeline = BlockingControlPipeline(second_reset=True)
        config = make_config(
            pipeline,
            teleop_control_pipeline=control_pipeline,
            retargeting_execution=RetargetingExecutionConfig(
                mode=RetargetingExecutionMode.PIPELINED,
                pacing=DeadlinePacingConfig(
                    frame_period_adaptation=0.01,
                    safety_margin_s=0.0,
                    startup_frame_period_s=0.25,
                    startup_compute_cost_s=0.0,
                ),
            ),
        )

        with mock_session_dependencies():
            with TeleopSession(config) as session:
                assert async_result_value(session.step()) == 0.0
                result = session.step()

                assert async_result_value(result) == 0.0
                assert session.last_step_info.ran_synchronously is False
                assert control_pipeline.second_started.wait(timeout=2.0)

                control_pipeline.release_second.set()
                assert session._async_runner is not None
                assert session._async_runner.wait_for_frame(1, timeout_s=2.0)
                result = session.step()

                assert async_result_value(result) == 1.0
                assert session.last_step_info.ran_synchronously is False
                assert session.last_step_info.returned_frame_id == 1
                assert session.last_step_info.submitted_frame_id == 2
                assert session.last_context.execution_events.reset is True

    def test_worker_inputs_are_snapshotted(self):
        pipeline = ExternalEchoPipeline()
        config = self._pipelined_config(pipeline)
        first_external = {"sim_state": make_external_scalar(1.0)}
        second_external = {"sim_state": make_external_scalar(2.0)}

        with mock_session_dependencies():
            with TeleopSession(config) as session:
                assert (
                    async_result_value(session.step(external_inputs=first_external))
                    == 1.0
                )
                previous_frame = session.step(external_inputs=second_external)
                assert async_result_value(previous_frame) == 1.0

                assert pipeline.second_started.wait(timeout=2.0)
                second_external["sim_state"]["external_data"][0] = 99.0
                pipeline.release_second.set()
                assert pipeline.second_done.wait(timeout=2.0)
                assert session._async_runner is not None
                assert session._async_runner.wait_for_frame(1, timeout_s=2.0)

                result = session.step(external_inputs=second_external)
                assert async_result_value(result) == 2.0

    def test_worker_context_is_snapshotted(self):
        pipeline = ContextEchoPipeline()
        config = self._pipelined_config(pipeline)
        reusable_events = ExecutionEvents(
            reset=False,
            execution_state=ExecutionState.RUNNING,
        )

        with mock_session_dependencies():
            with TeleopSession(config) as session:
                assert (
                    async_result_value(session.step(execution_events=reusable_events))
                    == 0.0
                )
                previous_frame = session.step(execution_events=reusable_events)
                assert async_result_value(previous_frame) == 0.0

                assert pipeline.second_started.wait(timeout=2.0)
                reusable_events.reset = True
                reusable_events.execution_state = ExecutionState.PAUSED
                pipeline.release_second.set()
                assert pipeline.second_done.wait(timeout=2.0)
                assert session._async_runner is not None
                assert session._async_runner.wait_for_frame(1, timeout_s=2.0)

                result = session.step(
                    execution_events=ExecutionEvents(
                        reset=False,
                        execution_state=ExecutionState.RUNNING,
                    )
                )
                assert async_result_value(result) == 0.0
                assert session.last_context.execution_events.reset is False
                assert (
                    session.last_context.execution_events.execution_state
                    == ExecutionState.RUNNING
                )

    def test_repeated_completed_outputs_are_snapshots(self):
        pipeline = CountingPipeline()
        config = self._pipelined_config(pipeline)

        with mock_session_dependencies():
            with TeleopSession(config) as session:
                first = session.step()
                first["result"][0] = 99.0

                second = session.step()
                assert async_result_value(second) == 0.0

    def test_pipelined_uncopyable_outputs_fail_with_clear_error(self):
        pipeline = UncopyableOutputPipeline()
        config = self._pipelined_config(pipeline)

        with mock_session_dependencies():
            with pytest.raises(TypeError, match="snapshot-copyable"):
                with TeleopSession(config) as session:
                    session.step()

    def test_sync_mode_allows_uncopyable_outputs(self):
        pipeline = UncopyableOutputPipeline()
        config = make_config(
            pipeline,
            retargeting_execution=RetargetingExecutionConfig(
                mode=RetargetingExecutionMode.SYNC
            ),
        )

        with mock_session_dependencies():
            with TeleopSession(config) as session:
                result = session.step()

                assert isinstance(result["result"][0], UncopyableValue)

    def test_sync_mode_allows_uncopyable_external_inputs(self):
        pipeline = AnyExternalRetargeter("opaque")
        config = make_config(
            pipeline,
            retargeting_execution=RetargetingExecutionConfig(
                mode=RetargetingExecutionMode.SYNC
            ),
        )

        with mock_session_dependencies():
            with TeleopSession(config) as session:
                result = session.step(
                    external_inputs={"opaque": make_opaque_external()}
                )

                assert async_result_value(result) == 1.0

    def test_pipelined_uncopyable_external_inputs_fail_before_worker(self):
        pipeline = AnyExternalRetargeter("opaque")
        config = self._pipelined_config(pipeline)

        with mock_session_dependencies():
            with pytest.raises(TypeError, match="snapshot-copyable"):
                with TeleopSession(config) as session:
                    session.step(external_inputs={"opaque": make_opaque_external()})

    def test_pipelined_ignores_unused_per_leaf_external_inputs_before_snapshot(self):
        pipeline = ExternalEchoPipeline()
        config = self._pipelined_config(pipeline)
        external = {"sim_state": make_external_scalar(3.0)}
        external["sim_state"]["ignored_opaque"] = make_opaque_external()["value"]

        with mock_session_dependencies():
            with TeleopSession(config) as session:
                result = session.step(external_inputs=external)

                assert async_result_value(result) == 3.0

    def test_worker_exception_surfaces_on_next_step(self):
        pipeline = CountingPipeline(fail_on_call=1)
        config = self._pipelined_config(pipeline)

        with mock_session_dependencies():
            with TeleopSession(config) as session:
                assert async_result_value(session.step()) == 0.0
                assert async_result_value(session.step()) == 0.0
                assert pipeline.failed.wait(timeout=2.0)

                with pytest.raises(RuntimeError, match="Async retarget worker failed"):
                    session.step()
                assert session.last_step_info.worker_exception is not None

    def test_worker_exception_surfaces_on_context_exit(self):
        pipeline = CountingPipeline(fail_on_call=1)
        config = self._pipelined_config(pipeline)

        with mock_session_dependencies():
            with pytest.raises(AsyncRetargetWorkerError):
                with TeleopSession(config) as session:
                    assert async_result_value(session.step()) == 0.0
                    assert async_result_value(session.step()) == 0.0
                    assert pipeline.failed.wait(timeout=2.0)

    def test_worker_exception_during_user_exception_is_logged(self, caplog):
        pipeline = CountingPipeline(fail_on_call=1)
        config = self._pipelined_config(pipeline)
        caplog.set_level(
            logging.ERROR,
            logger="isaaccapture.teleop_session_manager.teleop_session",
        )

        with mock_session_dependencies():
            with pytest.raises(ValueError, match="user boom"):
                with TeleopSession(config) as session:
                    assert async_result_value(session.step()) == 0.0
                    assert async_result_value(session.step()) == 0.0
                    assert pipeline.failed.wait(timeout=2.0)
                    raise ValueError("user boom")

        assert (
            "Async retarget worker failed during TeleopSession cleanup" in caplog.text
        )

    def test_first_frame_worker_exception_surfaces(self):
        pipeline = CountingPipeline(fail_on_call=0)
        config = self._pipelined_config(pipeline)

        with mock_session_dependencies():
            with pytest.raises(RuntimeError, match="pipeline boom"):
                with TeleopSession(config) as session:
                    session.step()

    def test_app_thread_runtime_error_is_not_labeled_worker_exception(self):
        source = FailingPollSource("failing", tracker=object())
        pipeline = MockPipeline(leaf_nodes=[source])
        config = self._pipelined_config(pipeline)

        with mock_session_dependencies():
            with TeleopSession(config) as session:
                with pytest.raises(RuntimeError, match="poll failed"):
                    session.step()
                assert session.last_step_info.worker_exception is None

    def test_worker_never_executes_graph_concurrently(self):
        pipeline = CountingPipeline(sleep_s=0.01)
        config = self._pipelined_config(pipeline)

        with mock_session_dependencies():
            with TeleopSession(config) as session:
                for _ in range(8):
                    session.step()

        assert pipeline.max_active == 1

    def test_frame_deadline_miss_marks_outputs_more_than_one_frame_old(self):
        pipeline = BlockingSecondCountingPipeline()
        config = self._pipelined_config(pipeline)

        with mock_session_dependencies():
            with TeleopSession(config) as session:
                assert async_result_value(session.step()) == 0.0
                session.step()
                assert pipeline.second_started.wait(timeout=1.0)

                try:
                    result = session.step()

                    assert async_result_value(result) == 0.0
                    assert session.last_step_info.returned_age_frames == 2
                    assert session.last_step_info.frame_deadline_miss is True
                finally:
                    pipeline.release_second.set()

    def test_deadline_pacing_uses_margin_and_cost_estimate(self):
        pacing = DeadlinePacingConfig(
            frame_period_adaptation=1.0,
            compute_cost_adaptation=1.0,
            spike_guard_window=4,
            spike_guard_percentile=0.90,
            safety_margin_s=0.015,
            startup_frame_period_s=0.050,
            startup_compute_cost_s=0.010,
        )

        sleep_s = pacing.compute_delay_s(
            submitted_time_s=100.0,
            now_s=100.020,
            submission_count=2,
            submit_period_s=0.050,
            compute_duration_s=0.010,
            compute_duration_samples=[0.004, 0.006, 0.012],
        )

        assert sleep_s == pytest.approx(0.003)

    def test_deadline_pacing_uses_seed_for_first_worker_request(self):
        pipeline = FrameIdPipeline()
        config = self._pipelined_config(
            pipeline,
            pacing=DeadlinePacingConfig(
                frame_period_adaptation=0.01,
                safety_margin_s=0.0,
                startup_frame_period_s=0.08,
                startup_compute_cost_s=0.0,
            ),
        )
        runner = make_async_runner(pipeline, config)
        now_s = time.monotonic()
        runner.publish_seed(
            RetargetFrame(
                frame_id=0,
                outputs=make_async_result(0.0),
                context=ComputeContext(
                    graph_time=GraphTime(sim_time_ns=0, real_time_ns=0),
                    execution_events=ExecutionEvents(
                        reset=False,
                        execution_state=ExecutionState.RUNNING,
                    ),
                ),
                submitted_time_s=now_s,
                started_time_s=now_s,
                completed_time_s=now_s,
                compute_duration_s=0.0,
            )
        )
        runner.start()
        try:
            assert (
                runner.submit(make_step_request(1, submitted_time_s=now_s + 0.001)) == 0
            )
            time.sleep(0.02)

            assert pipeline.executed_frame_ids == []
            latest = runner.latest()
            assert latest is not None
            assert latest.frame_id == 0

            frame = runner.wait_for_frame(1, timeout_s=1.0)
            assert frame is not None
            assert frame.frame_id == 1
        finally:
            runner.stop(timeout_s=1.0)

    def test_deadline_paced_unstarted_request_is_replaced_by_newer_submission(self):
        pipeline = FrameIdPipeline()
        config = self._pipelined_config(
            pipeline,
            pacing=DeadlinePacingConfig(
                frame_period_adaptation=0.01,
                safety_margin_s=0.0,
                startup_frame_period_s=0.25,
                startup_compute_cost_s=0.0,
            ),
        )
        runner = make_async_runner(pipeline, config)
        runner._submission_count = 1
        runner.start()
        try:
            assert runner.submit(make_step_request(1)) == 0
            time.sleep(0.05)
            assert runner.submit(make_step_request(2)) == 1

            frame = runner.wait_for_frame(2, timeout_s=1.0)

            assert frame is not None
            assert frame.frame_id == 2
            assert pipeline.executed_frame_ids == [2]
            assert runner.dropped_submissions == 1
        finally:
            runner.stop(timeout_s=1.0)

    def test_deadline_pacing_submit_period_estimator_updates(self):
        pipeline = CountingPipeline()
        config = self._pipelined_config(
            pipeline,
            pacing=DeadlinePacingConfig(frame_period_adaptation=1.0),
        )
        runner = make_async_runner(pipeline, config)

        with runner._cond:
            runner._record_submission_locked(10.0)
            runner._record_submission_locked(10.0005)
            assert runner._submit_period_s == pytest.approx(0.0005)

            runner._record_submission_locked(11.5005)
            assert runner._submit_period_s == pytest.approx(1.5)

    def test_published_frame_is_snapshotted_before_reused_output_mutates(self):
        pipeline = ReusingOutputPipeline()
        config = self._pipelined_config(pipeline)
        runner = make_async_runner(pipeline, config)
        runner.start()
        try:
            runner.submit(make_step_request(1))
            first_frame = runner.wait_for_frame(1, timeout_s=1.0)
            assert first_frame is not None
            assert async_result_value(first_frame.outputs) == 1.0

            runner.submit(make_step_request(2))
            assert pipeline.second_started.wait(timeout=1.0)

            latest = runner.latest()
            assert latest is not None
            assert latest.frame_id == 1
            assert async_result_value(latest.outputs) == 1.0

            pipeline.release_second.set()
        finally:
            pipeline.release_second.set()
            runner.stop(timeout_s=1.0)

    def test_submit_after_stop_raises(self):
        pipeline = CountingPipeline()
        config = self._pipelined_config(pipeline)
        runner = make_async_runner(pipeline, config)
        runner.start()
        assert runner.stop(timeout_s=1.0) is True
        assert runner.stop(timeout_s=1.0) is True

        with pytest.raises(AsyncRetargetRunnerStopped, match="submission"):
            runner.submit(make_step_request(1))

    def test_runner_cannot_restart_after_stop(self):
        pipeline = CountingPipeline()
        config = self._pipelined_config(pipeline)
        runner = make_async_runner(pipeline, config)
        runner.start()
        assert runner.stop(timeout_s=1.0) is True

        with pytest.raises(AsyncRetargetRunnerStopped, match="restarted"):
            runner.start()

    def test_stop_timeout_reports_active_worker_and_later_stop_cleans_up(self):
        pipeline = BlockingSecondCountingPipeline()
        config = self._pipelined_config(pipeline)
        runner = make_async_runner(pipeline, config)
        runner.start()
        try:
            runner.submit(make_step_request(1))
            assert runner.wait_for_frame(1, timeout_s=1.0) is not None
            runner.submit(make_step_request(2))
            assert pipeline.second_started.wait(timeout=1.0)

            assert runner.stop(timeout_s=0.01) is False
            assert runner._thread is not None

            pipeline.release_second.set()
            assert runner.stop(timeout_s=1.0) is True
            assert runner._thread is None
        finally:
            pipeline.release_second.set()
            runner.stop(timeout_s=1.0)

    def test_wait_for_frame_returns_none_after_stop(self):
        pipeline = CountingPipeline()
        config = self._pipelined_config(pipeline)
        runner = make_async_runner(pipeline, config)
        runner.start()
        try:
            result = []

            waiter = threading.Thread(
                target=lambda: result.append(runner.wait_for_frame(10)),
            )
            waiter.start()
            runner.stop(timeout_s=1.0)
            waiter.join(timeout=1.0)

            assert result == [None]
        finally:
            runner.stop(timeout_s=1.0)

    def test_wait_for_frame_returns_already_published_frame_after_stop(self):
        pipeline = CountingPipeline()
        config = self._pipelined_config(pipeline)
        runner = make_async_runner(pipeline, config)
        runner.start()
        try:
            runner.submit(make_step_request(1))
            frame = runner.wait_for_frame(1, timeout_s=1.0)
            runner.stop(timeout_s=1.0)

            assert frame is not None
            assert runner.wait_for_frame(1, timeout_s=0.0) is frame
        finally:
            runner.stop(timeout_s=1.0)

    def test_sync_mode_keeps_exact_current_frame_behavior(self):
        pipeline = CountingPipeline()
        config = make_config(
            pipeline,
            retargeting_execution=RetargetingExecutionConfig(
                mode=RetargetingExecutionMode.SYNC
            ),
        )

        with mock_session_dependencies():
            with TeleopSession(config) as session:
                assert async_result_value(session.step()) == 0.0
                assert async_result_value(session.step()) == 1.0
                assert session.last_step_info.returned_age_frames == 0
                assert session.last_step_info.ran_synchronously is True

    def test_execution_mode_is_latched_for_active_session_run(self):
        pipeline = BlockingSecondCountingPipeline()
        config = self._pipelined_config(pipeline)

        with mock_session_dependencies():
            with TeleopSession(config) as session:
                assert async_result_value(session.step()) == 0.0
                session.config.retargeting_execution.mode = (
                    RetargetingExecutionMode.SYNC
                )

                try:
                    result = session.step()

                    assert async_result_value(result) == 0.0
                    assert session.last_step_info.ran_synchronously is False
                    assert session._async_runner is not None
                    assert pipeline.second_started.wait(timeout=1.0)
                finally:
                    pipeline.release_second.set()


class TestTrackerDataCollection:
    """Test _collect_tracker_data for different tracker types."""

    def test_collect_head_tracker_data(self):
        """Head tracker data should be wrapped in TensorGroups."""
        head_source = MockHeadSource()
        pipeline = MockPipeline(leaf_nodes=[head_source])

        config = make_config(pipeline)
        with mock_session_dependencies():
            session = TeleopSession(config)
            with session:
                data = session._collect_tracker_data()

                # Should have data keyed by source name
                assert "head" in data
                # The data should contain the input spec keys
                assert "head_pose" in data["head"]
                # The TensorGroup should contain the head data from the tracker
                tg = data["head"]["head_pose"]
                assert isinstance(tg, TensorGroup)
                assert tg[0] == 42.0  # HeadTracker returns 42.0

    def test_collect_controller_tracker_data(self):
        """Controller tracker data should be split into left/right."""
        controller_source = MockControllersSource()
        pipeline = MockPipeline(leaf_nodes=[controller_source])

        config = make_config(pipeline)
        with mock_session_dependencies():
            session = TeleopSession(config)
            with session:
                data = session._collect_tracker_data()

                assert "controllers" in data
                assert "controller_left" in data["controllers"]
                assert "controller_right" in data["controllers"]

                # Left should get left_controller data
                tg_left = data["controllers"]["controller_left"]
                assert isinstance(tg_left, TensorGroup)
                assert tg_left[0] == 3.0

                # Right should get right_controller data
                tg_right = data["controllers"]["controller_right"]
                assert isinstance(tg_right, TensorGroup)
                assert tg_right[0] == 4.0

    def test_collect_hand_tracker_data(self):
        """Hand tracker data should be split into left/right."""
        hands_source = MockHandsSource()
        pipeline = MockPipeline(leaf_nodes=[hands_source])

        config = make_config(pipeline)
        with mock_session_dependencies():
            session = TeleopSession(config)
            with session:
                data = session._collect_tracker_data()

                assert "hands" in data
                assert "hand_left" in data["hands"]
                assert "hand_right" in data["hands"]

                tg_left = data["hands"]["hand_left"]
                assert isinstance(tg_left, TensorGroup)
                assert tg_left[0] == 1.0

                tg_right = data["hands"]["hand_right"]
                assert isinstance(tg_right, TensorGroup)
                assert tg_right[0] == 2.0

    def test_collect_multiple_sources(self):
        """Data collection works with multiple sources simultaneously."""
        head = MockHeadSource()
        controllers = MockControllersSource()
        pipeline = MockPipeline(leaf_nodes=[head, controllers])

        config = make_config(pipeline)
        with mock_session_dependencies():
            session = TeleopSession(config)
            with session:
                data = session._collect_tracker_data()

                assert "head" in data
                assert "controllers" in data
                assert len(data) == 2

    def test_collect_no_sources(self):
        """Data collection with no sources returns empty dict."""
        pipeline = MockPipeline(leaf_nodes=[])

        config = make_config(pipeline)
        with mock_session_dependencies():
            session = TeleopSession(config)
            with session:
                data = session._collect_tracker_data()
                assert data == {}


class TestElapsedTime:
    """Test get_elapsed_time."""

    def test_elapsed_time_increases(self):
        """Elapsed time should be >= 0 and increase over time."""
        pipeline = MockPipeline(leaf_nodes=[])

        config = make_config(pipeline)
        with mock_session_dependencies():
            session = TeleopSession(config)
            with session:
                t1 = session.get_elapsed_time()
                assert t1 >= 0.0

                # Small sleep to ensure measurable difference
                time.sleep(0.01)

                t2 = session.get_elapsed_time()
                assert t2 > t1


class TestPluginHealthChecking:
    """Test _check_plugin_health."""

    def test_check_plugin_health_calls_all_contexts(self):
        """_check_plugin_health should call check_health on all plugin contexts."""
        pipeline = MockPipeline(leaf_nodes=[])

        config = make_config(pipeline)
        session = TeleopSession(config)

        ctx1 = MockPluginContext()
        ctx2 = MockPluginContext()
        session.plugin_contexts = [ctx1, ctx2]

        session._check_plugin_health()

        assert ctx1.health_check_count == 1
        assert ctx2.health_check_count == 1

    def test_check_plugin_health_no_contexts(self):
        """_check_plugin_health with no contexts should not fail."""
        pipeline = MockPipeline(leaf_nodes=[])

        config = make_config(pipeline)
        session = TeleopSession(config)

        # Should not raise
        session._check_plugin_health()


class TestConfiguration:
    """Test TeleopSessionConfig and PluginConfig dataclasses."""

    def test_teleop_session_config_defaults(self):
        """TeleopSessionConfig should have sensible defaults."""
        pipeline = MockPipeline()
        config = TeleopSessionConfig(app_name="Test", pipeline=pipeline)

        assert config.app_name == "Test"
        assert config.pipeline is pipeline
        assert config.trackers == []
        assert config.plugins == []
        assert config.verbose is True
        assert config.retargeting_execution.mode == RetargetingExecutionMode.SYNC
        assert isinstance(config.retargeting_execution.pacing, ImmediatePacingConfig)

    def test_teleop_session_config_custom(self, tmp_path):
        """TeleopSessionConfig should accept custom values."""
        pipeline = MockPipeline()
        tracker = HeadTracker()
        plugin = PluginConfig(
            plugin_name="p", plugin_root_id="/r", search_paths=[tmp_path]
        )

        config = TeleopSessionConfig(
            app_name="CustomApp",
            pipeline=pipeline,
            trackers=[tracker],
            plugins=[plugin],
            verbose=False,
        )

        assert config.app_name == "CustomApp"
        assert len(config.trackers) == 1
        assert len(config.plugins) == 1
        assert config.verbose is False

    def test_retargeting_execution_config_accepts_pacing_config_objects(self):
        """RetargetingExecutionConfig should preserve typed pacing configs."""
        config = RetargetingExecutionConfig(
            pacing=DeadlinePacingConfig(safety_margin_s=0.015),
        )

        assert isinstance(config.pacing, DeadlinePacingConfig)
        assert config.pacing.safety_margin_s == 0.015

    def test_retargeting_execution_config_default_pacing_is_immediate(self):
        """RetargetingExecutionConfig should default to immediate pacing."""
        config = RetargetingExecutionConfig()

        assert isinstance(config.pacing, ImmediatePacingConfig)

    def test_retargeting_execution_config_accepts_supported_pacing_mode_strings(self):
        """Pacing mode strings should coerce to default pacing config objects."""
        config = RetargetingExecutionConfig(pacing="deadline")

        assert isinstance(config.pacing, DeadlinePacingConfig)

    def test_retargeting_execution_config_rejects_removed_deadline_guarded_pacing(self):
        """The old deadline_guarded spelling is no longer part of the public API."""
        with pytest.raises(ValueError, match="deadline_guarded"):
            RetargetingExecutionConfig(pacing="deadline_guarded")

    def test_retargeting_execution_config_rejects_removed_fixed_delay_pacing(self):
        """Fixed-delay pacing is no longer part of the public config surface."""
        with pytest.raises(ValueError, match="fixed_delay"):
            RetargetingExecutionConfig(pacing="fixed_delay")

    def test_package_exports_concrete_pacing_configs_not_internal_union_alias(self):
        """Users import concrete pacing configs; the union alias stays internal."""
        assert hasattr(teleop_session_manager, "ImmediatePacingConfig")
        assert hasattr(teleop_session_manager, "DeadlinePacingConfig")
        assert hasattr(teleop_session_manager, "AsyncRetargetRunnerStopped")
        assert not hasattr(teleop_session_manager, "DeadlineGuardedPacingConfig")
        assert not hasattr(teleop_session_manager, "FixedDelayPacingConfig")
        assert not hasattr(teleop_session_manager, "RetargetingMissPolicy")
        assert not hasattr(teleop_session_manager, "RetargetingPacingConfig")

    def test_plugin_config_defaults(self, tmp_path):
        """PluginConfig should be enabled and optional by default."""
        config = PluginConfig(
            plugin_name="test",
            plugin_root_id="/root",
            search_paths=[tmp_path],
        )

        assert config.enabled is True
        assert config.required is False

    def test_plugin_config_disabled_and_required(self, tmp_path):
        """PluginConfig accepts explicit lifecycle policy flags."""
        config = PluginConfig(
            plugin_name="test",
            plugin_root_id="/root",
            search_paths=[tmp_path],
            enabled=False,
            required=True,
        )

        assert config.enabled is False
        assert config.required is True


class TestSessionReuse:
    """Test that session state is properly reset between uses."""

    def test_frame_count_resets_on_reenter(self):
        """frame_count should reset when re-entering the session."""
        pipeline = MockPipeline(leaf_nodes=[])

        config = make_config(pipeline)
        with mock_session_dependencies():
            session = TeleopSession(config)
            with session:
                session.step()
                session.step()
                assert session.frame_count == 2

            # Re-entering should reset state
            # Need a new ExitStack since the old one was consumed
            session._exit_stack = __import__("contextlib").ExitStack()

            with session:
                assert session.frame_count == 0

    def test_plugin_lists_are_run_scoped(self, tmp_path):
        """Plugin lists should contain only resources from the current session run."""
        pipeline = MockPipeline(leaf_nodes=[])
        mock_pm = MockPluginManager(plugin_names=["test_plugin"])

        plugin_config = PluginConfig(
            plugin_name="test_plugin",
            plugin_root_id="/root",
            search_paths=[tmp_path],
            enabled=True,
        )

        config = make_config(pipeline, plugins=[plugin_config])
        with mock_session_dependencies(mock_pm=mock_pm):
            session = TeleopSession(config)
            with session:
                first_count = len(session.plugin_managers)

        assert first_count == 1


# ============================================================================
# Replay mode (SessionMode.REPLAY)
# ============================================================================


@contextmanager
def mock_replay_dependencies(mock_pm=None):
    """Patch ReplaySession, DeviceIOSession, and OpenXR so TeleopSession.__enter__ works in replay mode.

    Yields a namespace with:
        - replay_session: the mock returned by ReplaySession.run
        - create_replay: the mock replacing ReplaySession.run
        - create_live: the mock replacing DeviceIOSession.run
        - oxr_cls: the mock replacing OpenXRSession
        - replay_config_cls: the mock replacing McapReplayConfig
    """
    mock_dio_session = MockDeviceIOSession()

    mock_pm = mock_pm or MockPluginManager()

    with (
        patch(
            "isaaccapture.deviceio.ReplaySession.run",
            return_value=mock_dio_session,
        ) as create_replay,
        patch(
            "isaaccapture.deviceio.DeviceIOSession.run",
            return_value=MagicMock(),
        ) as create_live,
        patch("isaaccapture.oxr.OpenXRSession", return_value=MagicMock()) as oxr_cls,
        patch(
            "isaaccapture.deviceio.DeviceIOSession.get_required_extensions",
            return_value=[],
        ),
        patch("isaaccapture.plugin_manager.PluginManager", return_value=mock_pm),
        patch("isaaccapture.deviceio.McapReplayConfig") as replay_config_cls,
    ):
        replay_config_cls.return_value = MagicMock()
        ns = MagicMock()
        ns.replay_session = mock_dio_session
        ns.create_replay = create_replay
        ns.create_live = create_live
        ns.oxr_cls = oxr_cls
        ns.replay_config_cls = replay_config_cls
        yield ns


class TestReplayModeConfigValidation:
    """Tests for SessionMode field validation in TeleopSessionConfig."""

    def test_replay_mode_requires_mcap_config(self):
        """REPLAY mode without mcap_config raises ValueError."""
        with pytest.raises(ValueError, match="mcap_config is required"):
            TeleopSessionConfig(
                app_name="test",
                pipeline=MockPipeline(),
                mode=SessionMode.REPLAY,
                mcap_config=None,
            )

    def test_replay_mode_accepts_mcap_config(self):
        """REPLAY mode with mcap_config provided succeeds."""
        config = TeleopSessionConfig(
            app_name="test",
            pipeline=MockPipeline(),
            mode=SessionMode.REPLAY,
            mcap_config=MagicMock(),
        )
        assert config.mode == SessionMode.REPLAY
        assert config.mcap_config is not None

    def test_default_mode_is_live(self):
        """Default mode is SessionMode.LIVE."""
        config = TeleopSessionConfig(
            app_name="test",
            pipeline=MockPipeline(),
        )
        assert config.mode == SessionMode.LIVE


class TestReplayModeRejectsVendorSelection:
    """Vendor selection is live-only; a vendored source in REPLAY must fail fast."""

    @staticmethod
    def _vendored_head_source():
        source = MockHeadSource(name="head")
        # MockDeviceIOSource defaults get_vendor() to None; set any non-None
        # selection to simulate a vendored source.
        source._vendor = object()
        return source

    def test_vendored_source_in_replay_raises(self):
        """A source carrying a vendor is rejected at construction in REPLAY mode."""
        config = TeleopSessionConfig(
            app_name="test",
            pipeline=MockPipeline(leaf_nodes=[self._vendored_head_source()]),
            mode=SessionMode.REPLAY,
            mcap_config=MagicMock(),
        )
        with pytest.raises(ValueError, match="Vendor selection is only valid"):
            TeleopSession(config)

    def test_unvendored_source_in_replay_is_allowed(self):
        """A default (unvendored) source constructs fine in REPLAY mode."""
        config = TeleopSessionConfig(
            app_name="test",
            pipeline=MockPipeline(leaf_nodes=[MockHeadSource(name="head")]),
            mode=SessionMode.REPLAY,
            mcap_config=MagicMock(),
        )
        TeleopSession(config)  # get_vendor() is None -> no raise

    def test_vendored_source_in_live_is_allowed(self):
        """A vendored source does not trip the replay-only guard in LIVE mode.

        Construction must not raise: vendor validation is a live-session concern
        deferred to session entry (``__enter__`` -> ``VendorConfig``), so unlike
        REPLAY there is no construction-time rejection here.
        """
        config = TeleopSessionConfig(
            app_name="test",
            pipeline=MockPipeline(leaf_nodes=[self._vendored_head_source()]),
            mode=SessionMode.LIVE,
        )
        TeleopSession(config)  # replay-only guard does not fire in LIVE -> no raise

    def test_mode_flipped_to_replay_after_construction_rejects_on_enter(self):
        """The guard re-runs on context entry, not just at construction.

        config is mutable, so a session built in LIVE with a vendored source
        (which does not raise at construction) that is later switched to REPLAY
        must still fail fast on ``__enter__`` rather than silently ignore the
        source's vendor while replaying. Replay dependencies are fully mocked so
        the only thing that can reject entry is the revalidated vendor guard.
        """
        config = TeleopSessionConfig(
            app_name="test",
            pipeline=MockPipeline(leaf_nodes=[self._vendored_head_source()]),
            mode=SessionMode.LIVE,
        )
        session = TeleopSession(config)  # LIVE at construction -> no raise
        config.mode = SessionMode.REPLAY
        config.mcap_config = MagicMock()
        with mock_replay_dependencies():
            with pytest.raises(ValueError, match="Vendor selection is only valid"):
                with session:
                    pass


class TestReplayModeSessionEnter:
    """Tests for TeleopSession.__enter__ when mode is REPLAY."""

    def _make_replay_config(self):
        mcap_config = MagicMock()
        return TeleopSessionConfig(
            app_name="test",
            pipeline=MockPipeline(leaf_nodes=[]),
            mode=SessionMode.REPLAY,
            mcap_config=mcap_config,
        )

    def test_calls_create_replay_session(self):
        """ReplaySession.run is called in replay mode."""
        config = self._make_replay_config()

        with mock_replay_dependencies() as mocks:
            with patch("isaaccapture.deviceio.McapReplayConfig") as mock_replay_cls:
                mock_replay_cls.return_value = MagicMock()
                session = TeleopSession(config)
                session.__enter__()
                try:
                    mocks.create_replay.assert_called_once_with(
                        mock_replay_cls.return_value
                    )
                finally:
                    session.__exit__(None, None, None)

    def test_skips_oxr_session_creation(self):
        """OpenXRSession is NOT instantiated in replay mode."""
        config = self._make_replay_config()

        with mock_replay_dependencies() as mocks:
            session = TeleopSession(config)
            session.__enter__()
            try:
                mocks.oxr_cls.assert_not_called()
            finally:
                session.__exit__(None, None, None)

    def test_skips_create_live_session(self):
        """DeviceIOSession.run is NOT called in replay mode."""
        config = self._make_replay_config()

        with mock_replay_dependencies() as mocks:
            session = TeleopSession(config)
            session.__enter__()
            try:
                mocks.create_live.assert_not_called()
            finally:
                session.__exit__(None, None, None)

    def test_oxr_session_remains_none(self):
        """session.oxr_session stays None in replay mode."""
        config = self._make_replay_config()

        with mock_replay_dependencies():
            session = TeleopSession(config)
            session.__enter__()
            try:
                assert session.oxr_session is None
            finally:
                session.__exit__(None, None, None)

    def test_deviceio_session_is_set(self):
        """DeviceIO session is populated in replay mode."""
        config = self._make_replay_config()

        with mock_replay_dependencies() as mocks:
            session = TeleopSession(config)
            session.__enter__()
            try:
                assert session.deviceio_session is mocks.replay_session
            finally:
                session.__exit__(None, None, None)

    def test_replay_pipelined_mode_returns_latest_completed_result(self):
        """Replay pipelined mode follows the same latest-completed contract."""
        pipeline = BlockingSecondCountingPipeline()
        config = TeleopSessionConfig(
            app_name="test",
            pipeline=pipeline,
            mode=SessionMode.REPLAY,
            mcap_config=MagicMock(),
            retargeting_execution=RetargetingExecutionConfig(
                mode=RetargetingExecutionMode.PIPELINED
            ),
        )

        with mock_replay_dependencies():
            with TeleopSession(config) as session:
                assert async_result_value(session.step()) == 0.0
                result = session.step()

                assert async_result_value(result) == 0.0
                assert session.last_step_info.ran_synchronously is False
                assert session.last_step_info.returned_frame_id == 0
                assert session.last_step_info.submitted_frame_id == 1
                assert pipeline.second_started.wait(timeout=1.0)
                pipeline.release_second.set()


class TestReplayModePlugins:
    """Tests that plugins are initialized in replay mode."""

    def test_plugins_initialized_in_replay_mode(self, tmp_path):
        """Enabled plugins are started even when mode is REPLAY."""
        mock_pm = MockPluginManager(plugin_names=["test_plugin"])

        plugin_config = PluginConfig(
            plugin_name="test_plugin",
            plugin_root_id="/root",
            search_paths=[tmp_path],
            enabled=True,
        )

        config = TeleopSessionConfig(
            app_name="test",
            pipeline=MockPipeline(leaf_nodes=[]),
            mode=SessionMode.REPLAY,
            mcap_config=MagicMock(),
            plugins=[plugin_config],
        )

        with (
            patch(
                "isaaccapture.deviceio_trackers.PluginDeviceStatusTracker"
            ) as tracker_cls,
            mock_replay_dependencies(mock_pm=mock_pm),
        ):
            session = TeleopSession(config)
            session.__enter__()
            try:
                assert len(session.plugin_managers) == 1
                assert len(session.plugin_contexts) == 1
                assert [call["plugin_name"] for call in mock_pm.start_calls] == [
                    "test_plugin"
                ]
                tracker_cls.assert_not_called()
                assert session.get_status().providers == ()
                assert session.get_status().devices == ()
            finally:
                session.__exit__(None, None, None)


class TestReplayModeAutoPopulate:
    """Tests that replay mode auto-populates tracker names from pipeline sources."""

    def test_auto_populates_when_tracker_names_empty(self):
        """Empty get_tracker_names() triggers auto-populate from pipeline sources."""
        mcap_config = MagicMock()
        mcap_config.filename = "recording.mcap"
        mcap_config.get_tracker_names.return_value = []

        head_source = MockHeadSource(name="head")
        hands_source = MockHandsSource(name="hands")
        pipeline = MockPipeline(leaf_nodes=[head_source, hands_source])

        config = TeleopSessionConfig(
            app_name="test",
            pipeline=pipeline,
            mode=SessionMode.REPLAY,
            mcap_config=mcap_config,
        )

        with mock_replay_dependencies() as mocks:
            with patch("isaaccapture.deviceio.McapReplayConfig") as mock_replay_cls:
                mock_replay_cls.return_value = MagicMock()
                session = TeleopSession(config)
                session.__enter__()
                try:
                    mock_replay_cls.assert_called_once_with(
                        "recording.mcap",
                        [
                            (head_source.get_tracker(), "head"),
                            (hands_source.get_tracker(), "hands"),
                        ],
                    )
                    mocks.create_replay.assert_called_once_with(
                        mock_replay_cls.return_value
                    )
                finally:
                    session.__exit__(None, None, None)

    def test_merges_sources_with_explicit_tracker_names(self):
        """Explicit get_tracker_names() are appended after auto-discovered sources."""
        mcap_config = MagicMock()
        mcap_config.filename = "recording.mcap"
        mcap_config.get_tracker_names.return_value = [("extra_tracker", "extra")]

        head_source = MockHeadSource(name="head")
        pipeline = MockPipeline(leaf_nodes=[head_source])

        config = TeleopSessionConfig(
            app_name="test",
            pipeline=pipeline,
            mode=SessionMode.REPLAY,
            mcap_config=mcap_config,
        )

        with mock_replay_dependencies() as mocks:
            with patch("isaaccapture.deviceio.McapReplayConfig") as mock_replay_cls:
                mock_replay_cls.return_value = MagicMock()
                session = TeleopSession(config)
                session.__enter__()
                try:
                    mock_replay_cls.assert_called_once_with(
                        "recording.mcap",
                        [
                            (head_source.get_tracker(), "head"),
                            ("extra_tracker", "extra"),
                        ],
                    )
                    mocks.create_replay.assert_called_once_with(
                        mock_replay_cls.return_value
                    )
                finally:
                    session.__exit__(None, None, None)


# ============================================================================
# Live mode with MCAP recording
# ============================================================================


@contextmanager
def mock_live_dependencies_with_args():
    """Patch DeviceIO and OpenXR for live-mode tests that inspect DeviceIOSession.run args.

    Yields a namespace with:
        - create_live: the mock replacing DeviceIOSession.run (inspect call_args)
        - dio_session: the mock DeviceIO session returned by DeviceIOSession.run
        - recording_config_cls: the mock replacing McapRecordingConfig
    """
    mock_dio_session = MockDeviceIOSession()

    mock_oxr_session = MockOpenXRSession()

    with (
        patch(
            "isaaccapture.deviceio.DeviceIOSession.run",
            return_value=mock_dio_session,
        ) as create_live,
        patch("isaaccapture.oxr.OpenXRSession", return_value=mock_oxr_session),
        patch(
            "isaaccapture.deviceio.DeviceIOSession.get_required_extensions",
            return_value=[],
        ),
        patch("isaaccapture.plugin_manager.PluginManager", return_value=MagicMock()),
        patch("isaaccapture.deviceio.McapRecordingConfig") as recording_config_cls,
    ):
        recording_config_cls.return_value = MagicMock()
        ns = MagicMock()
        ns.create_live = create_live
        ns.dio_session = mock_dio_session
        ns.recording_config_cls = recording_config_cls
        yield ns


class TestLiveModeWithMcapRecording:
    """Tests that mcap_config is built from discovered sources in live mode."""

    def test_no_mcap_config_passes_none(self):
        """When mcap_config is not set, None is passed to DeviceIOSession.run."""
        config = TeleopSessionConfig(
            app_name="test",
            pipeline=MockPipeline(leaf_nodes=[]),
            mode=SessionMode.LIVE,
        )

        with mock_live_dependencies_with_args() as mocks:
            session = TeleopSession(config)
            session.__enter__()
            try:
                mocks.create_live.assert_called_once()
                actual_mcap = mocks.create_live.call_args[0][2]
                assert actual_mcap is None
            finally:
                session.__exit__(None, None, None)

    def test_mcap_auto_populates_when_tracker_names_empty(self):
        """Empty get_tracker_names() triggers auto-populate from pipeline sources."""
        mcap_config = MagicMock()
        mcap_config.filename = "test.mcap"
        mcap_config.get_tracker_names.return_value = []

        head_source = MockHeadSource(name="head")
        hands_source = MockHandsSource(name="hands")
        pipeline = MockPipeline(leaf_nodes=[head_source, hands_source])

        config = TeleopSessionConfig(
            app_name="test",
            pipeline=pipeline,
            mode=SessionMode.LIVE,
            mcap_config=mcap_config,
        )

        with mock_live_dependencies_with_args() as mocks:
            session = TeleopSession(config)
            session.__enter__()
            try:
                mocks.recording_config_cls.assert_called_once_with(
                    "test.mcap",
                    [
                        (head_source.get_tracker(), "head"),
                        (hands_source.get_tracker(), "hands"),
                    ],
                )
                actual_mcap = mocks.create_live.call_args[0][2]
                assert actual_mcap is mocks.recording_config_cls.return_value
            finally:
                session.__exit__(None, None, None)

    def test_mcap_merges_sources_with_explicit_tracker_names(self):
        """Explicit get_tracker_names() are appended after auto-discovered sources."""
        mcap_config = MagicMock()
        mcap_config.filename = "test.mcap"
        mcap_config.get_tracker_names.return_value = [("extra_tracker", "extra")]

        head_source = MockHeadSource(name="head")
        pipeline = MockPipeline(leaf_nodes=[head_source])

        config = TeleopSessionConfig(
            app_name="test",
            pipeline=pipeline,
            mode=SessionMode.LIVE,
            mcap_config=mcap_config,
        )

        with mock_live_dependencies_with_args() as mocks:
            session = TeleopSession(config)
            session.__enter__()
            try:
                mocks.recording_config_cls.assert_called_once_with(
                    "test.mcap",
                    [
                        (head_source.get_tracker(), "head"),
                        ("extra_tracker", "extra"),
                    ],
                )
                actual_mcap = mocks.create_live.call_args[0][2]
                assert actual_mcap is mocks.recording_config_cls.return_value
            finally:
                session.__exit__(None, None, None)


class TestMcapConfigGetTrackerNames:
    """Tests for McapRecordingConfig.get_tracker_names() (requires compiled C++ bindings)."""

    @pytest.fixture(autouse=True)
    def _import_deviceio(self):
        self.deviceio = pytest.importorskip("isaaccapture.deviceio")

    def test_get_tracker_names_returns_pairs(self):
        """get_tracker_names() returns the (tracker, name) pairs passed at construction."""
        hand = self.deviceio.HandTracker()
        head = self.deviceio.HeadTracker()
        config = self.deviceio.McapRecordingConfig(
            "out.mcap", [(hand, "hands"), (head, "head")]
        )

        result = config.get_tracker_names()
        assert len(result) == 2
        assert result[0][0] is hand
        assert result[0][1] == "hands"
        assert result[1][0] is head
        assert result[1][1] == "head"

    def test_get_tracker_names_empty_by_default(self):
        """McapRecordingConfig constructed with only a filename has empty tracker_names."""
        config = self.deviceio.McapRecordingConfig("out.mcap")

        result = config.get_tracker_names()
        assert result == []

    def test_get_tracker_names_single_tracker(self):
        """get_tracker_names() works with a single tracker."""
        head = self.deviceio.HeadTracker()
        config = self.deviceio.McapRecordingConfig("out.mcap", [(head, "tracking")])

        result = config.get_tracker_names()
        assert len(result) == 1
        assert result[0][0] is head
        assert result[0][1] == "tracking"


class TestMcapReplayConfigGetTrackerNames:
    """Tests for McapReplayConfig.get_tracker_names() (requires compiled C++ bindings)."""

    @pytest.fixture(autouse=True)
    def _import_deviceio(self):
        self.deviceio = pytest.importorskip("isaaccapture.deviceio")

    def test_get_tracker_names_returns_pairs(self):
        """get_tracker_names() returns the (tracker, name) pairs passed at construction."""
        hand = self.deviceio.HandTracker()
        head = self.deviceio.HeadTracker()
        config = self.deviceio.McapReplayConfig(
            "out.mcap", [(hand, "hands"), (head, "head")]
        )

        result = config.get_tracker_names()
        assert len(result) == 2
        assert result[0][0] is hand
        assert result[0][1] == "hands"
        assert result[1][0] is head
        assert result[1][1] == "head"

    def test_get_tracker_names_empty_by_default(self):
        """McapReplayConfig constructed with only a filename has empty tracker_names."""
        config = self.deviceio.McapReplayConfig("out.mcap")

        result = config.get_tracker_names()
        assert result == []

    def test_get_tracker_names_single_tracker(self):
        """get_tracker_names() works with a single tracker."""
        head = self.deviceio.HeadTracker()
        config = self.deviceio.McapReplayConfig("out.mcap", [(head, "tracking")])

        result = config.get_tracker_names()
        assert len(result) == 1
        assert result[0][0] is head
        assert result[0][1] == "tracking"


# ============================================================================
# Output sinks (IDeviceIOSink) discovery + post-graph flush
# ============================================================================


_SINK_RESULT_TYPE = TensorGroupType("type_result", [FloatType("value")])


class RecordingSink(IDeviceIOSink):
    """Test ``IDeviceIOSink`` that records applied values and flush sessions."""

    def __init__(self, name, tracker=None, input_type=_SINK_RESULT_TYPE):
        self._tracker = tracker
        self._input_type = input_type
        self.applied: list = []
        self.flush_sessions: list = []
        super().__init__(name)

    def input_spec(self) -> RetargeterIOType:
        return {"in": self._input_type}

    def get_tracker(self):
        return self._tracker

    def flush_to_device(self, deviceio_session):
        self.flush_sessions.append(deviceio_session)

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        self.applied.append(inputs["in"][0])


def _trivial_pipeline():
    """A real GraphExecutable pipeline (no inputs) for sink tests."""
    gen = MockEmptyInputRetargeter("gen")
    return OutputCombiner({"value": gen.output("value")})


def _external_step_request(
    leaf_name: str, input_name: str, value: float
) -> StepRequest:
    tg = TensorGroup(TensorGroupType("type_ext", [FloatType("value")]))
    tg[0] = value
    return StepRequest(
        frame_id=0,
        external_inputs={leaf_name: {input_name: tg}},
        graph_time=GraphTime(sim_time_ns=0, real_time_ns=0),
        execution_events=ExecutionEvents(
            reset=False, execution_state=ExecutionState.RUNNING
        ),
        submitted_time_s=time.monotonic(),
    )


class TestOutputSinks:
    def test_sink_and_its_external_leaf_are_discovered(self):
        ext = MockExternalRetargeter("sim_force")
        sink = RecordingSink("haptic")
        sink_graph = sink.connect({"in": ext.output("result")})
        config = TeleopSessionConfig(
            app_name="t", pipeline=_trivial_pipeline(), sinks=[sink_graph]
        )
        session = TeleopSession(config)

        assert [node for _, node in session._sinks] == [sink]
        # The leaf feeding the sink becomes an external input the app must supply.
        assert "sim_force" in session.get_external_input_specs()

    def test_sink_subgraph_source_is_discovered(self):
        source = MockDeviceIOSource("force_src", HeadTracker(), input_names=["input_0"])
        sink = RecordingSink(
            "haptic",
            input_type=TensorGroupType("type_output", [FloatType("value")]),
        )
        sink_graph = sink.connect({"in": source.output("output_0")})
        config = TeleopSessionConfig(
            app_name="t", pipeline=_trivial_pipeline(), sinks=[sink_graph]
        )
        session = TeleopSession(config)

        assert source in session._sources
        assert [node for _, node in session._sinks] == [sink]

    def test_sink_flushed_after_graph_with_external_value(self):
        ext = MockExternalRetargeter("sim_force")
        sink = RecordingSink("haptic")
        sink_graph = sink.connect({"in": ext.output("result")})
        config = TeleopSessionConfig(
            app_name="t", pipeline=_trivial_pipeline(), sinks=[sink_graph]
        )
        session = TeleopSession(config)
        session.deviceio_session = MockDeviceIOSession()

        request = _external_step_request("sim_force", "external_data", 0.42)
        session._execute_step_request(request)

        # The retargeter echoed the app-submitted value into the sink, and the
        # sink was flushed once with the active session.
        assert sink.applied == [pytest.approx(0.42)]
        assert sink.flush_sessions == [session.deviceio_session]

    def test_compute_runs_before_flush(self):
        """flush_to_device must observe the value _compute_fn stored this frame."""
        ext = MockExternalRetargeter("sim_force")

        class OrderSink(RecordingSink):
            def __init__(self, name):
                super().__init__(name)
                self.events: list = []

            def _compute_fn(self, inputs, outputs, context):
                self.events.append(("compute", inputs["in"][0]))

            def flush_to_device(self, deviceio_session):
                self.events.append(("flush",))

        sink = OrderSink("haptic")
        sink_graph = sink.connect({"in": ext.output("result")})
        config = TeleopSessionConfig(
            app_name="t", pipeline=_trivial_pipeline(), sinks=[sink_graph]
        )
        session = TeleopSession(config)
        session.deviceio_session = MockDeviceIOSession()

        session._execute_step_request(
            _external_step_request("sim_force", "external_data", 1.0)
        )

        assert sink.events == [("compute", pytest.approx(1.0)), ("flush",)]

    def test_config_rejects_non_sink(self):
        gen = MockEmptyInputRetargeter("gen")
        with pytest.raises(ValueError, match="IDeviceIOSink"):
            TeleopSessionConfig(
                app_name="t",
                pipeline=_trivial_pipeline(),
                sinks=[gen.connect({})],
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_bundled_plugins_dir_finds_the_root_that_holds_the_plugins(tmp_path):
    # An editable install puts the authored source root, which has no plugins/, beside the install root.
    source_root, install_root = tmp_path / "src", tmp_path / "site"
    source_root.mkdir()
    (install_root / "plugins").mkdir(parents=True)

    roots = [str(source_root), str(install_root)]
    assert _bundled_plugins_dir(roots) == install_root / "plugins"
    assert _bundled_plugins_dir(roots[:1]) == source_root / "plugins"

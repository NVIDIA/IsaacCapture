# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end tests for reset propagation through TeleopSession.

Verifies that ``TeleopSession.step(execution_events=ExecutionEvents(reset=True))``
correctly propagates the reset signal to stateful retargeters in a real pipeline
(not mocked). All OpenXR/DeviceIO dependencies are patched out.
"""

import numpy as np
import pytest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from isaaccapture.retargeting_engine.interface import (
    ExecutionEvents,
    ExecutionState,
    OptionalTensorGroup,
)

from isaaccapture.retargeters import (
    GripperRetargeter,
    GripperRetargeterConfig,
    LocomotionRootCmdRetargeter,
    LocomotionRootCmdRetargeterConfig,
)

from isaaccapture.retargeting_engine.tensor_types import ControllerInput, HandInput
from isaaccapture.teleop_session_manager import (
    RetargetingExecutionConfig,
    RetargetingExecutionMode,
    TeleopSession,
    TeleopSessionConfig,
)


# ============================================================================
# Helpers
# ============================================================================


@contextmanager
def _mock_session_deps():
    """Patch OpenXR/DeviceIO/PluginManager so TeleopSession.__enter__ works without hardware."""
    mock_oxr = MagicMock()
    mock_oxr.__enter__ = MagicMock(return_value=mock_oxr)
    mock_oxr.__exit__ = MagicMock(return_value=False)
    mock_oxr.get_handles.return_value = MagicMock()

    mock_dio = MagicMock()
    mock_dio.__enter__ = MagicMock(return_value=mock_dio)
    mock_dio.__exit__ = MagicMock(return_value=False)

    with (
        patch("isaaccapture.oxr.OpenXRSession", return_value=mock_oxr),
        # Mock trackers never reach the native binding; keep the OpenXR path these tests cover.
        patch(
            "isaaccapture.deviceio.DeviceIOSession.requires_openxr", return_value=True
        ),
        patch("isaaccapture.deviceio.DeviceIOSession.run", return_value=mock_dio),
        patch(
            "isaaccapture.deviceio.DeviceIOSession.get_required_extensions",
            return_value=[],
        ),
        patch("isaaccapture.plugin_manager.PluginManager", return_value=MagicMock()),
    ):
        yield


def _absent_controller_inputs():
    """Return absent OptionalTensorGroups for both controller slots."""
    return {
        "controller_left": OptionalTensorGroup(ControllerInput()),
        "controller_right": OptionalTensorGroup(ControllerInput()),
    }


def _absent_gripper_inputs(hand_side: str):
    """Return absent OptionalTensorGroups for gripper retargeter inputs."""
    inputs = {f"controller_{hand_side}": OptionalTensorGroup(ControllerInput())}
    inputs[f"hand_{hand_side}"] = OptionalTensorGroup(HandInput())
    return inputs


def _running_events(*, reset: bool = False) -> ExecutionEvents:
    return ExecutionEvents(reset=reset, execution_state=ExecutionState.RUNNING)


def _sync_retargeting() -> RetargetingExecutionConfig:
    return RetargetingExecutionConfig(mode=RetargetingExecutionMode.SYNC)


# ============================================================================
# Tests
# ============================================================================


class TestSessionResetLocomotion:
    """TeleopSession.step(execution_events=reset) resets LocomotionRootCmdRetargeter state."""

    @pytest.fixture()
    def retargeter(self):
        return LocomotionRootCmdRetargeter(
            LocomotionRootCmdRetargeterConfig(initial_hip_height=0.72),
            name="loco",
        )

    def test_reset_restores_hip_height_via_session(self, retargeter):
        """Hip height is restored to initial value when TeleopSession propagates reset."""
        config = TeleopSessionConfig(
            app_name="test_reset",
            pipeline=retargeter,
            retargeting_execution=_sync_retargeting(),
        )

        with _mock_session_deps():
            with TeleopSession(config) as session:
                ext = {"loco": _absent_controller_inputs()}

                # Run a step to establish baseline
                session.step(
                    external_inputs=ext,
                    execution_events=_running_events(),
                )

                # Mutate state
                retargeter._hip_height = 0.95

                # Step with reset=True
                result = session.step(
                    external_inputs=ext,
                    execution_events=_running_events(reset=True),
                )

                cmd = np.from_dlpack(result["root_command"][0])
                assert cmd[3] == pytest.approx(0.72), (
                    "hip_height should be restored to initial after reset"
                )

    def test_default_sync_reset_returns_current_reset_state(self, retargeter):
        """Default sync mode should return the reset frame's current state."""
        config = TeleopSessionConfig(app_name="test_reset", pipeline=retargeter)

        with _mock_session_deps():
            with TeleopSession(config) as session:
                ext = {"loco": _absent_controller_inputs()}

                session.step(
                    external_inputs=ext,
                    execution_events=_running_events(),
                )
                retargeter._hip_height = 0.95

                result = session.step(
                    external_inputs=ext,
                    execution_events=_running_events(reset=True),
                )

                cmd = np.from_dlpack(result["root_command"][0])
                assert cmd[3] == pytest.approx(0.72)
                assert session.last_step_info.returned_age_frames == 0
                assert session.last_step_info.ran_synchronously is True
                assert session.last_context.execution_events.reset is True

    def test_no_reset_preserves_state_via_session(self, retargeter):
        """Hip height stays mutated when reset is not signalled."""
        config = TeleopSessionConfig(
            app_name="test_no_reset",
            pipeline=retargeter,
            retargeting_execution=_sync_retargeting(),
        )

        with _mock_session_deps():
            with TeleopSession(config) as session:
                ext = {"loco": _absent_controller_inputs()}

                session.step(
                    external_inputs=ext,
                    execution_events=_running_events(),
                )

                retargeter._hip_height = 0.95

                result = session.step(
                    external_inputs=ext,
                    execution_events=_running_events(reset=False),
                )

                cmd = np.from_dlpack(result["root_command"][0])
                assert cmd[3] == pytest.approx(0.95), (
                    "hip_height should stay mutated without reset"
                )


class TestSessionResetGripper:
    """TeleopSession.step(execution_events=reset) resets GripperRetargeter state."""

    @pytest.fixture()
    def retargeter(self):
        return GripperRetargeter(
            GripperRetargeterConfig(hand_side="right"),
            name="gripper",
        )

    def test_reset_reopens_gripper_via_session(self, retargeter):
        """Gripper outputs open (1.0) after session-level reset."""
        config = TeleopSessionConfig(
            app_name="test_gripper_reset",
            pipeline=retargeter,
            retargeting_execution=_sync_retargeting(),
        )

        with _mock_session_deps():
            with TeleopSession(config) as session:
                ext = {"gripper": _absent_gripper_inputs("right")}

                session.step(
                    external_inputs=ext,
                    execution_events=_running_events(),
                )

                # Close the gripper
                retargeter._previous_gripper_command = True

                # Reset via session
                result = session.step(
                    external_inputs=ext,
                    execution_events=_running_events(reset=True),
                )

                cmd = result["gripper_command"][0]
                assert cmd == pytest.approx(1.0), (
                    "gripper should be open after session-level reset"
                )

    def test_no_reset_keeps_gripper_closed_via_session(self, retargeter):
        """Gripper internal state stays closed when reset is not signalled."""
        config = TeleopSessionConfig(
            app_name="test_gripper_no_reset",
            pipeline=retargeter,
            retargeting_execution=_sync_retargeting(),
        )

        with _mock_session_deps():
            with TeleopSession(config) as session:
                ext = {"gripper": _absent_gripper_inputs("right")}

                session.step(
                    external_inputs=ext,
                    execution_events=_running_events(),
                )

                retargeter._previous_gripper_command = True

                session.step(
                    external_inputs=ext,
                    execution_events=_running_events(reset=False),
                )

                assert retargeter._previous_gripper_command is True, (
                    "gripper state should stay closed without reset"
                )

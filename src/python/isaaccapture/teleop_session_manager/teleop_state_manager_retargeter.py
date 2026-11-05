# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Teleop state-manager retargeters.

Defines:
    - TeleopStateManager: abstract API retargeter with fixed outputs.
    - DefaultTeleopStateManager: concrete default execution state manager.
"""

from abc import abstractmethod

from isaaccapture.retargeting_engine.interface import BaseRetargeter, RetargeterIOType
from isaaccapture.retargeting_engine.interface.retargeter_core_types import (
    ComputeContext,
    RetargeterIO,
)
from isaaccapture.retargeting_engine.interface.tensor_group_type import OptionalType
from isaaccapture.retargeting_engine.interface.execution_events import (
    ExecutionState,
    ExecutionEvents,
)

from .teleop_state_manager_types import bool_signal, teleop_state_manager_output_spec


class TeleopStateManager(BaseRetargeter):
    """Abstract teleop state-manager API with fixed teleop-state/reset outputs."""

    OUTPUT_TELEOP_STATE = "teleop_state"
    OUTPUT_RESET_EVENT = "reset_event"

    def output_spec(self) -> RetargeterIOType:
        """Fixed output contract for all teleop state managers."""
        return teleop_state_manager_output_spec()

    @abstractmethod
    def _compute_execution_events(
        self, inputs: RetargeterIO, context: ComputeContext
    ) -> ExecutionEvents:
        """Compute teleop app state and reset pulse for this frame."""
        ...

    def _compute_fn(
        self, inputs: RetargeterIO, outputs: RetargeterIO, context: ComputeContext
    ) -> None:
        events = self._compute_execution_events(inputs, context)
        teleop_state = outputs[self.OUTPUT_TELEOP_STATE]

        for idx, tensor_type in enumerate(teleop_state.group_type.types):
            try:
                channel_state = ExecutionState(tensor_type.name)
            except ValueError as exc:
                raise ValueError(
                    "teleop_state output channels must be named with ExecutionState "
                    f"values; got unknown channel '{tensor_type.name}'"
                ) from exc
            teleop_state[idx] = channel_state == events.execution_state

        outputs[self.OUTPUT_RESET_EVENT][0] = events.reset


class DefaultTeleopStateManager(TeleopStateManager):
    """Default execution state manager.

    Inputs:
        - kill_button: dedicated safety button. While held, force STOPPED; its
          rising edge emits reset_event=True once per press.
        - run_toggle_button: rising edge drives the sequence:
          * STOPPED -> PAUSED
          * PAUSED -> RUNNING
          * RUNNING -> PAUSED
        - reset_button: rising edge emits reset_event=True without changing state.

    Safety:
        If kill/run-toggle input is absent (OptionalTensorGroup is None),
        fail-safe immediately to STOPPED and emit reset_event=True on the first
        frame of the loss, whether or not the optional reset input is present.
    """

    INPUT_KILL = "kill_button"
    INPUT_RUN_TOGGLE = "run_toggle_button"
    INPUT_RESET = "reset_button"

    def __init__(self, name: str) -> None:
        self._state = ExecutionState.STOPPED
        self._prev_kill = False
        self._prev_run_toggle = False
        self._prev_reset = False
        # Inputs absent from startup are not a loss; state already starts STOPPED.
        self._required_inputs_lost_prev = True
        super().__init__(name=name)

    def input_spec(self) -> RetargeterIOType:
        return {
            self.INPUT_KILL: OptionalType(bool_signal(self.INPUT_KILL)),
            self.INPUT_RUN_TOGGLE: OptionalType(bool_signal(self.INPUT_RUN_TOGGLE)),
            self.INPUT_RESET: OptionalType(bool_signal(self.INPUT_RESET)),
        }

    def _compute_execution_events(
        self, inputs: RetargeterIO, context: ComputeContext
    ) -> ExecutionEvents:
        del context

        reset_present = not inputs[self.INPUT_RESET].is_none

        if inputs[self.INPUT_KILL].is_none or inputs[self.INPUT_RUN_TOGGLE].is_none:
            self._state = ExecutionState.STOPPED
            reset_on_loss = not self._required_inputs_lost_prev
            self._required_inputs_lost_prev = True
            # Conservative fail-safe: require button release after input recovery
            # so a still-held button cannot appear as a synthetic rising edge.
            self._prev_kill = True
            self._prev_run_toggle = True
            self._prev_reset = True
            return ExecutionEvents(
                reset=reset_on_loss,
                execution_state=self._state,
            )

        self._required_inputs_lost_prev = False
        kill_pressed = bool(inputs[self.INPUT_KILL][0])
        run_toggle_pressed = bool(inputs[self.INPUT_RUN_TOGGLE][0])
        if reset_present:
            reset_pressed = bool(inputs[self.INPUT_RESET][0])
            reset_edge = reset_pressed and not self._prev_reset
            self._prev_reset = reset_pressed
        else:
            reset_edge = False

        kill_edge = kill_pressed and not self._prev_kill
        run_toggle_edge = run_toggle_pressed and not self._prev_run_toggle
        self._prev_kill = kill_pressed
        self._prev_run_toggle = run_toggle_pressed

        reset = reset_edge or kill_edge
        if kill_pressed:
            self._state = ExecutionState.STOPPED
        elif run_toggle_edge:
            if self._state == ExecutionState.STOPPED:
                self._state = ExecutionState.PAUSED
            elif self._state == ExecutionState.PAUSED:
                self._state = ExecutionState.RUNNING
            elif self._state == ExecutionState.RUNNING:
                self._state = ExecutionState.PAUSED

        return ExecutionEvents(reset=reset, execution_state=self._state)


# Backward-compatible alias kept for downstream code migration.
TwoButtonTeleopStateManager = DefaultTeleopStateManager

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Simple Teleop controls example.

Focuses on teleop-control wiring:
- Converts left-controller button states to bool control signals.
- Drives DefaultTeleopStateManager with bindings:
  - kill (B): any state -> stopped
  - run_toggle (A): stopped -> paused -> running -> paused -> ...
  - reset (thumbstick click): emit reset without changing state

Non-control demo plumbing (printing/observation retargeter) lives in
``teleop_controls_simple_helper.py``.
"""

import argparse
import sys
import time
from typing import Dict

from isaacteleop.cloudxr import CloudXRLauncher
from isaacteleop.retargeting_engine.deviceio_source_nodes import (
    HeadSource,
    HandsSource,
    ControllersSource,
)
from isaacteleop.retargeting_engine.interface.execution_events import ExecutionState
from isaacteleop.retargeting_engine.interface.retargeter_core_types import (
    GraphExecutable,
)
from isaacteleop.retargeting_engine.tensor_types import ControllerInputIndex
from isaacteleop.teleop_session_manager import (
    TeleopSession,
    TeleopSessionConfig,
    create_bool_selector,
    DefaultTeleopStateManager,
)

from .teleop_controls_simple_helper import (
    build_observation_pipeline,
    print_frame,
    print_header,
)


def _build_control_signals(
    controllers: ControllersSource,
) -> Dict[str, GraphExecutable]:
    left_selector = controllers.output(ControllersSource.LEFT)
    return {
        "kill_signal": create_bool_selector(
            left_selector,
            name="kill_signal_selector",
            selector_fn=lambda selected: selected[ControllerInputIndex.SECONDARY_CLICK],
        ),
        "run_toggle_signal": create_bool_selector(
            left_selector,
            name="run_toggle_signal_selector",
            selector_fn=lambda selected: selected[ControllerInputIndex.PRIMARY_CLICK],
        ),
        "reset_signal": create_bool_selector(
            left_selector,
            name="reset_signal_selector",
            selector_fn=lambda selected: selected[
                ControllerInputIndex.THUMBSTICK_CLICK
            ],
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args()

    with CloudXRLauncher.launch_context(args):
        head = HeadSource(name="head")
        hands = HandsSource(name="hands")
        controllers = ControllersSource(name="controllers")

        main_pipeline = build_observation_pipeline(head, hands, controllers)

        control_signals = _build_control_signals(controllers)
        teleop_manager = DefaultTeleopStateManager(name="teleop_manager")
        teleop_control_pipeline = teleop_manager.connect(
            {
                teleop_manager.INPUT_KILL: control_signals["kill_signal"].output(
                    "value"
                ),
                teleop_manager.INPUT_RUN_TOGGLE: control_signals[
                    "run_toggle_signal"
                ].output("value"),
                teleop_manager.INPUT_RESET: control_signals["reset_signal"].output(
                    "value"
                ),
            }
        )

        config = TeleopSessionConfig(
            app_name="TeleopControlsSimpleExample",
            pipeline=main_pipeline,
            teleop_control_pipeline=teleop_control_pipeline,
        )

        with TeleopSession(config) as session:
            print_header()

            # Example high-level hook: a caller can gate robot power/control using context.
            robot_enabled: bool | None = None

            while True:
                outputs = session.step()
                context = session.last_context
                if context is not None:
                    enabled_now = (
                        context.execution_events.execution_state
                        != ExecutionState.STOPPED
                    )
                    if robot_enabled is None or enabled_now != robot_enabled:
                        robot_enabled = enabled_now
                        print(f"[high-level] robot_enabled={robot_enabled}")
                    if context.execution_events.reset:
                        print("[high-level] reset pulse received")

                if session.frame_count % 30 == 0:
                    print_frame(outputs, session.get_elapsed_time())
                time.sleep(0.016)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)

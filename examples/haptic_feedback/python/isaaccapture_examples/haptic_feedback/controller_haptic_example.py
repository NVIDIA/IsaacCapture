# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal motion-controller haptic feedback example.

Pull a controller's trigger and that **same** controller rumbles. This is the
smallest end-to-end wiring of the Isaac Teleop device-output path, meant as a
reference for integrators (Isaac Lab, Isaac ROS, ...):

::

    ControllersSource (input)
            |  trigger value
            v
    TriggerToTactile -> TactileVectorToControllerPulse -> HapticSink (IDeviceIOSink)
                                                              |
                                          (after the graph)   v
                                  TeleopSession.flush_to_device(session)
                                          -> ControllerHapticDevice.flush
                                          -> ControllerTracker.apply_haptic_feedback

Key points for integrators:

* The haptic device is a *sink*: register it with
  ``TeleopSessionConfig(sinks=[...])``. The session runs it each frame after the
  main pipeline and then flushes it to hardware.
* ``ControllerHapticDevice`` reuses the **same**
  :class:`~isaaccapture.deviceio_trackers.ControllerTracker` as
  ``ControllersSource`` (pass ``controllers.get_tracker()``), so the session
  creates a single controller tracker and there is no action-set contention.
* Swap ``TriggerToTactile`` for any retargeter that emits a ``TactileVector``
  (e.g. an Isaac Lab ``ContactSensor`` fetch) to drive rumble from sim contact
  instead of a trigger -- the rest of the graph is unchanged.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from isaaccapture.haptic_devices.controller import ControllerHapticDevice
from isaaccapture.retargeters.tactile_retargeters import TactileVectorToControllerPulse
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    ControllersSource,
    HapticSink,
)
from isaaccapture.retargeting_engine.interface import BaseRetargeter, OutputCombiner
from isaaccapture.retargeting_engine.interface.retargeter_core_types import (
    ComputeContext,
    RetargeterIO,
    RetargeterIOType,
)
from isaaccapture.retargeting_engine.interface.tensor_group_type import OptionalType
from isaaccapture.retargeting_engine.tensor_types import (
    ControllerHapticPulseField,
    ControllerInput,
    ControllerInputIndex,
    TactileVector,
)
from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.teleop_session_manager import TeleopSession, TeleopSessionConfig


APP_NAME = "ControllerHapticFeedbackExample"
FPS = 60.0  # demo loop rate; the retargeting pipeline runs once per frame


class TriggerToTactile(BaseRetargeter):
    """Stream a controller's trigger value (0..1) as a ``TactileVector(1)``.

    Stands in for a sim-side tactile source: it has the same shape as an
    Isaac Lab ``ContactSensor`` fetch (one scalar), so the downstream graph
    cannot tell whether the signal came from a sim contact or a real trigger.
    """

    INPUT_CONTROLLER = "controller"
    OUTPUT_TACTILE = "tactile"

    def input_spec(self) -> RetargeterIOType:
        return {self.INPUT_CONTROLLER: OptionalType(ControllerInput())}

    def output_spec(self) -> RetargeterIOType:
        return {self.OUTPUT_TACTILE: TactileVector(1)}

    def _compute_fn(
        self, inputs: RetargeterIO, outputs: RetargeterIO, context: ComputeContext
    ) -> None:
        controller = inputs[self.INPUT_CONTROLLER]
        trigger = (
            0.0
            if controller.is_none
            else float(controller[ControllerInputIndex.TRIGGER_VALUE])
        )
        outputs[self.OUTPUT_TACTILE][0] = np.array([trigger], dtype=np.float32)


def _bar(value: float, width: int = 10) -> str:
    """Fixed-width ASCII bar for a value in [0, 1]."""
    filled = round(max(0.0, min(1.0, value)) * width)
    return "█" * filled + "░" * (width - filled)


def _scalar(result: dict, key: str) -> float:
    """First scalar of a ``TactileVector(1)`` output (0.0 if absent)."""
    group = result.get(key)
    if group is None or group.is_none:
        return 0.0
    return float(np.asarray(group[0]).ravel()[0])


def _amplitude(result: dict, key: str) -> float:
    """Amplitude field of a ``ControllerHapticPulse`` output (0.0 if absent)."""
    group = result.get(key)
    if group is None or group.is_none:
        return 0.0
    return float(np.asarray(group[0])[ControllerHapticPulseField.AMPLITUDE])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args()

    # 1. Input source (owns the ControllerTracker).
    controllers = ControllersSource("controllers")

    # 2. Haptic output device + sink. Reuse the ControllersSource tracker so the
    #    session creates a single controller tracker (no action-set contention).
    device = ControllerHapticDevice(controllers.get_tracker())
    sink = HapticSink("haptic_sink", device)

    # 3. Per hand: trigger value -> TactileVector -> ControllerHapticPulse.
    #    Each trigger drives its own controller (left -> left, right -> right).
    sink_inputs = {}
    monitoring = {}
    for side, controller_output in (
        ("left", controllers.output(ControllersSource.LEFT)),
        ("right", controllers.output(ControllersSource.RIGHT)),
    ):
        trigger = TriggerToTactile(f"{side}_trigger").connect(
            {TriggerToTactile.INPUT_CONTROLLER: controller_output}
        )
        pulse = TactileVectorToControllerPulse(f"{side}_pulse", num_taxels=1).connect(
            {
                TactileVectorToControllerPulse.INPUT_TACTILE: trigger.output(
                    TriggerToTactile.OUTPUT_TACTILE
                )
            }
        )
        sink_inputs[side] = pulse.output(TactileVectorToControllerPulse.OUTPUT_PULSE)
        monitoring[f"trigger_{side}"] = trigger.output(TriggerToTactile.OUTPUT_TACTILE)
        monitoring[f"haptic_{side}"] = sink_inputs[side]

    # 4. The main pipeline only carries the values we print each frame; the sink
    #    is registered separately and flushed to hardware by the session.
    config = TeleopSessionConfig(
        app_name=APP_NAME,
        pipeline=OutputCombiner(monitoring),
        sinks=[sink.connect(sink_inputs)],
    )

    print("Controller haptic feedback -- pull a trigger to rumble that controller.")
    print("Press Ctrl+C to exit.\n")

    frame_period_s = 1.0 / FPS
    with CloudXRLauncher.launch_context(args):
        with TeleopSession(config) as session:
            while True:
                result = session.step()
                trig_l, trig_r = (
                    _scalar(result, "trigger_left"),
                    _scalar(result, "trigger_right"),
                )
                hap_l, hap_r = (
                    _amplitude(result, "haptic_left"),
                    _amplitude(result, "haptic_right"),
                )
                line = (
                    f"L trig {_bar(trig_l)} {trig_l:.2f} -> L rumble {_bar(hap_l)} {hap_l:.2f}"
                    "   |   "
                    f"R trig {_bar(trig_r)} {trig_r:.2f} -> R rumble {_bar(hap_r)} {hap_r:.2f}"
                )
                print(f"\r{line:<96}", end="", flush=True)
                time.sleep(frame_period_s)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")

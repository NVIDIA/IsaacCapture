# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal haptic-glove example: feel each finger pinch toward the thumb.

Reads XR hand-tracking joints, turns each fingertip's distance to the thumb
tip into a per-finger vibration intensity, and sends those per-finger powers
to a haptic glove running in a separate plugin process. It is the glove
counterpart to ``controller_haptic_example.py`` and the cross-process
reference for the Isaac Teleop device-output path:

::

    HandsSource (input)
            |  hand joints
            v
    PinchProximityToTactile -> TactileVectorToFingerPower -> HapticSink (IDeviceIOSink)
                                                                 |
                                             (after the graph)   v
                                     TeleopSession flushes the sink to the device
                                          -> PushTensorHapticDevice.flush
                                          -> HapticCommand pushed over XR_NVX1_push_tensor
                                          -> glove plugin process (e.g. Manus)

Key points for integrators:

* The glove is a *cross-process* device. ``haptic_glove_device(...)`` returns a
  :class:`~isaaccapture.haptic_devices.push_tensor.PushTensorHapticDevice` that
  serialises each frame's per-finger powers into a vendor-neutral
  ``HapticCommand`` and pushes it on the ``collection_id`` below. A glove
  plugin (here the Manus plugin) reads the same collection and drives the
  hardware. To target a different glove, change ``COLLECTION_ID`` and run that
  vendor's plugin -- nothing else here changes.
* The device is a *sink*: register it with ``TeleopSessionConfig(sinks=[...])``
  and the session flushes it to the device each frame after the main pipeline.
* The mapping is split like the controller example: a thin
  ``PinchProximityToTactile`` adapter emits a vendor-neutral ``TactileVector``,
  and the library retargeter
  :class:`~isaaccapture.retargeters.tactile_retargeters.TactileVectorToFingerPower`
  shapes it (gain / deadband / saturation) into the ``FingerPowerVector`` every
  glove accepts. Swap the adapter for an Isaac Lab ``ContactSensor`` fetch (or
  use ``TactileHeatmapToFingerPower``) to drive the glove from sim contact.
"""

from __future__ import annotations

import time

import numpy as np

from isaaccapture.haptic_devices.glove import haptic_glove_device
from isaaccapture.retargeters.tactile_retargeters import TactileVectorToFingerPower
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    HandsSource,
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
    FingerIndex,
    HandInput,
    HandInputIndex,
    HandJointIndex,
    NUM_HAPTIC_FINGERS,
    TactileVector,
)
from isaaccapture.teleop_session_manager import TeleopSession, TeleopSessionConfig


APP_NAME = "HandPinchHapticExample"
FPS = 60.0  # demo loop rate; the retargeting pipeline runs once per frame

# Haptic-glove plugin's tensor-collection id. The Manus plugin uses this exact
# string (src/plugins/manus/core/inc/manus/manus_glove_collection.hpp); for a
# different vendor, point this at that plugin's collection_id.
COLLECTION_ID = "manus_glove_haptic"

# Pinch ramp: vibration starts as a fingertip comes within MAX_DISTANCE_M of
# the thumb tip and reaches full power at MIN_DISTANCE_M.
MAX_DISTANCE_M = 0.10
MIN_DISTANCE_M = 0.005

# FingerPowerVector channel -> fingertip joint, for the four non-thumb fingers.
# The thumb channel has no pinch-to-thumb distance, so it stays at zero.
_FINGER_TIP_JOINTS = {
    FingerIndex.INDEX: HandJointIndex.INDEX_TIP,
    FingerIndex.MIDDLE: HandJointIndex.MIDDLE_TIP,
    FingerIndex.RING: HandJointIndex.RING_TIP,
    FingerIndex.PINKY: HandJointIndex.LITTLE_TIP,
}

_FINGER_LABELS = ["Th", "Ix", "Md", "Rg", "Pk"]


class PinchProximityToTactile(BaseRetargeter):
    """Per hand: ``distance(thumb_tip, finger_tip)`` -> raw per-finger proximity.

    Emits a ``TactileVector(5)`` (order Thumb..Pinky) where each non-thumb
    channel is ``(MAX_DISTANCE_M - distance) / span`` -- ~1 when the fingertip
    touches the thumb, ~0 at ``MAX_DISTANCE_M``, and out of range otherwise.
    The downstream ``TactileVectorToFingerPower`` clamps and shapes it; this
    adapter just reads the sensor, like ``TriggerToTactile`` in the controller
    example. When the hand is not tracked the output is all zeros.
    """

    INPUT_HAND = "hand"
    OUTPUT_TACTILE = "tactile"

    def input_spec(self) -> RetargeterIOType:
        return {self.INPUT_HAND: OptionalType(HandInput())}

    def output_spec(self) -> RetargeterIOType:
        return {self.OUTPUT_TACTILE: TactileVector(NUM_HAPTIC_FINGERS)}

    def _compute_fn(
        self, inputs: RetargeterIO, outputs: RetargeterIO, context: ComputeContext
    ) -> None:
        proximity = np.zeros(NUM_HAPTIC_FINGERS, dtype=np.float32)
        hand = inputs[self.INPUT_HAND]

        if not hand.is_none:
            joint_positions = np.asarray(hand[HandInputIndex.JOINT_POSITIONS])
            joint_valid = np.asarray(hand[HandInputIndex.JOINT_VALID])
            if bool(joint_valid[HandJointIndex.THUMB_TIP]):
                thumb_tip = joint_positions[HandJointIndex.THUMB_TIP]
                span = MAX_DISTANCE_M - MIN_DISTANCE_M
                for finger, tip_joint in _FINGER_TIP_JOINTS.items():
                    if bool(joint_valid[tip_joint]):
                        distance = float(
                            np.linalg.norm(joint_positions[tip_joint] - thumb_tip)
                        )
                        proximity[finger] = (MAX_DISTANCE_M - distance) / span

        outputs[self.OUTPUT_TACTILE][0] = proximity


def _bar(value: float, width: int = 4) -> str:
    """Fixed-width ASCII bar for a value in [0, 1]."""
    filled = round(max(0.0, min(1.0, value)) * width)
    return "█" * filled + "░" * (width - filled)


def _powers(result: dict, key: str) -> np.ndarray:
    """The 5 finger powers of a ``FingerPowerVector`` output (zeros if absent)."""
    group = result.get(key)
    if group is None or group.is_none:
        return np.zeros(NUM_HAPTIC_FINGERS, dtype=np.float32)
    return np.asarray(group[0], dtype=np.float32).ravel()


def _row(label: str, powers: np.ndarray) -> str:
    """One-hand summary: a per-finger bar for each of the 5 channels."""
    cols = "  ".join(
        f"{_FINGER_LABELS[i]} {_bar(powers[i])}" for i in range(NUM_HAPTIC_FINGERS)
    )
    return f"{label}  {cols}"


def main() -> None:
    # 1. Input source.
    hands = HandsSource("hands")

    # 2. Cross-process glove device + sink. The device pushes per-finger powers
    #    to the glove plugin listening on COLLECTION_ID.
    device = haptic_glove_device(COLLECTION_ID)
    sink = HapticSink("haptic_sink", device)

    # 3. Per hand: hand joints -> TactileVector (thin adapter) -> FingerPowerVector
    #    (library mapper). Each hand drives its own glove (left -> "left", etc).
    sink_inputs = {}
    monitoring = {}
    for side, hand_output in (
        ("left", hands.output(HandsSource.LEFT)),
        ("right", hands.output(HandsSource.RIGHT)),
    ):
        proximity = PinchProximityToTactile(f"{side}_pinch").connect(
            {PinchProximityToTactile.INPUT_HAND: hand_output}
        )
        powers = TactileVectorToFingerPower(
            f"{side}_powers",
            num_taxels=NUM_HAPTIC_FINGERS,
            num_fingers=NUM_HAPTIC_FINGERS,
        ).connect(
            {
                TactileVectorToFingerPower.INPUT_TACTILE: proximity.output(
                    PinchProximityToTactile.OUTPUT_TACTILE
                )
            }
        )
        sink_inputs[side] = powers.output(TactileVectorToFingerPower.OUTPUT_POWERS)
        monitoring[f"powers_{side}"] = sink_inputs[side]

    # 4. The main pipeline only carries the values we print; the sink is
    #    registered separately and flushed to the glove by the session.
    config = TeleopSessionConfig(
        app_name=APP_NAME,
        pipeline=OutputCombiner(monitoring),
        sinks=[sink.connect(sink_inputs)],
    )

    print(
        "Haptic glove pinch demo -- bring a fingertip toward the thumb to vibrate it."
    )
    print(
        f"Pushing HapticCommands on collection '{COLLECTION_ID}'. Press Ctrl+C to exit.\n"
    )

    frame_period_s = 1.0 / FPS
    with TeleopSession(config) as session:
        while True:
            result = session.step()
            line = f"{_row('L', _powers(result, 'powers_left'))}   |   {_row('R', _powers(result, 'powers_right'))}"
            print(f"\r{line:<104}", end="", flush=True)
            time.sleep(frame_period_s)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")

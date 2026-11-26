# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic standard hand inputs, independent of any glove plugin."""

import numpy as np
from scipy.spatial.transform import Rotation

from isaaccapture.retargeting_engine.interface import (
    ComputeContext,
    ExecutionEvents,
    ExecutionState,
    OptionalTensorGroup,
    TensorGroup,
)
from isaaccapture.retargeting_engine.interface.retargeter_core_types import GraphTime
from isaaccapture.retargeting_engine.tensor_types import HandInputIndex


def human_sample(side, curl=0.0):
    points = np.zeros((26, 3))
    points[0] = (0, 0, -0.03)
    points[2] = (0.03, 0, -0.015)
    for index in range(3, 6):
        points[index] = points[index - 1] + Rotation.from_euler("Y", -0.6 - curl).apply(
            (0, 0, -0.025)
        )
    for finger, base_x in enumerate((0.03, 0.01, -0.01, -0.03)):
        start = 6 + 5 * finger
        points[start] = (base_x, 0, -0.02)
        for bone, length in enumerate((0.025, 0.035, 0.025, 0.02)):
            points[start + bone + 1] = points[start + bone] + Rotation.from_euler(
                "X", -curl * min(bone, 2)
            ).apply((0, 0, -length))
    if side == "right":
        points[:, 0] *= -1
    return {
        "joints": [
            {"position": point, "orientation": (0, 0, 0, 1), "radius": 0.008}
            for point in points
        ]
    }


def _feed_samples(retargeter, side, samples):
    name, spec = next(iter(retargeter.input_spec().items()))
    assert name == f"hand_{side}"
    inputs = {name: OptionalTensorGroup(spec)}
    outputs = {
        key: TensorGroup(value) for key, value in retargeter.output_spec().items()
    }
    context = ComputeContext(
        graph_time=GraphTime(sim_time_ns=0, real_time_ns=0),
        execution_events=ExecutionEvents(
            reset=False, execution_state=ExecutionState.RUNNING
        ),
    )
    results = []
    for sample in samples:
        joints = sample["joints"]
        group = inputs[name]
        group[HandInputIndex.JOINT_POSITIONS] = np.asarray(
            [j["position"] for j in joints], dtype=np.float32
        )
        group[HandInputIndex.JOINT_ORIENTATIONS] = np.asarray(
            [j["orientation"] for j in joints], dtype=np.float32
        )
        group[HandInputIndex.JOINT_RADII] = np.asarray(
            [j["radius"] for j in joints], dtype=np.float32
        )
        group[HandInputIndex.JOINT_VALID] = np.ones(26, dtype=np.uint8)
        retargeter.compute(inputs, outputs, context)
        result = np.asarray([float(value) for value in outputs["hand_joints"]])
        assert np.all(np.isfinite(result))
        results.append(result)
    return results

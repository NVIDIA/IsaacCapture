# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Feed production C++ glove geometry through the standard Python HandInput contract."""

import numpy as np
import pytest
from importlib.resources import files

from isaaccapture.retargeters import DexHandRetargeter
from test_realhand_example import _EXAMPLE
from isaaccapture.retargeters.realhand.profiles import get_realhand_profile
from isaaccapture.retargeting_engine.interface import (
    ComputeContext,
    ExecutionEvents,
    ExecutionState,
    OptionalTensorGroup,
    TensorGroup,
)
from isaaccapture.retargeting_engine.interface.retargeter_core_types import GraphTime
from isaaccapture.retargeting_engine.tensor_types import HandInputIndex


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


@pytest.mark.parametrize("model", ("l6", "o6", "l20"))
@pytest.mark.parametrize("side", ("left", "right"))
def test_production_glove_skeleton_retargets_every_hand(
    model, side, skeleton_samples, caplog
):
    retargeter = DexHandRetargeter(
        _EXAMPLE.dex_hand_config(model, side),
        name=f"ffg_{model}_{side}",
    )
    results = _feed_samples(retargeter, side, skeleton_samples[side])
    assert np.linalg.norm(results[-1] - results[0]) > 0.5
    assert np.linalg.norm(results[1] - results[0]) > 0.1
    for result in results:
        specs = get_realhand_profile(model).joints(side)
        # dex-retargeting expands optimizer limits by 1e-3 rad internally.
        assert np.all(result >= np.asarray([s.lower for s in specs]) - 1.01e-3)
        assert np.all(result <= np.asarray([s.upper for s in specs]) + 1.01e-3)
    assert not [r for r in caplog.records if r.levelname == "ERROR"]


@pytest.mark.parametrize("side", ("left", "right"))
def test_production_glove_skeleton_retargets_wuji(side, skeleton_samples):
    pytest.importorskip("wuji_sdk")
    from isaaccapture.retargeters import WujiHandRetargeter, WujiHandRetargeterConfig

    retargeter = WujiHandRetargeter(
        WujiHandRetargeterConfig(model="wuji_hand_2", hand_side=side),
        name=f"ffg_wuji_{side}",
    )
    results = _feed_samples(retargeter, side, skeleton_samples[side])
    assert results[0].shape == (20,)
    assert np.linalg.norm(results[-1] - results[0]) > 0.1


def test_production_glove_skeleton_retargets_sharpa(skeleton_samples):
    pytest.importorskip("pink")
    from isaaccapture.retargeters import (
        SharpaHandRetargeter,
        SharpaHandRetargeterConfig,
    )

    asset = (
        files("robotic_grounding")
        / "assets/xmls/sharpawave/right_sharpawave_nomesh.xml"
    )
    retargeter = SharpaHandRetargeter(
        SharpaHandRetargeterConfig(
            robot_asset_path=str(asset), hand_side="right", max_iter=50, frequency=200.0
        ),
        name="ffg_sharpa_right",
    )
    results = _feed_samples(retargeter, "right", skeleton_samples["right"])
    assert results[0].shape == (22,)
    assert np.linalg.norm(results[-1] - results[0]) > 0.1

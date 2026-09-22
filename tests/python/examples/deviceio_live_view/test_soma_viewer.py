# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import numpy as np
import pytest
from isaaccapture.deviceio_trackers import (
    SomaBodyJointPosesV0Tracker,
    SomaBodyJointRotationsV0Tracker,
)
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    FullBodySource,
    SomaBodyRepresentation,
    SomaBodySource,
)
from isaaccapture.retargeting_engine.deviceio_source_nodes import soma_body_source
from isaaccapture.retargeting_engine.interface.base_retargeter import _make_output_group
from isaaccapture.retargeting_engine.interface.output_combiner import OutputCombiner
from isaaccapture.retargeting_engine.interface.tensor_group import TensorGroup
from isaaccapture.retargeting_engine.interface.tensor_group_type import OptionalType
from isaaccapture.retargeting_engine.tensor_types import (
    SomaBodyInput,
    SomaBodyInputIndex,
)
from isaaccapture.schema import (
    Point,
    SomaBodyJointPoseArrayV0,
    SomaBodyJointPosesV0,
    SomaBodyJointRotationArrayV0,
    SomaBodyJointRotationsV0,
)
from isaaccapture_examples.deviceio_live_view import live_deviceio, soma_body
from isaaccapture_examples.deviceio_live_view.body_pipeline import (
    BodySchema,
    BodyViewLayout,
    create_body_view_pipeline,
)
from isaaccapture_examples.deviceio_live_view.deviceio_pipeline import (
    build_all_human_pipeline,
)
from isaaccapture_examples.deviceio_live_view.deviceio_viser import (
    HumanDeviceIOViz,
    _valid_bone_segments,
)
from isaaccapture_examples.deviceio_live_view.full_body_pose import (
    FULL_BODY_POSE_LAYOUT,
)


def fake_layer():
    layer = MagicMock()
    layer.public_joint_names = ("Root", *(f"Joint{index}" for index in range(77)))
    layer.output_joint_parent_ids = np.array([0, 0, *range(1, 77)])
    return layer


def soma_joint_rotations(quaternions, translation):
    joints = SomaBodyJointRotationArrayV0()
    joints.rotations[:] = quaternions
    joints.is_valid[:] = 1
    return SomaBodyJointRotationsV0(joints, Point(*translation), True)


def soma_joint_poses(positions, orientations):
    joints = SomaBodyJointPoseArrayV0()
    joints.positions[:] = positions
    joints.orientations[:] = orientations
    joints.is_valid[:] = 1
    return SomaBodyJointPosesV0(joints)


def fake_viz(layout: BodyViewLayout = FULL_BODY_POSE_LAYOUT):
    server = MagicMock()
    for name in ("add_point_cloud", "add_line_segments", "add_frame"):
        getattr(server.scene, name).side_effect = lambda **kwargs: MagicMock(**kwargs)
    return HumanDeviceIOViz(server, MagicMock(), layout)


def test_default_pipeline_keeps_full_body_source():
    pipeline = build_all_human_pipeline()
    sources = {source.name: source for source in pipeline.get_leaf_nodes()}
    assert set(sources) == {"hands", "head", "controllers", "body"}
    assert type(sources["body"]) is FullBodySource
    assert set(pipeline.output_types()) == {
        "hand_left",
        "hand_right",
        "head",
        "controller_left",
        "controller_right",
        "body",
    }


@pytest.mark.parametrize("explicit", [False, True])
def test_soma_pipeline_replaces_only_body_source(monkeypatch, tmp_path, explicit):
    monkeypatch.setattr(soma_body, "create_layer", lambda _: fake_layer())
    tracker_factory = MagicMock(wraps=SomaBodyJointRotationsV0Tracker)
    monkeypatch.setattr(
        soma_body_source, "SomaBodyJointRotationsV0Tracker", tracker_factory
    )
    body = create_body_view_pipeline(
        tmp_path,
        body_schema=BodySchema.SOMA if explicit else None,
        soma_collection_id="vendor_body",
    )
    pipeline = build_all_human_pipeline(body=body)
    sources = {source.name: source for source in pipeline.get_leaf_nodes()}
    assert set(sources) == {"hands", "head", "controllers", "body"}
    assert type(sources["body"]) is SomaBodySource
    assert isinstance(sources["body"].get_tracker(), SomaBodyJointRotationsV0Tracker)
    tracker_factory.assert_called_once_with("vendor_body")
    assert len(body.layout.joint_names) == 77
    assert len(body.layout.bones) == 76


def test_soma_joint_pose_pipeline_selects_direct_tracker(monkeypatch, tmp_path):
    monkeypatch.setattr(soma_body, "create_layer", lambda _: fake_layer())
    tracker_factory = MagicMock(wraps=SomaBodyJointPosesV0Tracker)
    monkeypatch.setattr(
        soma_body_source, "SomaBodyJointPosesV0Tracker", tracker_factory
    )
    body = create_body_view_pipeline(
        tmp_path,
        body_schema=BodySchema.SOMA,
        soma_collection_id="vendor_body",
        soma_body_representation=SomaBodyRepresentation.JOINT_POSES,
    )
    pipeline = build_all_human_pipeline(body=body)
    source = next(node for node in pipeline.get_leaf_nodes() if node.name == "body")
    assert isinstance(source.get_tracker(), SomaBodyJointPosesV0Tracker)
    tracker_factory.assert_called_once_with("vendor_body")


def test_explicit_full_body_selection_does_not_load_soma(monkeypatch, tmp_path):
    create = MagicMock(side_effect=AssertionError("SOMA must remain optional"))
    monkeypatch.setattr(soma_body, "create_layer", create)
    body = create_body_view_pipeline(tmp_path, body_schema=BodySchema.FULL_BODY_POSE)
    pipeline = build_all_human_pipeline(body=body)
    source = next(node for node in pipeline.get_leaf_nodes() if node.name == "body")
    assert type(source) is FullBodySource
    create.assert_not_called()


def test_soma_evaluation_runs_once_per_step_downstream(monkeypatch, tmp_path):
    monkeypatch.setattr(soma_body, "create_layer", lambda _: fake_layer())
    body = create_body_view_pipeline(tmp_path)
    selector = body.output
    pipeline = build_all_human_pipeline(body=body)
    source = next(node for node in pipeline.get_leaf_nodes() if node.name == "body")
    raw = soma_joint_rotations(np.tile([0, 0, 0, 1], (77, 1)), [1, 2, 3])
    source._tracker = MagicMock()
    source._tracker.get_data.return_value = raw
    evaluate = MagicMock(
        return_value=(
            np.zeros((77, 3), dtype=np.float32),
            np.tile([0, 0, 0, 1], (77, 1)).astype(np.float32),
            np.ones(77, dtype=np.uint8),
        )
    )
    monkeypatch.setattr(source._evaluator, "evaluate", evaluate)
    inputs = source.poll_tracker(object())
    combined = OutputCombiner({"view": selector, "other_consumer": selector})
    result = combined.execute_pipeline({source.name: inputs})
    evaluate.assert_called_once_with(raw)
    assert result["view"] is result["other_consumer"]
    combined.execute_pipeline({source.name: inputs})
    assert evaluate.call_count == 2


def test_viewer_uses_selected_native_body_layout_and_hides_missing_devices():
    pipeline = build_all_human_pipeline()
    result = {
        name: _make_output_group(kind) for name, kind in pipeline.output_types().items()
    }
    for group in result.values():
        group.set_none()

    layout = BodyViewLayout(
        joint_names=tuple(f"Joint{index}" for index in range(77)),
        bones=((0, 1), (1, 2)),
        positions_index=int(SomaBodyInputIndex.JOINT_POSITIONS),
        valid_index=int(SomaBodyInputIndex.JOINT_VALID),
    )
    body = _make_output_group(OptionalType(SomaBodyInput()))
    positions = np.arange(231, dtype=np.float32).reshape(77, 3) / 100
    valid = np.zeros(77, dtype=np.uint8)
    valid[:3] = (1, 1, 0)
    body[SomaBodyInputIndex.JOINT_POSITIONS] = positions
    body[SomaBodyInputIndex.JOINT_ORIENTATIONS] = np.tile(
        np.array([0, 0, 0, 1], dtype=np.float32), (77, 1)
    )
    body[SomaBodyInputIndex.JOINT_VALID] = valid
    result["body"] = body

    viz = fake_viz(layout)
    active = viz.update(result)
    assert active["body_active"]
    assert active["body_joints"] == 2
    assert not any(
        active[key]
        for key in (
            "hand_left",
            "hand_right",
            "head",
            "controller_left",
            "controller_right",
        )
    )
    np.testing.assert_array_equal(viz.body.points.points, positions)
    np.testing.assert_array_equal(
        viz.body.bones.points, _valid_bone_segments(positions, valid, layout.bones)
    )
    assert len(viz.body.bones.points) == 1
    viz._ground.follow.assert_called_once()
    body.set_none()
    assert not viz.update(result)["body_active"]
    assert not viz.body.points.visible
    assert not viz.body.bones.visible


@pytest.mark.parametrize(
    "schema,use_assets,expected,collection",
    [
        (None, False, BodySchema.FULL_BODY_POSE, "soma_demo"),
        (None, True, BodySchema.SOMA, "soma_demo"),
        ("soma", True, BodySchema.SOMA, "vendor_body"),
        ("full-body-pose", True, BodySchema.FULL_BODY_POSE, "soma_demo"),
    ],
)
def test_live_cli_selects_source_and_launches_runtime(
    monkeypatch, tmp_path, schema, use_assets, expected, collection
):
    body = create_body_view_pipeline()
    select = MagicMock(return_value=body)
    monkeypatch.setattr(live_deviceio, "create_body_view_pipeline", select)
    pipeline = build_all_human_pipeline()
    build = MagicMock(return_value=pipeline)
    monkeypatch.setattr(live_deviceio, "build_all_human_pipeline", build)
    launch = MagicMock()
    launch.return_value.__enter__.return_value.owns_runtime = False
    monkeypatch.setattr(live_deviceio.CloudXRLauncher, "launch_context", launch)
    server = MagicMock()
    monkeypatch.setattr(live_deviceio.viser, "ViserServer", lambda **kwargs: server)
    session = MagicMock()
    session.__enter__.return_value = session
    session.step.side_effect = KeyboardInterrupt
    create_session = MagicMock(return_value=session)
    monkeypatch.setattr(live_deviceio, "TeleopSession", create_session)
    args = ["viewer"]
    if schema is not None:
        args += ["--body-schema", schema]
    if use_assets:
        args += ["--soma-data-root", str(tmp_path)]
    args += ["--soma-collection-id", collection]
    assert live_deviceio.main(args) == 0
    select.assert_called_once_with(
        tmp_path if use_assets else None,
        body_schema=expected,
        soma_collection_id=collection,
        soma_body_representation="joint-rotations",
    )
    build.assert_called_once_with(body=body)
    launch.assert_called_once()
    assert create_session.call_args.args[0].pipeline is pipeline


@pytest.mark.parametrize(
    "args,message",
    [
        (["--body-schema", "soma"], "requires --soma-data-root"),
        (["--body-schema", "unknown"], "invalid choice"),
        (
            [
                "--body-schema",
                "soma",
                "--soma-data-root",
                "/unused",
                "--soma-collection-id",
                "",
            ],
            "collection_id must not be empty",
        ),
    ],
)
def test_bad_body_selection_fails_before_launch(monkeypatch, capsys, args, message):
    monkeypatch.setattr(soma_body, "create_layer", lambda _: fake_layer())
    launch = MagicMock()
    monkeypatch.setattr(live_deviceio.CloudXRLauncher, "launch_context", launch)
    server = MagicMock()
    monkeypatch.setattr(live_deviceio.viser, "ViserServer", server)
    with pytest.raises(SystemExit) as error:
        live_deviceio.main(["viewer", *args])
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    launch.assert_not_called()
    server.assert_not_called()


def test_native_soma_pose_reaches_renderer(soma_assets):
    body = create_body_view_pipeline(soma_assets)
    pipeline = build_all_human_pipeline(body=body)
    source = next(node for node in pipeline.get_leaf_nodes() if node.name == "body")
    source._tracker = MagicMock()
    inputs = {}
    for node in pipeline.get_leaf_nodes():
        inputs[node.name] = {}
        for key, kind in node.input_spec().items():
            group = TensorGroup(kind)
            group[0] = None
            inputs[node.name][key] = group
    viz = fake_viz(body.layout)
    pose = soma_joint_rotations(np.tile([0, 0, 0, 1], (77, 1)), [0, 0, 0])
    source._tracker.get_data.return_value = pose
    inputs[source.name] = source.poll_tracker(object())
    result = pipeline.execute_pipeline(inputs)
    assert viz.update(result)["body_joints"] == 77
    expected_positions, _, _ = source._evaluator.evaluate(pose)
    np.testing.assert_allclose(viz.body.points.points, expected_positions)

    source._tracker.get_data.return_value = None
    inputs[source.name] = source.poll_tracker(object())
    assert not viz.update(pipeline.execute_pipeline(inputs))["body_active"]

    invalid = soma_joint_rotations(np.tile([0, 0, 0, 1], (77, 1)), [0, 0, 0])
    invalid.joint_rotations.is_valid[10] = 0
    source._tracker.get_data.return_value = invalid
    inputs[source.name] = source.poll_tracker(object())
    active = viz.update(pipeline.execute_pipeline(inputs))
    assert active["body_active"]
    assert 0 < active["body_joints"] < 77


def test_evaluated_soma_pose_reaches_renderer_without_fk(soma_assets):
    body = create_body_view_pipeline(
        soma_assets,
        body_schema=BodySchema.SOMA,
        soma_body_representation=SomaBodyRepresentation.JOINT_POSES,
    )
    pipeline = build_all_human_pipeline(body=body)
    source = next(node for node in pipeline.get_leaf_nodes() if node.name == "body")
    source._tracker = MagicMock()
    source._evaluator.evaluate = MagicMock(
        side_effect=AssertionError("joint poses must bypass FK")
    )
    inputs = {}
    for node in pipeline.get_leaf_nodes():
        inputs[node.name] = {}
        for key, kind in node.input_spec().items():
            group = TensorGroup(kind)
            group[0] = None
            inputs[node.name][key] = group
    positions = np.arange(231, dtype=np.float32).reshape(77, 3) / 100
    orientations = np.tile([0, 0, 0, 1], (77, 1)).astype(np.float32)
    source._tracker.get_data.return_value = soma_joint_poses(positions, orientations)
    inputs[source.name] = source.poll_tracker(object())

    viz = fake_viz(body.layout)
    assert viz.update(pipeline.execute_pipeline(inputs))["body_joints"] == 77
    np.testing.assert_array_equal(viz.body.points.points, positions)
    source._evaluator.evaluate.assert_not_called()

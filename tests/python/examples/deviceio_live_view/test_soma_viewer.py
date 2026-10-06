# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import numpy as np
import pytest
from isaaccapture.deviceio_trackers import (
    SomaBodyJointPosesTracker,
    SomaBodyJointRotationsTracker,
    SomaHandJointPosesTracker,
    SomaHandJointRotationsTracker,
)
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    FullBodySource,
    SomaBodyRepresentation,
    SomaBodySource,
    SomaHandRepresentation,
    SomaHandSource,
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
    SomaBodyJointPoseArray,
    SomaBodyJointPoses,
    SomaBodyJointRotationArray,
    SomaBodyJointRotations,
    SomaHandedness,
    SomaHandJointPoseArray,
    SomaHandJointPoses,
    SomaHandJointRotationArray,
    SomaHandJointRotations,
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
from isaaccapture_examples.deviceio_live_view.hand_pipeline import (
    HandSchema,
    HandViewLayout,
    create_hand_view_pipeline,
)
from isaaccapture_examples.deviceio_live_view.openxr_hand_pose import (
    OPENXR_HAND_LAYOUT,
)
from isaaccapture_examples.deviceio_live_view.soma_hand import SOMA_HAND_LAYOUT


def fake_layer():
    layer = MagicMock()
    layer.public_joint_names = ("Root", *(f"Joint{index}" for index in range(77)))
    layer.output_joint_parent_ids = np.array([0, 0, *range(1, 77)])
    return layer


def soma_joint_rotations(quaternions, translation):
    joints = SomaBodyJointRotationArray()
    joints.rotations[:] = quaternions
    joints.is_valid[:] = 1
    return SomaBodyJointRotations(joints, Point(*translation), True)


def soma_joint_poses(positions, orientations):
    joints = SomaBodyJointPoseArray()
    joints.positions[:] = positions
    joints.orientations[:] = orientations
    joints.is_valid[:] = 1
    return SomaBodyJointPoses(joints)


def soma_hand_joint_poses(positions, orientations, handedness):
    joints = SomaHandJointPoseArray()
    joints.positions[:] = positions
    joints.orientations[:] = orientations
    joints.is_valid[:] = 1
    return SomaHandJointPoses(joints, handedness)


def soma_hand_joint_rotations(rotations, translation, handedness):
    joints = SomaHandJointRotationArray()
    joints.rotations[:] = rotations
    joints.is_valid[:] = 1
    return SomaHandJointRotations(joints, Point(*translation), True, handedness)


def fake_viz(
    layout: BodyViewLayout = FULL_BODY_POSE_LAYOUT,
    hand_layout: HandViewLayout = OPENXR_HAND_LAYOUT,
):
    server = MagicMock()
    for name in ("add_point_cloud", "add_line_segments", "add_frame"):
        getattr(server.scene, name).side_effect = lambda **kwargs: MagicMock(**kwargs)
    return HumanDeviceIOViz(server, MagicMock(), layout, hand_layout)


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


def test_soma_pipeline_replaces_only_body_source(monkeypatch):
    monkeypatch.setattr(soma_body, "create_layer", fake_layer)
    tracker_factory = MagicMock(wraps=SomaBodyJointRotationsTracker)
    monkeypatch.setattr(
        soma_body_source, "SomaBodyJointRotationsTracker", tracker_factory
    )
    body = create_body_view_pipeline(
        body_schema=BodySchema.SOMA,
        soma_body_collection_id="vendor_body",
    )
    pipeline = build_all_human_pipeline(body=body)
    sources = {source.name: source for source in pipeline.get_leaf_nodes()}
    assert set(sources) == {"hands", "head", "controllers", "body"}
    assert type(sources["body"]) is SomaBodySource
    assert isinstance(sources["body"].get_tracker(), SomaBodyJointRotationsTracker)
    tracker_factory.assert_called_once_with("vendor_body")
    assert len(body.layout.joint_names) == 77
    assert len(body.layout.bones) == 76


def test_soma_joint_pose_pipeline_selects_direct_tracker(monkeypatch):
    create = MagicMock(side_effect=AssertionError("joint poses must not load SOMA"))
    monkeypatch.setattr(soma_body, "create_layer", create)
    tracker_factory = MagicMock(wraps=SomaBodyJointPosesTracker)
    monkeypatch.setattr(soma_body_source, "SomaBodyJointPosesTracker", tracker_factory)
    body = create_body_view_pipeline(
        body_schema=BodySchema.SOMA,
        soma_body_collection_id="vendor_body",
        soma_body_representation=SomaBodyRepresentation.JOINT_POSES,
    )
    pipeline = build_all_human_pipeline(body=body)
    source = next(node for node in pipeline.get_leaf_nodes() if node.name == "body")
    assert isinstance(source.get_tracker(), SomaBodyJointPosesTracker)
    tracker_factory.assert_called_once_with("vendor_body")
    create.assert_not_called()


def test_soma_joint_pose_hands_replace_only_openxr_hands():
    hands = create_hand_view_pipeline(
        hand_schema=HandSchema.SOMA,
        soma_left_collection_id="vendor.left",
        soma_right_collection_id="vendor.right",
        soma_hand_representation=SomaHandRepresentation.JOINT_POSES,
    )
    pipeline = build_all_human_pipeline(hands=hands)
    sources = {source.name: source for source in pipeline.get_leaf_nodes()}

    assert set(sources) == {
        "hand_left",
        "hand_right",
        "head",
        "controllers",
        "body",
    }
    assert isinstance(sources["hand_left"], SomaHandSource)
    assert isinstance(sources["hand_right"], SomaHandSource)
    assert isinstance(sources["hand_left"].get_tracker(), SomaHandJointPosesTracker)
    assert hands.layout is SOMA_HAND_LAYOUT


def test_soma_hand_pose_reaches_native_25_joint_renderer():
    hands = create_hand_view_pipeline(
        hand_schema=HandSchema.SOMA,
        soma_hand_representation=SomaHandRepresentation.JOINT_POSES,
    )
    pipeline = build_all_human_pipeline(hands=hands)
    sources = {source.name: source for source in pipeline.get_leaf_nodes()}
    for source in (sources["hand_left"], sources["hand_right"]):
        source._tracker = MagicMock()
    positions = np.arange(75, dtype=np.float32).reshape(25, 3) / 100
    orientations = np.tile([0, 0, 0, 1], (25, 1)).astype(np.float32)
    sources["hand_left"]._tracker.get_data.return_value = soma_hand_joint_poses(
        positions, orientations, SomaHandedness.LEFT
    )
    sources["hand_right"]._tracker.get_data.return_value = None

    inputs = {}
    for source in pipeline.get_leaf_nodes():
        inputs[source.name] = {}
        for key, kind in source.input_spec().items():
            group = TensorGroup(kind)
            group[0] = None
            inputs[source.name][key] = group
    inputs["hand_left"] = sources["hand_left"].poll_tracker(object())
    inputs["hand_right"] = sources["hand_right"].poll_tracker(object())
    result = pipeline.execute_pipeline(inputs)
    viz = fake_viz(hand_layout=hands.layout)
    active = viz.update(result)

    assert active["hand_left"]
    assert not active["hand_right"]
    np.testing.assert_array_equal(viz.hand_left.points.points, positions)
    assert len(viz.hand_left.bones.points) == len(SOMA_HAND_LAYOUT.bones)


def test_explicit_full_body_selection_does_not_load_soma(monkeypatch):
    create = MagicMock(side_effect=AssertionError("SOMA must remain optional"))
    monkeypatch.setattr(soma_body, "create_layer", create)
    body = create_body_view_pipeline(body_schema=BodySchema.FULL_BODY_POSE)
    pipeline = build_all_human_pipeline(body=body)
    source = next(node for node in pipeline.get_leaf_nodes() if node.name == "body")
    assert type(source) is FullBodySource
    create.assert_not_called()


def test_soma_evaluation_runs_once_per_step_downstream(monkeypatch):
    monkeypatch.setattr(soma_body, "create_layer", fake_layer)
    body = create_body_view_pipeline(body_schema=BodySchema.SOMA)
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
    "schema,expected,collection",
    [
        (None, BodySchema.FULL_BODY_POSE, "soma_body_demo"),
        ("soma", BodySchema.SOMA, "vendor_body"),
        ("full-body-pose", BodySchema.FULL_BODY_POSE, "soma_body_demo"),
    ],
)
def test_live_cli_selects_source_and_launches_runtime(
    monkeypatch, schema, expected, collection
):
    body = create_body_view_pipeline()
    hands = create_hand_view_pipeline()
    select = MagicMock(return_value=body)
    monkeypatch.setattr(live_deviceio, "create_body_view_pipeline", select)
    select_hands = MagicMock(return_value=hands)
    monkeypatch.setattr(live_deviceio, "create_hand_view_pipeline", select_hands)
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
    args += ["--soma-body-collection-id", collection]
    assert live_deviceio.main(args) == 0
    select.assert_called_once_with(
        body_schema=expected,
        soma_body_collection_id=collection,
        soma_body_representation="joint-rotations",
    )
    select_hands.assert_called_once_with(
        hand_schema="openxr-hand-pose",
        soma_left_collection_id="soma_hand_left_demo",
        soma_right_collection_id="soma_hand_right_demo",
        soma_hand_representation="joint-rotations",
    )
    build.assert_called_once_with(body=body, hands=hands)
    launch.assert_called_once()
    assert create_session.call_args.args[0].pipeline is pipeline


@pytest.mark.parametrize(
    "args,message",
    [
        (["--body-schema", "unknown"], "invalid choice"),
        (["--soma-data-root", "/unused"], "unrecognized arguments"),
        (
            [
                "--body-schema",
                "soma",
                "--soma-body-collection-id",
                "",
            ],
            "collection_id must not be empty",
        ),
    ],
)
def test_bad_body_selection_fails_before_launch(monkeypatch, capsys, args, message):
    monkeypatch.setattr(soma_body, "create_layer", fake_layer)
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


def test_native_soma_pose_reaches_renderer(monkeypatch, soma_assets):
    create_layer = soma_body.create_layer
    monkeypatch.setattr(soma_body, "create_layer", lambda: create_layer(soma_assets))
    body = create_body_view_pipeline(body_schema=BodySchema.SOMA)
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


def test_native_soma_hand_rotations_reach_renderer(monkeypatch, soma_assets):
    from isaaccapture_examples.deviceio_live_view import soma_hand

    create_layer = soma_hand.create_layer
    monkeypatch.setattr(
        soma_hand,
        "create_layer",
        lambda side: create_layer(side, soma_assets),
    )
    hands = create_hand_view_pipeline(hand_schema=HandSchema.SOMA)
    pipeline = build_all_human_pipeline(hands=hands)
    sources = {source.name: source for source in pipeline.get_leaf_nodes()}
    inputs = {}
    for node in pipeline.get_leaf_nodes():
        inputs[node.name] = {}
        for key, kind in node.input_spec().items():
            group = TensorGroup(kind)
            group[0] = None
            inputs[node.name][key] = group

    identity = np.tile([0, 0, 0, 1], (25, 1)).astype(np.float32)
    for name, handedness in (
        ("hand_left", SomaHandedness.LEFT),
        ("hand_right", SomaHandedness.RIGHT),
    ):
        source = sources[name]
        assert isinstance(source.get_tracker(), SomaHandJointRotationsTracker)
        source._tracker = MagicMock()
        source._tracker.get_data.return_value = soma_hand_joint_rotations(
            identity, [0, 0, 0], handedness
        )
        inputs[name] = source.poll_tracker(object())

    viz = fake_viz(hand_layout=hands.layout)
    active = viz.update(pipeline.execute_pipeline(inputs))
    assert active["hand_left"]
    assert active["hand_right"]
    assert viz.hand_left.points.points.shape == (25, 3)
    assert viz.hand_right.points.points.shape == (25, 3)


def test_evaluated_soma_pose_reaches_renderer_without_fk(monkeypatch):
    create = MagicMock(side_effect=AssertionError("joint poses must not load SOMA"))
    monkeypatch.setattr(soma_body, "create_layer", create)
    body = create_body_view_pipeline(
        body_schema=BodySchema.SOMA,
        soma_body_representation=SomaBodyRepresentation.JOINT_POSES,
    )
    pipeline = build_all_human_pipeline(body=body)
    source = next(node for node in pipeline.get_leaf_nodes() if node.name == "body")
    source._tracker = MagicMock()
    assert source._evaluator is None
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
    create.assert_not_called()

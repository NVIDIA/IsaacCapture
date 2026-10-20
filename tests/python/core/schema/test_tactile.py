# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaccapture.schema import (
    ContentDescriptor,
    IsaacCapturePoseTactileBinding,
    OpenXRHandTactileBinding,
    Point,
    Pose,
    ProfileHandling,
    Quaternion,
    TACTILE_ISAACCAPTURE_POSE_PROFILE_ID,
    TACTILE_OPENXR_HAND_PROFILE_ID,
    TACTILE_URDF_PROFILE_ID,
    TactileBindingKind,
    TactileBindingProfile,
    TactileData,
    TactileHandedness,
    TactileMeasurement,
    TactileMetadata,
    TactileNode,
    TactileQuantity,
    TactileSkinWeight,
    TactileTaxel,
    TactileUnit,
    TactileValueFrame,
    UrdfEntityKind,
    UrdfNodeBinding,
    UrdfTactileBinding,
    make_tactile_binding_profile,
    validate_tactile_data,
)


def identity_pose(x: float = 0.0) -> Pose:
    return Pose(Point(x, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0))


def metadata():
    measurement = TactileMeasurement(
        TactileQuantity.FORCE,
        TactileUnit.NEWTON,
        3,
        TactileValueFrame.TAXEL,
    )
    nodes = [TactileNode("index_distal", identity_pose())]
    taxels = [
        TactileTaxel(
            f"taxel-{index}",
            identity_pose(index * 0.01),
            [TactileSkinWeight(0, 1.0)],
        )
        for index in range(2)
    ]
    openxr = TactileBindingProfile(
        openxr_hand=OpenXRHandTactileBinding(TactileHandedness.LEFT, [9])
    )
    pose_stream = TactileBindingProfile(
        isaaccapture_pose=IsaacCapturePoseTactileBinding(
            "core.HandPose",
            tracker_name="hand",
            channel="left",
            subject_id="operator",
            device_instance_id="glove-left",
            skeleton_instance_id="operator-left-hand",
            node_pose_indices=[9],
        )
    )
    profiles = [
        make_tactile_binding_profile(openxr, ProfileHandling.REQUIRED),
        make_tactile_binding_profile(pose_stream, ProfileHandling.OPTIONAL),
    ]
    return TactileMetadata("glove-left", measurement, taxels, nodes, profiles)


def test_tactile_metadata_and_profiles_round_trip() -> None:
    value = TactileMetadata.from_bytes(metadata().to_bytes())
    assert value.schema_version == 1
    assert value.shape_frame_id == "glove-left"
    assert [taxel.taxel_id for taxel in value.taxels] == ["taxel-0", "taxel-1"]
    assert [profile.profile_id for profile in value.profiles] == [
        TACTILE_OPENXR_HAND_PROFILE_ID,
        TACTILE_ISAACCAPTURE_POSE_PROFILE_ID,
    ]

    binding = TactileBindingProfile.from_bytes(value.profiles[0].payload)
    assert binding.kind == TactileBindingKind.OPENXR_HAND
    assert binding.openxr_hand.node_joint_indices == [9]
    assert binding.urdf is None


def test_tactile_frame_validation() -> None:
    shape = metadata()
    data = TactileData([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    validate_tactile_data(shape, data)
    with pytest.raises(ValueError, match="value count"):
        validate_tactile_data(shape, TactileData([1.0]))


def test_urdf_profile_is_typed_and_content_addressed() -> None:
    resource = ContentDescriptor(
        "hand-model",
        media_type="application/xml",
        version="1",
        uris=["package://hand/model.urdf"],
        sha256=b"d" * 32,
        byte_size=7,
        inline_data=b"<robot>",
    )
    binding = TactileBindingProfile(
        urdf=UrdfTactileBinding(
            "hand-model", [UrdfNodeBinding(0, UrdfEntityKind.LINK, "finger")]
        )
    )
    profile = make_tactile_binding_profile(binding, resources=[resource])
    measurement = TactileMeasurement(
        TactileQuantity.PRESSURE,
        TactileUnit.PASCAL,
        1,
        TactileValueFrame.NONE,
    )

    value = TactileMetadata(
        "hand/base",
        measurement,
        [TactileTaxel("pad", identity_pose(), [TactileSkinWeight(0, 1.0)])],
        [TactileNode("finger", identity_pose())],
        [profile],
    )
    assert value.profiles[0].profile_id == TACTILE_URDF_PROFILE_ID


def test_tactile_metadata_rejects_invalid_skinning() -> None:
    measurement = TactileMeasurement(
        TactileQuantity.RAW,
        TactileUnit.COUNT,
        1,
        TactileValueFrame.NONE,
    )
    with pytest.raises(ValueError, match="sum to 1"):
        TactileMetadata(
            "array",
            measurement,
            [TactileTaxel("taxel", identity_pose(), [TactileSkinWeight(0, 0.5)])],
            [TactileNode("root", identity_pose())],
        )

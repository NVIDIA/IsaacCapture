# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SkeletonBinding, the skeleton profile registry and the asset checks."""

from isaaccapture.schema import (
    SKELETON_PROFILE_OPENXR_HAND,
    SKELETON_PROFILE_URDF_LINK,
    AssetRef,
    BodySide,
    HandJoint,
    SkeletonBinding,
    SkeletonProfileRef,
    asset_digest,
    check_skeleton_asset,
    find_skeleton_profile,
    skeleton_profiles,
    validate_skeleton_binding,
)

URDF = b"""<robot name="glove">
  <link name="base_link"/>
  <link name="index_tip"/>
  <joint name="index" type="fixed"><parent link="base_link"/><child link="index_tip"/></joint>
</robot>"""


def urdf_binding(names, data=URDF):
    return SkeletonBinding(
        names,
        profile=SkeletonProfileRef(SKELETON_PROFILE_URDF_LINK, 1),
        asset=AssetRef(
            asset_digest(data), "package://glove/glove.urdf", "application/xml"
        ),
    )


def test_binding_round_trips():
    binding = SkeletonBinding(
        ["WRIST", "INDEX_TIP"], profile=SkeletonProfileRef("openxr.hand", 1)
    )

    assert binding.segment_names == ["WRIST", "INDEX_TIP"]
    assert binding.profile.id == "openxr.hand"
    assert binding.profile.version == 1
    assert binding.asset is None
    assert SkeletonBinding(["a"]).profile is None
    assert binding.profile.to_bytes() == SkeletonProfileRef("openxr.hand", 1).to_bytes()


def test_registry_lists_openxr_hand_joints():
    hand = find_skeleton_profile(SKELETON_PROFILE_OPENXR_HAND, 1)

    assert hand.requires_side
    assert hand.segment_names == [
        name for name in HandJoint.__members__ if name != "NUM_JOINTS"
    ]
    assert find_skeleton_profile("openxr.hand", 2) is None
    assert {p.id for p in skeleton_profiles()} == {
        "openxr.hand",
        "openxr.body.bd",
        "urdf.link",
    }


def test_validate_reports_the_first_problem():
    binding = SkeletonBinding(["WRIST"], profile=SkeletonProfileRef("openxr.hand", 1))

    assert validate_skeleton_binding(binding, 1, BodySide.RIGHT) == ""
    assert "side" in validate_skeleton_binding(binding, 1)
    assert "2 segments" in validate_skeleton_binding(binding, 2, BodySide.RIGHT)


def test_asset_digest_matches_rfc6920_example():
    assert (
        asset_digest(b"Hello World!")
        == "ni:///sha-256;f4OxZX_x_FO5LcGBSKHWXfwtSx-j1ncoSt3SABJtkGk"
    )


def test_check_skeleton_asset():
    binding = urdf_binding(["base_link", "index_tip"])

    assert validate_skeleton_binding(binding, 2) == ""
    assert check_skeleton_asset(binding, URDF) == ""
    assert "digest" in check_skeleton_asset(binding, URDF + b" ")
    assert "not a link" in check_skeleton_asset(urdf_binding(["index"]), URDF)
    assert "no asset" in check_skeleton_asset(SkeletonBinding(["a"]), URDF)

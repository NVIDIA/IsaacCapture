// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the skeleton profile registry and validate_skeleton_binding().

#include <catch2/catch_test_macros.hpp>
#include <schema/serialized.hpp>
#include <schema/skeleton_binding.hpp>

#include <memory>
#include <string>
#include <vector>

namespace
{

const std::string kDigest = "ni:///sha-256;f4OxZX_x_FO5LcGBSKHWXfwtSx-j1ncoSt3SABJtkGk";

core::SkeletonBindingT binding(std::vector<std::string> names, const std::string& profile_id = "", uint32_t version = 1)
{
    core::SkeletonBindingT native;
    native.segment_names = std::move(names);
    if (!profile_id.empty())
    {
        native.profile = std::make_shared<core::SkeletonProfileRefT>();
        native.profile->id = profile_id;
        native.profile->version = version;
    }
    return native;
}

core::SkeletonBindingT with_asset(core::SkeletonBindingT native, const std::string& digest = kDigest)
{
    native.asset = std::make_shared<core::AssetRefT>();
    native.asset->digest = digest;
    native.asset->media_type = "application/xml";
    return native;
}

std::string validate(const core::SkeletonBindingT& native, size_t segment_count, core::BodySide side)
{
    return core::validate_skeleton_binding(*core::pack<core::SkeletonBinding>(native), segment_count, side);
}

} // namespace

TEST_CASE("Registered profiles take their names from the joint enums", "[unit][skeleton_binding]")
{
    const auto* hand = core::find_skeleton_profile(core::SKELETON_PROFILE_OPENXR_HAND, 1);
    REQUIRE(hand != nullptr);
    REQUIRE(hand->segment_names.size() == core::HandJoint_NUM_JOINTS);
    CHECK(hand->segment_names.front() == "PALM");
    CHECK(hand->segment_names.back() == "LITTLE_TIP");
    CHECK(hand->requires_side);
    CHECK_FALSE(hand->requires_asset);

    const auto* body = core::find_skeleton_profile(core::SKELETON_PROFILE_OPENXR_BODY_BD, 1);
    REQUIRE(body != nullptr);
    CHECK(body->segment_names.size() == core::BodyJoint_NUM_JOINTS);

    const auto* urdf = core::find_skeleton_profile(core::SKELETON_PROFILE_URDF_LINK, 1);
    REQUIRE(urdf != nullptr);
    CHECK(urdf->segment_names.empty());
    CHECK(urdf->requires_asset);

    CHECK(core::find_skeleton_profile(core::SKELETON_PROFILE_OPENXR_HAND, 2) == nullptr);
    CHECK(core::skeleton_profiles().size() == 3);
}

TEST_CASE("Registry ids are dotted lowercase tokens", "[unit][skeleton_binding]")
{
    CHECK(core::is_valid_registry_id("openxr.hand"));
    CHECK(core::is_valid_registry_id("com.example.glove_v2"));
    CHECK_FALSE(core::is_valid_registry_id("openxr"));
    CHECK_FALSE(core::is_valid_registry_id("OpenXR.hand"));
    CHECK_FALSE(core::is_valid_registry_id("openxr..hand"));
    CHECK_FALSE(core::is_valid_registry_id("openxr.hand."));
    CHECK_FALSE(core::is_valid_registry_id("1x.hand"));
    CHECK_FALSE(core::is_valid_registry_id("openxr.hand-left"));
}

TEST_CASE("Asset digests are RFC 6920 SHA-256 names", "[unit][skeleton_binding]")
{
    CHECK(core::is_valid_asset_digest(kDigest));
    CHECK_FALSE(core::is_valid_asset_digest("sha-256;f4OxZX_x_FO5LcGBSKHWXfwtSx-j1ncoSt3SABJtkGk"));
    CHECK_FALSE(core::is_valid_asset_digest(kDigest + "="));
    CHECK_FALSE(core::is_valid_asset_digest("ni:///sha-256;f4OxZX/x+FO5LcGBSKHWXfwtSx-j1ncoSt3SABJtkGk"));
}

TEST_CASE("validate_skeleton_binding accepts conforming bindings", "[unit][skeleton_binding]")
{
    CHECK(validate(binding({ "WRIST", "INDEX_TIP" }, "openxr.hand"), 2, core::BodySide_LEFT).empty());
    CHECK(validate(binding({ "LEFT_HAND", "RIGHT_HAND" }, "openxr.body.bd"), 2, core::BodySide_UNSPECIFIED).empty());
    CHECK(validate(with_asset(binding({ "base_link" }, "urdf.link")), 1, core::BodySide_UNSPECIFIED).empty());
    CHECK(validate(binding({ "palm", "pad" }), 2, core::BodySide_UNSPECIFIED).empty());
    // A profile this build does not know reads as a generic binding.
    CHECK(validate(binding({ "anything" }, "com.example.hand", 3), 1, core::BodySide_UNSPECIFIED).empty());
}

TEST_CASE("validate_skeleton_binding rejects nonconforming bindings", "[unit][skeleton_binding]")
{
    CHECK_FALSE(validate(binding({ "WRIST" }, "openxr.hand"), 2, core::BodySide_LEFT).empty());
    CHECK_FALSE(validate(binding({ "WRIST", "" }, "openxr.hand"), 2, core::BodySide_LEFT).empty());
    CHECK_FALSE(validate(binding({ "WRIST", "WRIST" }, "openxr.hand"), 2, core::BodySide_LEFT).empty());
    CHECK_FALSE(validate(binding({ "WRIST" }, "openxr.hand"), 1, core::BodySide_UNSPECIFIED).empty());
    CHECK_FALSE(validate(binding({ "LEFT_HAND" }, "openxr.hand"), 1, core::BodySide_LEFT).empty());
    CHECK_FALSE(validate(binding({ "WRIST" }, "openxr.hand", 0), 1, core::BodySide_LEFT).empty());
    CHECK_FALSE(validate(binding({ "WRIST" }, "OpenXR.Hand"), 1, core::BodySide_LEFT).empty());
    CHECK_FALSE(validate(with_asset(binding({ "WRIST" }, "openxr.hand")), 1, core::BodySide_LEFT).empty());
    CHECK_FALSE(validate(binding({ "base_link" }, "urdf.link"), 1, core::BodySide_UNSPECIFIED).empty());
    CHECK_FALSE(
        validate(with_asset(binding({ "base_link" }, "urdf.link"), "md5:abc"), 1, core::BodySide_UNSPECIFIED).empty());
    CHECK_FALSE(validate(with_asset(binding({ "base_link" })), 1, core::BodySide_UNSPECIFIED).empty());
}

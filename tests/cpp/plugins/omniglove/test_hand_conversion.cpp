// SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <omniglove/omniglove.h>
#include <omniglove_plugin/omniglove_hand_conversion.hpp>
#include <openxr/openxr.h>

#include <array>
#include <cmath>
#include <limits>

namespace
{

using plugins::omniglove::ConversionStatus;

void fill_pose(omniglove_pose6_t& pose, float x, float y, float z, const XrQuaternionf& orientation)
{
    pose.position_m[0] = x;
    pose.position_m[1] = y;
    pose.position_m[2] = z;
    pose.quaternion_xyzw[0] = orientation.x;
    pose.quaternion_xyzw[1] = orientation.y;
    pose.quaternion_xyzw[2] = orientation.z;
    pose.quaternion_xyzw[3] = orientation.w;
}

void fill_hand(omniglove_post_data_t& post, const XrQuaternionf& orientation = { 0.0F, 0.0F, 0.0F, 2.0F })
{
    post.mapped = 1;
    post.flags = plugins::omniglove::kPostIkConverged | plugins::omniglove::kPostBoneModelValid;
    float x = 0.01F;
    auto fill = [&x, &orientation](omniglove_pose6_t& pose)
    {
        fill_pose(pose, x, x + 0.01F, x + 0.02F, orientation);
        x += 0.01F;
    };
    fill(post.wrist.landmark);
    fill(post.thumb.landmarks.cmc);
    fill(post.thumb.landmarks.mcp);
    fill(post.thumb.landmarks.ip);
    fill(post.thumb.landmarks.tip);
    const std::array<omniglove_post_finger_t*, 4> fingers = { &post.index, &post.middle, &post.ring, &post.pinky };
    for (auto* finger : fingers)
    {
        fill(finger->landmarks.mcp);
        fill(finger->landmarks.pip);
        fill(finger->landmarks.dip);
        fill(finger->landmarks.tip);
    }
}

bool near(float left, float right)
{
    return std::abs(left - right) < 1.0e-6F;
}

std::array<float, 3> rotate(const XrQuaternionf& q, const std::array<float, 3>& v)
{
    const std::array<float, 3> t = { 2.0F * (q.y * v[2] - q.z * v[1]), 2.0F * (q.z * v[0] - q.x * v[2]),
                                     2.0F * (q.x * v[1] - q.y * v[0]) };
    return { v[0] + q.w * t[0] + q.y * t[2] - q.z * t[1], v[1] + q.w * t[1] + q.z * t[0] - q.x * t[2],
             v[2] + q.w * t[2] + q.x * t[1] - q.y * t[0] };
}

bool openxr_axes(bool is_left, const XrQuaternionf& orientation)
{
    omniglove_post_data_t post{};
    fill_hand(post, orientation);
    std::array<XrHandJointLocationEXT, XR_HAND_JOINT_COUNT_EXT> joints{};
    if (plugins::omniglove::convert_post_data(post, is_left, joints) != ConversionStatus::Success)
    {
        return false;
    }
    const float norm = std::sqrt(orientation.x * orientation.x + orientation.y * orientation.y +
                                 orientation.z * orientation.z + orientation.w * orientation.w);
    const XrQuaternionf unit = { orientation.x / norm, orientation.y / norm, orientation.z / norm, orientation.w / norm };
    for (std::size_t index = 0; index < joints.size(); ++index)
    {
        const auto& joint = joints[index];
        if (joint.locationFlags == 0)
        {
            continue;
        }
        for (const std::array<float, 3>& axis :
             { std::array<float, 3>{ 0.0F, 1.0F, 0.0F }, std::array<float, 3>{ 0.0F, 0.0F, 1.0F } })
        {
            auto sdk_axis = axis;
            if (!is_left)
            {
                sdk_axis[0] = -sdk_axis[0];
                sdk_axis[1] = -sdk_axis[1];
            }
            auto expected = rotate(unit, sdk_axis);
            if (!is_left)
            {
                expected[0] = -expected[0];
                expected[1] = -expected[1];
            }
            auto xr_axis = axis;
            if (index != XR_HAND_JOINT_WRIST_EXT && axis[2] != 0.0F)
            {
                xr_axis[2] = -xr_axis[2];
            }
            const auto actual = rotate(joint.pose.orientation, xr_axis);
            for (std::size_t component = 0; component < actual.size(); ++component)
            {
                if (!near(actual[component], expected[component]))
                {
                    return false;
                }
            }
        }
    }
    return true;
}

} // namespace

TEST_CASE("OmniGlove conversion preserves its OpenXR hand contract", "[omniglove][conversion]")
{
    omniglove_post_data_t post{};
    fill_hand(post);
    std::array<XrHandJointLocationEXT, XR_HAND_JOINT_COUNT_EXT> joints{};
    REQUIRE(plugins::omniglove::convert_post_data(post, true, joints) == ConversionStatus::Success);
    constexpr XrSpaceLocationFlags valid = XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
    int valid_count = 0;
    for (const auto& joint : joints)
    {
        valid_count += (joint.locationFlags & valid) == valid;
    }
    CHECK(valid_count == 21);
    CHECK(joints[XR_HAND_JOINT_PALM_EXT].locationFlags == 0);
    CHECK(joints[XR_HAND_JOINT_INDEX_METACARPAL_EXT].locationFlags == 0);
    CHECK(near(joints[XR_HAND_JOINT_WRIST_EXT].pose.orientation.w, 1.0F));

    for (bool is_left : { true, false })
    {
        CHECK(openxr_axes(is_left, { 0.0F, 0.0F, 0.0F, 2.0F }));
        CHECK(openxr_axes(is_left, { 1.0F, 2.0F, 3.0F, 4.0F }));
    }
    const XrPosef left_wrist = joints[XR_HAND_JOINT_WRIST_EXT].pose;
    REQUIRE(plugins::omniglove::convert_post_data(post, false, joints) == ConversionStatus::Success);
    CHECK(near(joints[XR_HAND_JOINT_WRIST_EXT].pose.position.x, -left_wrist.position.x));
    CHECK(near(joints[XR_HAND_JOINT_WRIST_EXT].pose.position.y, -left_wrist.position.y));
    CHECK(near(joints[XR_HAND_JOINT_WRIST_EXT].pose.position.z, left_wrist.position.z));

    post.flags = plugins::omniglove::kPostBoneModelValid;
    CHECK(plugins::omniglove::convert_post_data(post, true, joints) == ConversionStatus::IkNotConverged);
    post.flags = plugins::omniglove::kPostIkConverged;
    CHECK(plugins::omniglove::convert_post_data(post, true, joints) == ConversionStatus::BoneModelInvalid);
    fill_hand(post);
    post.mapped = 0;
    CHECK(plugins::omniglove::convert_post_data(post, true, joints) == ConversionStatus::NotMapped);
    fill_hand(post);
    post.index.landmarks.mcp.position_m[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(plugins::omniglove::convert_post_data(post, true, joints) == ConversionStatus::InvalidPose);
}

// SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <omniglove/omniglove.h>
#include <omniglove_plugin/omniglove_hand_conversion.hpp>
#include <openxr/openxr.h>
#include <oxr_utils/math.hpp>

#include <array>
#include <cmath>
#include <limits>

namespace
{

using plugins::omniglove::ConversionStatus;
using plugins::omniglove::HandJoints;

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

std::array<float, 3> subtract(const XrVector3f& a, const XrVector3f& b)
{
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

float dot(const std::array<float, 3>& a, const std::array<float, 3>& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

std::array<float, 3> normalized(const std::array<float, 3>& v)
{
    const float length = std::sqrt(dot(v, v));
    return { v[0] / length, v[1] / length, v[2] / length };
}

std::array<float, 3> cross(const std::array<float, 3>& a, const std::array<float, 3>& b)
{
    return { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
}

bool openxr_axes(bool is_left, const XrQuaternionf& orientation)
{
    omniglove_post_data_t post{};
    fill_hand(post, orientation);
    const auto result = plugins::omniglove::convert_post_data(post, is_left);
    const auto* converted = std::get_if<HandJoints>(&result);
    if (converted == nullptr)
    {
        return false;
    }
    const auto& joints = *converted;
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
            if (index >= XR_HAND_JOINT_THUMB_METACARPAL_EXT && index <= XR_HAND_JOINT_THUMB_TIP_EXT && axis[1] != 0.0F)
            {
                xr_axis = { is_left ? -1.0F : 1.0F, 0.0F, 0.0F };
            }
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
    auto result = plugins::omniglove::convert_post_data(post, true);
    REQUIRE(std::holds_alternative<HandJoints>(result));
    auto joints = std::get<HandJoints>(result);
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
    result = plugins::omniglove::convert_post_data(post, false);
    REQUIRE(std::holds_alternative<HandJoints>(result));
    joints = std::get<HandJoints>(result);
    CHECK(near(joints[XR_HAND_JOINT_WRIST_EXT].pose.position.x, -left_wrist.position.x));
    CHECK(near(joints[XR_HAND_JOINT_WRIST_EXT].pose.position.y, -left_wrist.position.y));
    CHECK(near(joints[XR_HAND_JOINT_WRIST_EXT].pose.position.z, left_wrist.position.z));

    post.flags = plugins::omniglove::kPostBoneModelValid;
    CHECK(std::get<ConversionStatus>(plugins::omniglove::convert_post_data(post, true)) ==
          ConversionStatus::IkNotConverged);
    post.flags = plugins::omniglove::kPostIkConverged;
    CHECK(std::get<ConversionStatus>(plugins::omniglove::convert_post_data(post, true)) ==
          ConversionStatus::BoneModelInvalid);
    fill_hand(post);
    post.mapped = 0;
    CHECK(std::get<ConversionStatus>(plugins::omniglove::convert_post_data(post, true)) == ConversionStatus::NotMapped);
    fill_hand(post);
    post.index.landmarks.mcp.position_m[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(std::get<ConversionStatus>(plugins::omniglove::convert_post_data(post, true)) == ConversionStatus::InvalidPose);
}

TEST_CASE("OmniGlove bent thumb hinges align with local X", "[omniglove][conversion]")
{
    const float inverse_norm = 1.0F / std::sqrt(30.0F);
    const std::array<XrPosef, 2> source_transforms = { XrPosef{ { 0.0F, 0.0F, 0.0F, 1.0F }, {} },
                                                       XrPosef{ { inverse_norm, 2.0F * inverse_norm,
                                                                  3.0F * inverse_norm, 4.0F * inverse_norm },
                                                                { 0.11F, -0.07F, 0.09F } } };
    constexpr std::array thumb_joints = { XR_HAND_JOINT_THUMB_METACARPAL_EXT, XR_HAND_JOINT_THUMB_PROXIMAL_EXT,
                                          XR_HAND_JOINT_THUMB_DISTAL_EXT, XR_HAND_JOINT_THUMB_TIP_EXT };
    for (bool is_left : { true, false })
    {
        for (const XrPosef& transform : source_transforms)
        {
            for (float ip_bend : { 0.4F, 1.1F })
            {
                CAPTURE(is_left, ip_bend, transform.orientation.x);
                constexpr float mcp_bend = 0.45F;
                const float distal_angle = mcp_bend + ip_bend;
                const XrQuaternionf mcp_orientation{ 0.0F, std::sin(mcp_bend / 2.0F), 0.0F, std::cos(mcp_bend / 2.0F) };
                const XrQuaternionf ip_orientation{ 0.0F, std::sin(distal_angle / 2.0F), 0.0F,
                                                    std::cos(distal_angle / 2.0F) };
                const XrVector3f mcp{ 0.02F, 0.01F, 0.025F };
                const XrVector3f ip{ mcp.x + 0.03F * std::sin(mcp_bend), mcp.y, mcp.z + 0.03F * std::cos(mcp_bend) };
                const XrVector3f tip{ ip.x + 0.02F * std::sin(distal_angle), ip.y, ip.z + 0.02F * std::cos(distal_angle) };
                // Thumb flexion is in the SDK XZ plane, with its hinge along Y.
                const std::array<XrPosef, 4> local_poses = {
                    XrPosef{ { 0.0F, 0.0F, 0.0F, 1.0F }, { 0.02F, 0.01F, 0.0F } }, XrPosef{ mcp_orientation, mcp },
                    XrPosef{ ip_orientation, ip }, XrPosef{ ip_orientation, tip }
                };
                omniglove_post_data_t post{};
                fill_hand(post);
                const std::array landmarks = { &post.thumb.landmarks.cmc, &post.thumb.landmarks.mcp,
                                               &post.thumb.landmarks.ip, &post.thumb.landmarks.tip };
                std::array<XrVector3f, 4> expected_positions;
                for (std::size_t i = 0; i < local_poses.size(); ++i)
                {
                    const XrPosef pose = oxr_utils::multiply_poses(transform, local_poses[i]);
                    fill_pose(*landmarks[i], pose.position.x, pose.position.y, pose.position.z, pose.orientation);
                    expected_positions[i] = pose.position;
                    if (!is_left)
                    {
                        expected_positions[i].x = -expected_positions[i].x;
                        expected_positions[i].y = -expected_positions[i].y;
                    }
                }
                std::array<std::array<float, 3>, 4> bone_directions;
                for (std::size_t i = 0; i < bone_directions.size() - 1; ++i)
                {
                    bone_directions[i] = normalized(subtract(expected_positions[i + 1], expected_positions[i]));
                }
                bone_directions.back() = bone_directions[2];
                const auto ip_hinge = normalized(cross(bone_directions[2], bone_directions[1]));
                const auto mcp_hinge = normalized(cross(bone_directions[1], bone_directions[0]));
                const auto result = plugins::omniglove::convert_post_data(post, is_left);
                REQUIRE(std::holds_alternative<HandJoints>(result));
                const auto& joints = std::get<HandJoints>(result);
                for (std::size_t i = 0; i < thumb_joints.size(); ++i)
                {
                    CAPTURE(i);
                    const auto& pose = joints[thumb_joints[i]].pose;
                    CHECK(near(pose.position.x, expected_positions[i].x));
                    CHECK(near(pose.position.y, expected_positions[i].y));
                    CHECK(near(pose.position.z, expected_positions[i].z));
                    const auto x_axis = rotate(pose.orientation, { 1.0F, 0.0F, 0.0F });
                    CHECK(dot(x_axis, ip_hinge) > 0.99999F);
                    CHECK(dot(x_axis, mcp_hinge) > 0.99999F);
                    CHECK(dot(rotate(pose.orientation, { 0.0F, 0.0F, -1.0F }), bone_directions[i]) > 0.99999F);
                }
            }
        }
    }
}

TEST_CASE("OmniGlove returns no joints when the last landmark is invalid", "[omniglove][conversion]")
{
    omniglove_post_data_t post{};
    fill_hand(post);
    post.pinky.landmarks.tip.quaternion_xyzw[3] = std::numeric_limits<float>::infinity();
    auto result = plugins::omniglove::convert_post_data(post, true);
    REQUIRE(std::holds_alternative<ConversionStatus>(result));
    CHECK(std::get<ConversionStatus>(result) == ConversionStatus::InvalidPose);

    fill_hand(post);
    for (auto& component : post.pinky.landmarks.tip.quaternion_xyzw)
    {
        component = 0.0F;
    }
    result = plugins::omniglove::convert_post_data(post, false);
    REQUIRE(std::holds_alternative<ConversionStatus>(result));
    CHECK(std::get<ConversionStatus>(result) == ConversionStatus::InvalidPose);
}

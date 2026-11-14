// SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "inc/omniglove_plugin/omniglove_hand_conversion.hpp"

#include <omniglove/omniglove.h>
#include <oxr_utils/math.hpp>
#include <plugin_utils/hand_pose_basis.hpp>

#include <array>
#include <cmath>
#include <cstddef>

namespace plugins::omniglove
{

namespace
{

constexpr float kJointRadiusM = 0.01F;
constexpr XrSpaceLocationFlags kValidFlags =
    XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;

struct JointSource
{
    XrHandJointEXT joint;
    const omniglove_pose6_t* pose;
};

auto joint_sources(const omniglove_post_data_t& post)
{
    return std::array{ JointSource{ XR_HAND_JOINT_WRIST_EXT, &post.wrist.landmark },
                       JointSource{ XR_HAND_JOINT_THUMB_METACARPAL_EXT, &post.thumb.landmarks.cmc },
                       JointSource{ XR_HAND_JOINT_THUMB_PROXIMAL_EXT, &post.thumb.landmarks.mcp },
                       JointSource{ XR_HAND_JOINT_THUMB_DISTAL_EXT, &post.thumb.landmarks.ip },
                       JointSource{ XR_HAND_JOINT_THUMB_TIP_EXT, &post.thumb.landmarks.tip },
                       JointSource{ XR_HAND_JOINT_INDEX_PROXIMAL_EXT, &post.index.landmarks.mcp },
                       JointSource{ XR_HAND_JOINT_INDEX_INTERMEDIATE_EXT, &post.index.landmarks.pip },
                       JointSource{ XR_HAND_JOINT_INDEX_DISTAL_EXT, &post.index.landmarks.dip },
                       JointSource{ XR_HAND_JOINT_INDEX_TIP_EXT, &post.index.landmarks.tip },
                       JointSource{ XR_HAND_JOINT_MIDDLE_PROXIMAL_EXT, &post.middle.landmarks.mcp },
                       JointSource{ XR_HAND_JOINT_MIDDLE_INTERMEDIATE_EXT, &post.middle.landmarks.pip },
                       JointSource{ XR_HAND_JOINT_MIDDLE_DISTAL_EXT, &post.middle.landmarks.dip },
                       JointSource{ XR_HAND_JOINT_MIDDLE_TIP_EXT, &post.middle.landmarks.tip },
                       JointSource{ XR_HAND_JOINT_RING_PROXIMAL_EXT, &post.ring.landmarks.mcp },
                       JointSource{ XR_HAND_JOINT_RING_INTERMEDIATE_EXT, &post.ring.landmarks.pip },
                       JointSource{ XR_HAND_JOINT_RING_DISTAL_EXT, &post.ring.landmarks.dip },
                       JointSource{ XR_HAND_JOINT_RING_TIP_EXT, &post.ring.landmarks.tip },
                       JointSource{ XR_HAND_JOINT_LITTLE_PROXIMAL_EXT, &post.pinky.landmarks.mcp },
                       JointSource{ XR_HAND_JOINT_LITTLE_INTERMEDIATE_EXT, &post.pinky.landmarks.pip },
                       JointSource{ XR_HAND_JOINT_LITTLE_DISTAL_EXT, &post.pinky.landmarks.dip },
                       JointSource{ XR_HAND_JOINT_LITTLE_TIP_EXT, &post.pinky.landmarks.tip } };
}

bool convert_pose(const omniglove_pose6_t& source, bool is_left, XrPosef& destination)
{
    for (float value : source.position_m)
    {
        if (!std::isfinite(value))
        {
            return false;
        }
    }
    for (float value : source.quaternion_xyzw)
    {
        if (!std::isfinite(value))
        {
            return false;
        }
    }

    const float norm = std::sqrt(
        source.quaternion_xyzw[0] * source.quaternion_xyzw[0] + source.quaternion_xyzw[1] * source.quaternion_xyzw[1] +
        source.quaternion_xyzw[2] * source.quaternion_xyzw[2] + source.quaternion_xyzw[3] * source.quaternion_xyzw[3]);
    if (norm < 1.0e-6F)
    {
        return false;
    }

    destination.position = { source.position_m[0], source.position_m[1], source.position_m[2] };
    destination.orientation = { source.quaternion_xyzw[0] / norm, source.quaternion_xyzw[1] / norm,
                                source.quaternion_xyzw[2] / norm, source.quaternion_xyzw[3] / norm };

    destination = plugin_utils::remap_to_openxr_basis(destination, is_left);
    return true;
}

} // namespace

ConversionResult convert_post_data(const omniglove_post_data_t& post, bool is_left)
{
    HandJoints joints{};
    for (auto& joint : joints)
    {
        joint = {};
        joint.pose.orientation.w = 1.0F;
        joint.radius = kJointRadiusM;
    }

    if (post.mapped == 0U)
    {
        return ConversionStatus::NotMapped;
    }
    if ((post.flags & kPostIkConverged) == 0U)
    {
        return ConversionStatus::IkNotConverged;
    }
    if ((post.flags & kPostBoneModelValid) == 0U)
    {
        return ConversionStatus::BoneModelInvalid;
    }

    for (const JointSource& source : joint_sources(post))
    {
        XrHandJointLocationEXT& destination = joints[static_cast<std::size_t>(source.joint)];
        if (!convert_pose(*source.pose, is_left, destination.pose))
        {
            return ConversionStatus::InvalidPose;
        }
        if (source.joint != XR_HAND_JOINT_WRIST_EXT)
        {
            // SDK finger frames point +Z toward the tip; OpenXR points -Z.
            // Post-multiply by a local Y half-turn to preserve dorsal +Y.
            // The SDK wrist already has proximal +Z and must not be flipped.
            constexpr XrPosef kYHalfTurn{ { 0.0F, 1.0F, 0.0F, 0.0F }, { 0.0F, 0.0F, 0.0F } };
            destination.pose = oxr_utils::multiply_poses(destination.pose, kYHalfTurn);
        }
        destination.locationFlags = kValidFlags;
    }
    return joints;
}

const char* conversion_status_name(ConversionStatus status) noexcept
{
    switch (status)
    {
    case ConversionStatus::NotMapped:
        return "not mapped";
    case ConversionStatus::IkNotConverged:
        return "IK not converged";
    case ConversionStatus::BoneModelInvalid:
        return "bone model invalid";
    case ConversionStatus::InvalidPose:
        return "invalid pose";
    }
    return "unknown";
}

} // namespace plugins::omniglove

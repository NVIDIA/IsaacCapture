// SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "io_exo_glove_ros2_pack.hpp"

#include <flatbuffers/flatbuffers.h>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace plugins
{
namespace io_exo_glove_ros2
{

namespace
{

//! JointName keys parallel to kFingertipFrameSuffixes, ascending -- the order the schema requires
//! for the keyed vector it binary-searches.
constexpr core::JointName kFingertipJoints[kFingertipCount] = {
    core::JointName_HAND_RAW_THUMB_TIP, core::JointName_HAND_RAW_INDEX_TIP,  core::JointName_HAND_RAW_MIDDLE_TIP,
    core::JointName_HAND_RAW_RING_TIP,  core::JointName_HAND_RAW_LITTLE_TIP,
};

//! Matches the hand's fingertip frames, leaving nullptr for the fingers it does not publish.
std::array<const geometry_msgs::msg::TransformStamped*, kFingertipCount> find_fingertips(const tf2_msgs::msg::TFMessage& msg,
                                                                                         Hand hand)
{
    const std::string prefix = (hand == Hand::Left) ? "left" : "right";

    std::array<std::string, kFingertipCount> wanted;
    for (std::size_t i = 0; i < kFingertipCount; ++i)
    {
        wanted[i] = prefix + kFingertipFrameSuffixes[i];
    }

    std::array<const geometry_msgs::msg::TransformStamped*, kFingertipCount> tips{};
    for (const auto& transform : msg.transforms)
    {
        for (std::size_t i = 0; i < kFingertipCount; ++i)
        {
            if (transform.child_frame_id == wanted[i])
            {
                tips[i] = &transform;
            }
        }
    }
    return tips;
}

} // namespace

TipPosePackResult pack_fingertip_poses(const tf2_msgs::msg::TFMessage& msg,
                                       Hand hand,
                                       const std::string& device_id,
                                       const std::string& reference_frame)
{
    TipPosePackResult result;
    result.reference_frame = reference_frame;

    const auto tips = find_fingertips(msg, hand);

    // Sorted by JointName: the schema requires the keyed vector to be sorted (and key-unique) because
    // readers binary-search it, and kFingertipJoints is ascending while this loop keeps that order.
    std::vector<core::JointSe3Pose> poses;
    poses.reserve(kFingertipCount);
    for (std::size_t i = 0; i < kFingertipCount; ++i)
    {
        const auto* transform = tips[i];
        if (transform == nullptr)
        {
            continue; // Sparse: an absent frame is simply not tracked.
        }
        if (transform->header.frame_id != reference_frame)
        {
            ++result.skipped;
            continue;
        }

        const auto& pose = transform->transform;
        poses.emplace_back(
            kFingertipJoints[i],
            core::Pose(core::Point(static_cast<float>(pose.translation.x), static_cast<float>(pose.translation.y),
                                   static_cast<float>(pose.translation.z)),
                       core::Quaternion(static_cast<float>(pose.rotation.x), static_cast<float>(pose.rotation.y),
                                        static_cast<float>(pose.rotation.z), static_cast<float>(pose.rotation.w))));
    }

    if (poses.empty())
    {
        return result;
    }

    flatbuffers::FlatBufferBuilder builder(kTipMaxFlatbufferSize);
    const auto device = builder.CreateString(device_id);
    // Build the keyed vector through the sorted helper: JointSe3PoseOutput::Pack() emits it with
    // CreateVectorOfStructs and does not sort, which would make the reader's LookupByKey return the
    // wrong pose instead of failing (see the schema's note on the joints field).
    const auto joints = builder.CreateVectorOfSortedStructs(&poses);
    builder.Finish(core::CreateJointSe3PoseOutput(builder, core::JointType_HAND_RAW, joints, device));

    result.buffer.assign(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
    result.poses = poses.size();
    return result;
}

} // namespace io_exo_glove_ros2
} // namespace plugins

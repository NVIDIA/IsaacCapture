// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/jmex/body_joint_mapping.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <optional>

namespace plugins
{
namespace jmex
{

namespace
{

// The skeleton's root and finger joints fill no slot, so they have no row here.
constexpr SlotNameTable make_standard_human_skeleton_joint_names()
{
    SlotNameTable names{};
    names[core::BodyJoint_PELVIS] = "pelvis_joint";
    names[core::BodyJoint_LEFT_HIP] = "left_hip_joint";
    names[core::BodyJoint_RIGHT_HIP] = "right_hip_joint";
    names[core::BodyJoint_SPINE1] = "spine1_joint";
    names[core::BodyJoint_LEFT_KNEE] = "left_knee_joint";
    names[core::BodyJoint_RIGHT_KNEE] = "right_knee_joint";
    names[core::BodyJoint_SPINE2] = "spine2_joint";
    names[core::BodyJoint_LEFT_ANKLE] = "left_ankle_joint";
    names[core::BodyJoint_RIGHT_ANKLE] = "right_ankle_joint";
    names[core::BodyJoint_SPINE3] = "spine3_joint";
    names[core::BodyJoint_LEFT_FOOT] = "left_foot_joint";
    names[core::BodyJoint_RIGHT_FOOT] = "right_foot_joint";
    names[core::BodyJoint_NECK] = "neck_joint";
    names[core::BodyJoint_LEFT_COLLAR] = "left_collar_joint";
    names[core::BodyJoint_RIGHT_COLLAR] = "right_collar_joint";
    names[core::BodyJoint_HEAD] = "head_joint";
    names[core::BodyJoint_LEFT_SHOULDER] = "left_shoulder_joint";
    names[core::BodyJoint_RIGHT_SHOULDER] = "right_shoulder_joint";
    names[core::BodyJoint_LEFT_ELBOW] = "left_elbow_joint";
    names[core::BodyJoint_RIGHT_ELBOW] = "right_elbow_joint";
    names[core::BodyJoint_LEFT_WRIST] = "left_wrist_joint";
    names[core::BodyJoint_RIGHT_WRIST] = "right_wrist_joint";
    names[core::BodyJoint_LEFT_HAND] = "left_hand_joint";
    names[core::BodyJoint_RIGHT_HAND] = "right_hand_joint";
    return names;
}

// How far a quaternion may drift from unit length and still be treated as a rotation; float32
// round-off down the hierarchy stays inside it.
constexpr float kQuaternionNormTolerance = 1e-3f;

//! Whether a pose can go out valid: a finite position and a unit quaternion.
bool is_usable(const core::Pose& pose)
{
    const core::Point& p = pose.position();
    if (!std::isfinite(p.x()) || !std::isfinite(p.y()) || !std::isfinite(p.z()))
    {
        return false;
    }

    const core::Quaternion& q = pose.orientation();
    if (!std::isfinite(q.x()) || !std::isfinite(q.y()) || !std::isfinite(q.z()) || !std::isfinite(q.w()))
    {
        return false;
    }
    const float norm = std::sqrt(q.x() * q.x() + q.y() * q.y() + q.z() * q.z() + q.w() * q.w());
    return std::fabs(norm - 1.0f) <= kQuaternionNormTolerance;
}

//! Slot a joint name fills, or nullopt for a name the table does not carry.
std::optional<size_t> body_slot_for_joint_name(std::string_view joint_name)
{
    for (size_t slot = 0; slot < kStandardHumanSkeletonJointNames.size(); ++slot)
    {
        if (kStandardHumanSkeletonJointNames[slot] == joint_name)
        {
            return slot;
        }
    }
    return std::nullopt;
}

} // namespace

const SlotNameTable kStandardHumanSkeletonJointNames = make_standard_human_skeleton_joint_names();

BodyJointMapping::BodyJointMapping(const std::vector<std::string>& joint_names) : joint_count_(joint_names.size())
{
    for (size_t joint = 0; joint < joint_names.size(); ++joint)
    {
        const std::optional<size_t> slot = body_slot_for_joint_name(joint_names[joint]);
        if (!slot)
        {
            continue;
        }
        if (slot_to_joint_[*slot])
        {
            contested_joint_names_.push_back(joint_names[joint]);
            continue;
        }
        slot_to_joint_[*slot] = joint;
        ++filled_slot_count_;
    }
}

std::optional<size_t> BodyJointMapping::joint_index(core::BodyJoint slot) const
{
    assert(static_cast<size_t>(slot) < slot_to_joint_.size());
    return slot_to_joint_[static_cast<size_t>(slot)];
}

size_t BodyJointMapping::apply(const std::vector<core::Pose>& joint_poses, core::BodyJoints& out) const
{
    assert(joint_poses.size() == joint_count_);

    const core::BodyJointPose invalid(
        core::Pose(core::Point(0.0f, 0.0f, 0.0f), core::Quaternion(0.0f, 0.0f, 0.0f, 1.0f)), false);
    size_t valid_slots = 0;
    for (size_t slot = 0; slot < slot_to_joint_.size(); ++slot)
    {
        core::BodyJointPose sample = invalid;

        const std::optional<size_t> joint = slot_to_joint_[slot];
        if (joint && is_usable(joint_poses[*joint]))
        {
            sample = core::BodyJointPose(joint_poses[*joint], true);
            ++valid_slots;
        }

        out.mutable_joints()->Mutate(static_cast<flatbuffers::uoffset_t>(slot), sample);
    }
    return valid_slots;
}

} // namespace jmex
} // namespace plugins

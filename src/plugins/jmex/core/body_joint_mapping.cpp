// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/jmex/body_joint_mapping.hpp"

#include <cmath>
#include <cstddef>

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

constexpr JointPose k_identity_pose{ { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 1.0f } };

// How far a quaternion may drift from unit length and still be treated as a rotation. Wide enough
// to absorb float32 round-off down the hierarchy, narrow enough that an uninitialized or partly
// written frame does not pass.
constexpr float k_quaternion_norm_tolerance = 1e-3f;

//! Whether a pose can go out valid: a finite position and a unit quaternion.
bool is_usable(const JointPose& pose)
{
    for (float component : pose.position)
    {
        if (!std::isfinite(component))
        {
            return false;
        }
    }

    float norm_squared = 0.0f;
    for (float component : pose.orientation_xyzw)
    {
        if (!std::isfinite(component))
        {
            return false;
        }
        norm_squared += component * component;
    }

    return std::fabs(std::sqrt(norm_squared) - 1.0f) <= k_quaternion_norm_tolerance;
}

core::BodyJointPose to_schema(const JointPose& pose, bool is_valid)
{
    return core::BodyJointPose(core::Pose(core::Point(pose.position[0], pose.position[1], pose.position[2]),
                                          core::Quaternion(pose.orientation_xyzw[0], pose.orientation_xyzw[1],
                                                           pose.orientation_xyzw[2], pose.orientation_xyzw[3])),
                               is_valid);
}

//! Slot a joint name fills, or -1 for a name the table does not carry.
int body_slot_for_joint_name(std::string_view joint_name)
{
    if (joint_name.empty())
    {
        return -1;
    }
    for (size_t slot = 0; slot < k_standard_human_skeleton_joint_names.size(); ++slot)
    {
        if (k_standard_human_skeleton_joint_names[slot] == joint_name)
        {
            return static_cast<int>(slot);
        }
    }
    return -1;
}

} // namespace

const SlotNameTable k_standard_human_skeleton_joint_names = make_standard_human_skeleton_joint_names();

BodyJointMapping BodyJointMapping::resolve(const std::vector<std::string>& joint_names)
{
    BodyJointMapping mapping;
    mapping.slot_to_joint_.fill(-1);

    // Walked joint-first rather than slot-first so that a second joint claiming a taken slot is
    // seen at all: a slot-first lookup would find the first match and never know about the rest.
    for (size_t joint = 0; joint < joint_names.size(); ++joint)
    {
        const int slot = body_slot_for_joint_name(joint_names[joint]);
        if (slot < 0)
        {
            continue;
        }
        if (mapping.slot_to_joint_[static_cast<size_t>(slot)] >= 0)
        {
            mapping.contested_joint_names_.push_back(joint_names[joint]);
            continue;
        }
        mapping.slot_to_joint_[static_cast<size_t>(slot)] = static_cast<int>(joint);
        ++mapping.filled_slot_count_;
    }

    return mapping;
}

int BodyJointMapping::joint_index(core::BodyJoint slot) const
{
    if (static_cast<size_t>(slot) >= slot_to_joint_.size())
    {
        return -1;
    }
    return slot_to_joint_[static_cast<size_t>(slot)];
}

size_t BodyJointMapping::apply(const std::vector<JointPose>& joint_poses, core::BodyJoints& out) const
{
    size_t valid_slots = 0;
    for (size_t slot = 0; slot < slot_to_joint_.size(); ++slot)
    {
        core::BodyJointPose sample = to_schema(k_identity_pose, false);

        const int joint = slot_to_joint_[slot];
        if (joint >= 0 && static_cast<size_t>(joint) < joint_poses.size())
        {
            const JointPose& pose = joint_poses[static_cast<size_t>(joint)];
            if (is_usable(pose))
            {
                sample = to_schema(pose, true);
                ++valid_slots;
            }
        }

        out.mutable_joints()->Mutate(static_cast<flatbuffers::uoffset_t>(slot), sample);
    }
    return valid_slots;
}

} // namespace jmex
} // namespace plugins

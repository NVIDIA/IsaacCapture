// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <schema/full_body_generated.h>

#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace plugins
{
namespace jmex
{

//! A joint's unconstrained pose: the quaternion the motion frame carried, composed down the
//! hierarchy from the base. Meters, in OpenXR axes (see openxr_convention.h).
struct JointPose
{
    float position[3];
    float orientation_xyzw[4];
};

//! The joint name for each FullBodyPose slot, indexed by core::BodyJoint.
using SlotNameTable = std::array<std::string_view, core::BodyJoint_NUM_JOINTS>;

/*!
 * @brief The joint name MOXI Player reports for each slot of the standard human skeleton. This
 *        table is the whole slot mapping.
 *
 * Only a joint carrying a controller rotates. The spine, foot, neck, collar and hand joints move
 * with the hierarchy: read them for position, not rotation.
 */
extern const SlotNameTable k_standard_human_skeleton_joint_names;

/*!
 * @brief Which joint fills each of the 24 slots.
 *
 * Split by lifetime: `resolve()` matches joint names against the table once per pairing, so
 * `apply()` does no lookups at all. Pure -- no SDK state and no vendor header.
 */
class BodyJointMapping
{
public:
    /*!
     * @brief Match this pairing's joint table against the name table.
     *
     * @param joint_names Every joint on the channel, in index order.
     */
    static BodyJointMapping resolve(const std::vector<std::string>& joint_names);

    /*!
     * @brief Fill the 24 slots from one frame's poses.
     *
     * A slot's `is_valid` says only whether it carries a usable pose on this frame; it does not
     * describe the device's sensing coverage.
     *
     * @param joint_poses This frame's poses, indexed as `resolve()`'s joint_names were. A frame
     *        that is short, empty, or carries values a consumer cannot use leaves the affected
     *        slots invalid; it never reads out of range and never ends the pairing.
     * @return How many slots went out valid.
     */
    size_t apply(const std::vector<JointPose>& joint_poses, core::BodyJoints& out) const;

    //! Index of the joint filling a slot, or -1 when this model has no joint of that name.
    int joint_index(core::BodyJoint slot) const;

    //! How many of the 24 slots this model fills. 24 for the standard human skeleton.
    size_t filled_slot_count() const
    {
        return filled_slot_count_;
    }

    /*!
     * @brief Joint names that matched a slot already taken by an earlier joint.
     *
     * Non-empty means the model's joint names are not the keys the table assumes. The earlier
     * joint keeps the slot; the pairing log names the rest.
     */
    const std::vector<std::string>& contested_joint_names() const
    {
        return contested_joint_names_;
    }

private:
    std::array<int, core::BodyJoint_NUM_JOINTS> slot_to_joint_{};
    std::vector<std::string> contested_joint_names_;
    size_t filled_slot_count_ = 0;
};

} // namespace jmex
} // namespace plugins

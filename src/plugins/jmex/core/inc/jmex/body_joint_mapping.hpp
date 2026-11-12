// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <schema/full_body_generated.h>

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace plugins
{
namespace jmex
{

//! The joint name for each FullBodyPose slot, indexed by core::BodyJoint.
using SlotNameTable = std::array<std::string_view, core::BodyJoint_NUM_JOINTS>;

/*!
 * @brief The joint name MOXI Player reports for each slot of the standard human skeleton. This
 *        table is the whole slot mapping.
 */
extern const SlotNameTable kStandardHumanSkeletonJointNames;

/*!
 * @brief Which joint fills each FullBodyPose slot.
 */
class BodyJointMapping
{
public:
    /*!
     * @brief Match this pairing's joint table against the name table.
     *
     * @param joint_names Every joint on the channel, in index order.
     */
    explicit BodyJointMapping(const std::vector<std::string>& joint_names);

    /*!
     * @brief Fill every slot from one frame's poses.
     *
     * @param joint_poses Unconstrained global poses in OpenXR axes, one per joint, in joint_names
     *        order. A pose a consumer cannot use leaves its slot invalid and never ends the pairing.
     * @return How many slots went out valid.
     */
    size_t apply(const std::vector<core::Pose>& joint_poses, core::BodyJoints& out) const;

    //! Index of the joint filling a slot, or nullopt when this model has no joint of that name.
    std::optional<size_t> joint_index(core::BodyJoint slot) const;

    //! How many joints the constructor's joint_names had.
    size_t joint_count() const
    {
        return joint_count_;
    }

    //! How many slots this model fills; every one for the standard human skeleton.
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
    std::array<std::optional<size_t>, core::BodyJoint_NUM_JOINTS> slot_to_joint_;
    std::vector<std::string> contested_joint_names_;
    size_t joint_count_ = 0;
    size_t filled_slot_count_ = 0;
};

} // namespace jmex
} // namespace plugins

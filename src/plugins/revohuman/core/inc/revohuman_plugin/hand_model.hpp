// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace plugins
{
namespace revohuman
{

//! The SDK link poses of a fitted model (kFittedModelHashes) mapped onto the OpenXR joints.
class HandModel
{
public:
    //! The model of a product the SDK identified ("P1", "PTV1"). Throws std::runtime_error when the SDK
    //! has no model for it or the model is not fitted.
    explicit HandModel(const std::string& product_code);

    //! For the logs: SDK model name and model hash.
    const std::string& id() const
    {
        return id_;
    }

    //! The OpenXR joints in the WRIST frame. The SDK fails only on arguments the constructor already checked,
    //! so a failure throws std::runtime_error.
    HandJointPoses joint_poses(Hand hand,
                               const std::array<double, kGloveJointCount>& joints_rad,
                               std::uint32_t joints_valid_mask) const;

    //! Verified TactileMetadata for the SDK taxel layout: one segment per OpenXR joint bound at zero
    //! joint angles, each taxel rigid on the joint that moves with its SDK link, +Z along its outward
    //! normal. Throws std::runtime_error when the SDK has no layout for this model.
    std::vector<std::uint8_t> tactile_metadata(Hand hand, std::uint32_t taxel_count) const;

private:
    std::int32_t sdk_model_ = 0;
    std::string id_;
};

} // namespace revohuman
} // namespace plugins

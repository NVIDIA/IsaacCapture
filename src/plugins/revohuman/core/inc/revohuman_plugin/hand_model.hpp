// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct RhHandPoints;

namespace plugins
{
namespace revohuman
{

//! The joints HandModel::joint_poses() gives for raw SDK hand points of a fitted model (kNewGeometryHashes).
//! Throws on another hash or an incompatible point set. @p retain_unmapped_positions keeps invalid joints' poses
//! for inspection instead of the identity.
HandJointPoses fit_hand_points(const std::string& model_sha256,
                               Hand hand,
                               const RhHandPoints& points,
                               bool retain_unmapped_positions = false);

//! The SDK link poses of a fitted model (kNewGeometryHashes) mapped onto the OpenXR joints.
class HandModel
{
public:
    //! The model of a product the SDK identified ("P1", "PTV1"). Throws std::runtime_error when the SDK
    //! has no model for it or the model is not fitted.
    static HandModel for_product(const std::string& product_code);

    //! For the logs: SDK model name, fitted table and model hash.
    const std::string& id() const
    {
        return id_;
    }

    //! The OpenXR joints in the WRIST frame. The SDK fails only on arguments for_product() already checked,
    //! so a failure throws std::runtime_error.
    HandJointPoses joint_poses(Hand hand,
                               const std::array<double, kGloveJointCount>& joints_rad,
                               std::uint32_t joints_valid_mask) const;

    //! Verified TactileMetadata for the SDK taxel layout: one segment per OpenXR joint bound at zero
    //! joint angles, each taxel rigid on the joint that moves with its SDK link, +Z along its outward
    //! normal. Throws std::runtime_error when the SDK has no layout for this model.
    std::vector<std::uint8_t> tactile_metadata(Hand hand, std::uint32_t taxel_count) const;

private:
    HandModel(std::int32_t sdk_model, std::string id) : sdk_model_(sdk_model), id_(std::move(id))
    {
    }

    std::int32_t sdk_model_;
    std::string id_;
};

} // namespace revohuman
} // namespace plugins

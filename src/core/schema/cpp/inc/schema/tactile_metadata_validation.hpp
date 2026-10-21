// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <schema/tactile_metadata_generated.h>

#include <array>
#include <cmath>

namespace core
{

//! Checks the semantics of the supported LBS/HandSkeletonRef layout after FlatBuffers verification.
//! Weight sums may differ from one by at most 1e-5 to allow floating-point rounding.
[[nodiscard]] inline bool verify_tactile_metadata(const TactileMetadata& metadata)
{
    const auto* lbs = metadata.geometry_mapping_as_LBS();
    const auto* hand = metadata.segment_source_as_HandSkeletonRef();
    if (lbs == nullptr || hand == nullptr)
    {
        return false;
    }

    const auto taxel_count = lbs->taxel_bind_poses()->size();
    if (lbs->taxel_segment_indices()->size() != taxel_count || lbs->taxel_segment_weights()->size() != taxel_count ||
        lbs->segment_bind_poses()->size() != hand->joint_ids()->size())
    {
        return false;
    }

    std::array<bool, HandJoint_NUM_JOINTS> seen{};
    for (const auto joint_id : *hand->joint_ids())
    {
        if (joint_id >= HandJoint_NUM_JOINTS || seen[joint_id])
        {
            return false;
        }
        seen[joint_id] = true;
    }

    for (const auto* weights : *lbs->taxel_segment_weights())
    {
        double sum = 0.0;
        for (const float weight : *weights->values())
        {
            if (!std::isfinite(weight) || weight < 0.0f)
            {
                return false;
            }
            sum += weight;
        }
        if (std::abs(sum - 1.0) > 1e-5)
        {
            return false;
        }
    }
    return true;
}

} // namespace core

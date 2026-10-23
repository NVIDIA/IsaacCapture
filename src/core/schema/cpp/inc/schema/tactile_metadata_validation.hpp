// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <schema/tactile_metadata_generated.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace core
{

//! Checks the semantics of the supported LBS/HandSkeletonRef layout after FlatBuffers verification.
//! Weight sums may differ from one by at most 1e-5 to allow floating-point rounding.
//! Throws std::runtime_error naming the invalid field and the violated constraint.
inline void verify_tactile_metadata(const TactileMetadata& metadata)
{
    const auto* lbs = metadata.geometry_mapping_as_LBS();
    const auto* hand = metadata.segment_source_as_HandSkeletonRef();
    if (lbs == nullptr)
    {
        throw std::runtime_error("TactileMetadata.geometry_mapping: expected LBS");
    }
    if (hand == nullptr)
    {
        throw std::runtime_error("TactileMetadata.segment_source: expected HandSkeletonRef");
    }

    const auto taxel_count = lbs->taxel_bind_poses()->size();
    if (lbs->taxel_segment_indices()->size() != taxel_count)
    {
        throw std::runtime_error("TactileMetadata.taxel_segment_indices: expected " + std::to_string(taxel_count) +
                                 " rows to match taxel_bind_poses, got " +
                                 std::to_string(lbs->taxel_segment_indices()->size()));
    }
    if (lbs->taxel_segment_weights()->size() != taxel_count)
    {
        throw std::runtime_error("TactileMetadata.taxel_segment_weights: expected " + std::to_string(taxel_count) +
                                 " rows to match taxel_bind_poses, got " +
                                 std::to_string(lbs->taxel_segment_weights()->size()));
    }
    if (lbs->segment_bind_poses()->size() != hand->joint_ids()->size())
    {
        throw std::runtime_error(
            "TactileMetadata.segment_bind_poses: expected " + std::to_string(hand->joint_ids()->size()) +
            " poses to match segment_source.joint_ids, got " + std::to_string(lbs->segment_bind_poses()->size()));
    }

    std::array<bool, HandJoint_NUM_JOINTS> seen{};
    for (flatbuffers::uoffset_t i = 0; i < hand->joint_ids()->size(); ++i)
    {
        const auto joint_id = hand->joint_ids()->Get(i);
        if (joint_id >= HandJoint_NUM_JOINTS)
        {
            throw std::runtime_error("TactileMetadata.joint_ids[" + std::to_string(i) +
                                     "]: invalid HandJoint identifier " + std::to_string(joint_id));
        }
        if (seen[joint_id])
        {
            throw std::runtime_error("TactileMetadata.joint_ids[" + std::to_string(i) +
                                     "]: duplicate HandJoint identifier " + std::to_string(joint_id));
        }
        seen[joint_id] = true;
    }

    for (flatbuffers::uoffset_t i = 0; i < taxel_count; ++i)
    {
        const auto* indices = lbs->taxel_segment_indices()->Get(i);
        for (flatbuffers::uoffset_t k = 0; k < indices->values()->size(); ++k)
        {
            const auto segment_index = indices->values()->Get(k);
            if (segment_index >= lbs->segment_bind_poses()->size())
            {
                throw std::runtime_error("TactileMetadata.taxel_segment_indices[" + std::to_string(i) + "][" +
                                         std::to_string(k) + "]: segment index " + std::to_string(segment_index) +
                                         " is outside segment_bind_poses (size " +
                                         std::to_string(lbs->segment_bind_poses()->size()) + ")");
            }
        }
    }

    for (flatbuffers::uoffset_t i = 0; i < taxel_count; ++i)
    {
        const auto* weights = lbs->taxel_segment_weights()->Get(i);
        double sum = 0.0;
        for (flatbuffers::uoffset_t k = 0; k < weights->values()->size(); ++k)
        {
            const float weight = weights->values()->Get(k);
            if (!std::isfinite(weight) || weight < 0.0f)
            {
                throw std::runtime_error("TactileMetadata.taxel_segment_weights[" + std::to_string(i) + "][" +
                                         std::to_string(k) + "]: weights must be finite and nonnegative");
            }
            sum += weight;
        }
        if (std::abs(sum - 1.0) > 1e-5)
        {
            throw std::runtime_error("TactileMetadata.taxel_segment_weights[" + std::to_string(i) +
                                     "]: weights must sum to one, got " + std::to_string(sum));
        }
    }
}

} // namespace core

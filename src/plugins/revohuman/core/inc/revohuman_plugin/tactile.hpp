// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.hpp"

#include <schema/tactile_metadata_generated.h>
#include <schema/tracker_metadata.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace plugins
{
namespace revohuman
{

//! Match the tracker metadata reader's IPC cap.
constexpr std::size_t kShapeMaxBytes = core::TRACKER_METADATA_MAX_SIZE;
constexpr std::size_t kTactileMaxBytes = 16384;
constexpr std::size_t kJointStateMaxBytes = 4096;

//! Three tables and six vectors, including 29 bytes per segment and 60 per taxel.
constexpr std::size_t shape_size_bound(std::size_t joints, std::size_t taxels)
{
    return 256 + 29 * joints + 60 * taxels;
}

//! Upper bound of a serialized TactileDataArray: table, one Point vector, slack.
constexpr std::size_t tactile_size_bound(std::size_t taxels)
{
    return 40 + 12 * taxels + 32;
}

//! LBS bind poses in WRIST space. Segment slots index joint_ids, not OpenXR ordinals.
struct ShapeSpec
{
    std::vector<Pose> bind_poses;
    std::vector<core::HandJoint> joint_ids;
    std::vector<Pose> taxel_bind_poses;
    std::vector<std::array<std::uint32_t, 4>> joint_indices;
    std::vector<std::array<float, 4>> joint_weights;
};

//! Serializes a bare TactileMetadata, without a Record wrapper.
std::vector<std::uint8_t> build_shape_buffer(const ShapeSpec& spec);

//! Validates LBS/HandSkeletonRef semantics and finite, unit bind poses.
std::vector<std::string> validate_shape(const core::TactileMetadata& shape);

//! TactileMetadata of a glove, verified and ready to publish and skin.
class LoadedShape
{
public:
    //! Requires valid metadata within the IPC cap and one rigid segment per taxel.
    explicit LoadedShape(std::vector<std::uint8_t> bytes);

    const std::vector<std::uint8_t>& bytes() const
    {
        return bytes_;
    }
    //! SHA-256 of bytes(), hex, to name the metadata in logs.
    const std::string& sha256() const
    {
        return sha256_;
    }
    std::uint32_t taxel_count() const
    {
        return taxel_count_;
    }
    const core::TactileMetadata& shape() const;

    //! Skinning of the bind positions with the current WRIST-frame joint poses. With rigid
    //! binding each taxel moves as T_current[j] * inverse(T_bind[j]) * p_bind.
    std::vector<std::array<double, 3>> skin_positions(const HandKinematics& current) const;
    //! Every positively weighted joint must have a finite, valid unit orientation.
    bool has_valid_orientations(const HandKinematics& current) const;
    //! Taxel normals at the current pose, unit length, in the WRIST frame.
    std::vector<std::array<double, 3>> current_normals(const HandKinematics& current) const;

private:
    std::vector<std::uint8_t> bytes_;
    std::string sha256_;
    std::uint32_t taxel_count_ = 0;
    std::vector<Pose> bind_inverse_;
};

//! The SDK's calibrated reading, unchanged, along each taxel's outward +Z: the normal force the skin exerts on
//! what it touches.
std::vector<std::array<float, 3>> taxel_forces(const std::vector<std::uint16_t>& raw);

//! The SDK reading recovered from a force vector: round(|f|).
long force_to_counts(const std::array<float, 3>& force);

} // namespace revohuman
} // namespace plugins

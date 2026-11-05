// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/revohuman_plugin/tactile.hpp"

#include "inc/revohuman_plugin/pose_math.hpp"
#include "inc/revohuman_plugin/sha256.hpp"

#include <flatbuffers/flatbuffers.h>
#include <schema/tactile_metadata_validation.hpp>

#include <cmath>
#include <stdexcept>

namespace plugins
{
namespace revohuman
{
namespace
{

Pose to_pose(const core::Pose& p)
{
    const auto& v = p.position();
    const auto& q = p.orientation();
    return { { v.x(), v.y(), v.z() }, { q.x(), q.y(), q.z(), q.w() } };
}

std::vector<core::Pose> encode_poses(const std::vector<Pose>& poses)
{
    std::vector<core::Pose> out;
    for (const auto& p : poses)
        out.emplace_back(core::Point(p.p[0], p.p[1], p.p[2]), core::Quaternion(p.q[0], p.q[1], p.q[2], p.q[3]));
    return out;
}

bool valid_pose(const core::Pose& encoded)
{
    const auto p = to_pose(encoded);
    double norm2 = 0;
    for (const auto v : p.p)
        if (!std::isfinite(v))
            return false;
    for (const auto v : p.q)
        norm2 += v * v;
    return std::isfinite(norm2) && std::abs(norm2 - 1) <= 1e-5;
}

} // namespace

std::vector<std::uint8_t> build_shape_buffer(const ShapeSpec& spec)
{
    core::LBST lbs;
    lbs.segment_bind_poses = encode_poses(spec.bind_poses);
    lbs.taxel_bind_poses = encode_poses(spec.taxel_bind_poses);
    for (const auto& slots : spec.joint_indices)
        lbs.taxel_segment_indices.emplace_back(flatbuffers::span<const std::uint32_t, 4>(slots));
    for (const auto& slots : spec.joint_weights)
        lbs.taxel_segment_weights.emplace_back(flatbuffers::span<const float, 4>(slots));
    core::HandSkeletonRefT hand;
    hand.joint_ids = spec.joint_ids;
    core::TactileMetadataT metadata;
    metadata.geometry_mapping.Set(std::move(lbs));
    metadata.segment_source.Set(std::move(hand));
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(core::TactileMetadata::Pack(builder, &metadata));
    return { builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize() };
}

std::vector<std::string> validate_shape(const core::TactileMetadata& shape)
{
    try
    {
        core::verify_tactile_metadata(shape);
    }
    catch (const std::runtime_error& error)
    {
        return { std::string("invalid LBS/HandSkeletonRef metadata: ") + error.what() };
    }
    const auto& lbs = *shape.geometry_mapping_as_LBS();
    if (lbs.segment_bind_poses()->size() == 0 || lbs.taxel_bind_poses()->size() == 0)
        return { "metadata must contain segments and taxels" };
    for (const auto* poses : { lbs.segment_bind_poses(), lbs.taxel_bind_poses() })
        for (const auto* p : *poses)
            if (!valid_pose(*p))
                return { "bind pose position must be finite and orientation must be unit" };
    return {};
}

LoadedShape::LoadedShape(std::vector<std::uint8_t> bytes)
    : bytes_(std::move(bytes)), sha256_(sha256_hex(bytes_.data(), bytes_.size()))
{
    if (bytes_.size() > kShapeMaxBytes)
        throw std::runtime_error("metadata exceeds the tracker IPC cap");
    flatbuffers::Verifier verifier(bytes_.data(), bytes_.size());
    if (!verifier.VerifyBuffer<core::TactileMetadata>(nullptr))
        throw std::runtime_error("metadata: buffer does not verify as TactileMetadata");
    const auto issues = validate_shape(shape());
    if (!issues.empty())
        throw std::runtime_error("metadata: " + issues.front());
    const auto& lbs = *shape().geometry_mapping_as_LBS();
    taxel_count_ = lbs.taxel_bind_poses()->size();
    if (taxel_count_ > kMaxTaxels)
        throw std::runtime_error("metadata taxel count exceeds the glove protocol limit");
    for (const auto* weights : *lbs.taxel_segment_weights())
    {
        int nonzero = 0;
        for (const auto w : *weights->values())
            nonzero += w > 0 ? 1 : 0;
        if (nonzero != 1)
            throw std::runtime_error("metadata: only rigid binding (one segment, weight 1) is supported");
    }
    for (const auto* p : *lbs.segment_bind_poses())
        bind_inverse_.push_back(inverse(to_pose(*p)));
}

const core::TactileMetadata& LoadedShape::shape() const
{
    return *flatbuffers::GetRoot<core::TactileMetadata>(bytes_.data());
}

std::vector<std::array<double, 3>> LoadedShape::skin_positions(const HandKinematics& current) const
{
    const auto& lbs = *shape().geometry_mapping_as_LBS();
    const auto* ids = shape().segment_source_as_HandSkeletonRef()->joint_ids();
    std::vector<Vec3> out(taxel_count_, { 0, 0, 0 });
    for (std::uint32_t i = 0; i < taxel_count_; ++i)
    {
        const auto bind = to_pose(*lbs.taxel_bind_poses()->Get(i));
        for (int k = 0; k < 4; ++k)
        {
            const double w = lbs.taxel_segment_weights()->Get(i)->values()->Get(k);
            if (w == 0)
                continue;
            const auto segment = lbs.taxel_segment_indices()->Get(i)->values()->Get(k);
            const auto joint = ids->Get(segment);
            const auto moved = transform_point(current.joints[joint], transform_point(bind_inverse_[segment], bind.p));
            for (int c = 0; c < 3; ++c)
                out[i][c] += w * moved[c];
        }
    }
    return out;
}

bool LoadedShape::has_valid_orientations(const HandKinematics& current) const
{
    const auto& lbs = *shape().geometry_mapping_as_LBS();
    const auto* ids = shape().segment_source_as_HandSkeletonRef()->joint_ids();
    for (std::uint32_t i = 0; i < taxel_count_; ++i)
        for (int k = 0; k < 4; ++k)
        {
            if (lbs.taxel_segment_weights()->Get(i)->values()->Get(k) == 0)
                continue;
            const auto joint = ids->Get(lbs.taxel_segment_indices()->Get(i)->values()->Get(k));
            if (!(current.orientation_valid_mask & (1u << joint)))
                return false;
            double norm2 = 0;
            for (const auto v : current.joints[joint].q)
                norm2 += v * v;
            if (!std::isfinite(norm2) || std::abs(norm2 - 1) > 1e-5)
                return false;
        }
    return true;
}

std::vector<std::array<double, 3>> LoadedShape::current_normals(const HandKinematics& current) const
{
    const auto& lbs = *shape().geometry_mapping_as_LBS();
    const auto* ids = shape().segment_source_as_HandSkeletonRef()->joint_ids();
    std::vector<Vec3> out(taxel_count_, { 0, 0, 0 });
    for (std::uint32_t i = 0; i < taxel_count_; ++i)
    {
        const auto normal = quat_rotate(to_pose(*lbs.taxel_bind_poses()->Get(i)).q, { 0, 0, 1 });
        for (int k = 0; k < 4; ++k)
        {
            const double w = lbs.taxel_segment_weights()->Get(i)->values()->Get(k);
            if (w == 0)
                continue;
            const auto segment = lbs.taxel_segment_indices()->Get(i)->values()->Get(k);
            const auto moved =
                quat_rotate(current.joints[ids->Get(segment)].q, quat_rotate(bind_inverse_[segment].q, normal));
            for (int c = 0; c < 3; ++c)
                out[i][c] += w * moved[c];
        }
        out[i] = normalize(out[i]);
    }
    return out;
}

std::vector<std::array<float, 3>> taxel_forces(const std::vector<std::uint16_t>& raw)
{
    std::vector<std::array<float, 3>> out;
    out.reserve(raw.size());
    for (const auto value : raw)
        out.push_back({ 0, 0, static_cast<float>(value) });
    return out;
}

long force_to_counts(const std::array<float, 3>& force)
{
    const double x = force[0], y = force[1], z = force[2];
    return std::lround(std::sqrt(x * x + y * y + z * z));
}

} // namespace revohuman
} // namespace plugins

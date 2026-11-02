// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/revohuman_plugin/serialize.hpp"

#include "inc/revohuman_plugin/tactile.hpp"

#include <schema/joint_state_generated.h>
#include <schema/tactile_array_generated.h>

#include <memory>
#include <stdexcept>

namespace plugins
{
namespace revohuman
{
namespace
{

static_assert(tactile_size_bound(kMaxTaxels) <= kTactileMaxBytes, "350 taxels must fit one TactileDataArray push");

std::vector<std::uint8_t> finish(flatbuffers::FlatBufferBuilder& builder, std::size_t cap, const char* what)
{
    if (builder.GetSize() > cap)
    {
        throw std::length_error(std::string(what) + ": " + std::to_string(builder.GetSize()) + " bytes exceeds " +
                                std::to_string(cap));
    }
    return std::vector<std::uint8_t>(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
}

} // namespace

std::vector<std::uint8_t> serialize_joint_state(const std::string& device_id,
                                                const std::array<std::string, kGloveJointCount>& names,
                                                const std::array<double, kGloveJointCount>& joints_rad,
                                                std::uint32_t joints_valid_mask)
{
    core::JointStateOutputT out;
    out.device_id = device_id;
    out.has_velocity = false;
    out.has_effort = false;
    out.ee_pose_valid = false;
    for (std::size_t i = 0; i < kGloveJointCount; ++i)
    {
        auto joint = std::make_shared<core::JointStateT>();
        joint->name = names[i];
        joint->position = static_cast<float>(joints_rad[i]);
        joint->valid = ((joints_valid_mask >> i) & 1u) != 0;
        out.joints.push_back(std::move(joint));
    }
    flatbuffers::FlatBufferBuilder builder(kJointStateMaxBytes);
    builder.Finish(core::JointStateOutput::Pack(builder, &out));
    return finish(builder, kJointStateMaxBytes, "joint_state");
}

std::vector<std::uint8_t> serialize_tactile(const std::vector<std::array<float, 3>>& forces)
{
    std::vector<core::Point> points;
    points.reserve(forces.size());
    for (const auto& f : forces)
    {
        points.emplace_back(f[0], f[1], f[2]);
    }
    flatbuffers::FlatBufferBuilder builder(kTactileMaxBytes);
    builder.Finish(core::CreateTactileDataArrayDirect(builder, &points));
    return finish(builder, kTactileMaxBytes, "tactile");
}

} // namespace revohuman
} // namespace plugins

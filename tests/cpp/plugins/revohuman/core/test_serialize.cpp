// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <revohuman_plugin/serialize.hpp>
#include <revohuman_plugin/tactile.hpp>
#include <schema/joint_state_generated.h>
#include <schema/tactile_array_generated.h>

#include <stdexcept>
#include <string>

using namespace plugins::revohuman;

namespace
{

template <typename T>
const T* verified_root(const std::vector<std::uint8_t>& bytes)
{
    flatbuffers::Verifier verifier(bytes.data(), bytes.size());
    REQUIRE(verifier.VerifyBuffer<T>(nullptr));
    return flatbuffers::GetRoot<T>(bytes.data());
}

std::array<std::string, kGloveJointCount> joint_names()
{
    std::array<std::string, kGloveJointCount> names;
    for (std::size_t i = 0; i < kGloveJointCount; ++i)
    {
        names[i] = "right_joint_" + std::to_string(20 - i);
    }
    return names;
}

} // namespace

TEST_CASE("joint state keeps SDK order, float radians and the valid mask", "[revohuman][serialize]")
{
    const auto names = joint_names();
    std::array<double, kGloveJointCount> rad{};
    for (std::size_t i = 0; i < kGloveJointCount; ++i)
    {
        rad[i] = 0.123456789012345 * static_cast<double>(i) - 1.0;
    }
    const std::uint32_t mask = 0x1FFFFFu & ~((1u << 3) | (1u << 17));

    const auto bytes = serialize_joint_state("revohuman_right", names, rad, mask);
    REQUIRE(bytes.size() <= kJointStateMaxBytes);
    const auto* out = verified_root<core::JointStateOutput>(bytes);
    REQUIRE(out->device_id()->str() == "revohuman_right");
    REQUIRE_FALSE(out->has_velocity());
    REQUIRE_FALSE(out->has_effort());
    REQUIRE_FALSE(out->ee_pose_valid());
    REQUIRE(out->joints()->size() == kGloveJointCount);
    for (std::size_t i = 0; i < kGloveJointCount; ++i)
    {
        const auto* j = out->joints()->Get(static_cast<flatbuffers::uoffset_t>(i));
        REQUIRE(j->name()->str() == names[i]);
        REQUIRE(j->position() == static_cast<float>(rad[i]));
        REQUIRE(j->valid() == ((mask >> i) & 1u));
    }
}

TEST_CASE("joint state over its cap is refused", "[revohuman][serialize]")
{
    auto names = joint_names();
    for (auto& n : names)
    {
        n = std::string(300, 'j');
    }
    REQUIRE_THROWS_AS(serialize_joint_state("x", names, {}, 0), std::length_error);
}

TEST_CASE("tactile data carries the count and every force", "[revohuman][serialize]")
{
    std::vector<std::array<float, 3>> forces(kMaxTaxels);
    for (std::size_t i = 0; i < forces.size(); ++i)
    {
        forces[i] = { 0.001f * i, -0.5f, static_cast<float>(i) };
    }
    const auto bytes = serialize_tactile(forces);
    REQUIRE(bytes.size() <= tactile_size_bound(kMaxTaxels));
    REQUIRE(bytes.size() <= kTactileMaxBytes);
    const auto* out = verified_root<core::TactileDataArray>(bytes);
    REQUIRE(out->taxel_forces()->size() == kMaxTaxels);
    for (std::size_t i = 0; i < forces.size(); ++i)
    {
        const auto* f = out->taxel_forces()->Get(static_cast<flatbuffers::uoffset_t>(i));
        REQUIRE(f->x() == forces[i][0]);
        REQUIRE(f->y() == forces[i][1]);
        REQUIRE(f->z() == forces[i][2]);
    }

    REQUIRE(verified_root<core::TactileDataArray>(serialize_tactile({}))->taxel_forces()->size() == 0);
    REQUIRE_THROWS_AS(serialize_tactile(std::vector<std::array<float, 3>>(2000)), std::length_error);
}

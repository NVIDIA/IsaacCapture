// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <revohuman_plugin/rh_frame.hpp>

#include <cstring>
#include <stdexcept>

using namespace plugins::revohuman;

namespace
{

RhGloveFrame blank_frame()
{
    RhGloveFrame f;
    std::memset(&f, 0, sizeof(f));
    f.struct_size = sizeof(RhGloveFrame);
    f.hand = RH_HAND_RIGHT;
    return f;
}

std::array<double, kGloveJointCount> radians()
{
    std::array<double, kGloveJointCount> rad{};
    for (std::size_t i = 0; i < kGloveJointCount; ++i)
    {
        rad[i] = 0.05 * static_cast<double>(i);
    }
    return rad;
}

} // namespace

TEST_CASE("SDK frame without tactile converts its header fields", "[revohuman][rh_frame]")
{
    RhGloveFrame f = blank_frame();
    f.generation = 4;
    f.device_tick = 0xFFFFFFF0u;
    f.joints_valid_mask = 0x1FFFFFu;
    f.flags = RH_FRAME_FLAG_GATE_OPEN | RH_FRAME_FLAG_HAS_SAMPLE_TIME;
    f.sample_mono_ns = 987'654'321;

    const GloveSample g = from_rh_frame(f, radians());
    REQUIRE(g.hand == Hand::Right);
    REQUIRE(g.generation == 4);
    REQUIRE(g.device_tick == 0xFFFFFFF0u);
    REQUIRE(g.joints_valid_mask == 0x1FFFFFu);
    REQUIRE(g.has_sample_time);
    REQUIRE(g.sample_mono_ns == 987'654'321);
    REQUIRE(g.joints_rad == radians());
    REQUIRE_FALSE(g.tactile);
    REQUIRE_FALSE(g.kinematics);
}

TEST_CASE("SDK frame tactile block is copied with its flags", "[revohuman][rh_frame]")
{
    RhGloveFrame f = blank_frame();
    f.hand = RH_HAND_LEFT;
    f.flags = RH_FRAME_FLAG_HAS_TACTILE;
    f.tactile.point_count = RH_TACTILE_POINTS_TYPE_B;
    f.tactile.valid = 1;
    f.tactile.calibrated = 1;
    f.tactile.has_sample_time = 1;
    f.tactile.device_tick = 55;
    f.tactile.sample_mono_ns = 111;
    for (int i = 0; i < RH_TACTILE_POINTS_TYPE_B; ++i)
    {
        f.tactile.values[i] = static_cast<std::uint16_t>(1000 + i);
    }

    const GloveSample g = from_rh_frame(f, radians());
    REQUIRE(g.hand == Hand::Left);
    REQUIRE_FALSE(g.has_sample_time);
    REQUIRE(g.tactile);
    REQUIRE(g.tactile->values.size() == RH_TACTILE_POINTS_TYPE_B);
    REQUIRE(g.tactile->values.front() == 1000);
    REQUIRE(g.tactile->values.back() == 1000 + RH_TACTILE_POINTS_TYPE_B - 1);
    REQUIRE(g.tactile->valid);
    REQUIRE(g.tactile->calibrated);
    REQUIRE(g.tactile->has_sample_time);
    REQUIRE(g.tactile->device_tick == 55);
    REQUIRE(g.tactile->sample_mono_ns == 111);

    f.tactile.calibrated = 0;
    REQUIRE_FALSE(from_rh_frame(f, radians()).tactile->calibrated);
}

TEST_CASE("SDK frame conversion rejects what it cannot trust", "[revohuman][rh_frame]")
{
    RhGloveFrame f = blank_frame();
    f.struct_size = sizeof(RhGloveFrame) - 8;
    REQUIRE_THROWS_AS(from_rh_frame(f, radians()), std::invalid_argument);

    f = blank_frame();
    f.hand = RH_HAND_UNKNOWN;
    REQUIRE_THROWS_AS(from_rh_frame(f, radians()), std::invalid_argument);

    f = blank_frame();
    f.flags = RH_FRAME_FLAG_HAS_TACTILE;
    f.tactile.point_count = RH_TACTILE_MAX_POINTS + 1;
    REQUIRE_THROWS_AS(from_rh_frame(f, radians()), std::invalid_argument);

    f.flags = 0;
    REQUIRE_NOTHROW(from_rh_frame(f, radians()));
}

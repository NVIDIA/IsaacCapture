// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <spacemouse/hid_report.hpp>

#include <array>
#include <cstdint>
#include <vector>

using plugins::spacemouse::apply_report;
using plugins::spacemouse::SpaceMouseState;

namespace
{

// Little-endian 16-bit axis count.
std::array<uint8_t, 2> le(int16_t value)
{
    const auto raw = static_cast<uint16_t>(value);
    return { static_cast<uint8_t>(raw & 0xFF), static_cast<uint8_t>(raw >> 8) };
}

std::vector<uint8_t> axes_report(uint8_t id, int16_t x, int16_t y, int16_t z)
{
    std::vector<uint8_t> report{ id };
    for (int16_t value : { x, y, z })
    {
        const auto bytes = le(value);
        report.insert(report.end(), bytes.begin(), bytes.end());
    }
    return report;
}

bool apply_hid_report(SpaceMouseState& state, const std::vector<uint8_t>& report, bool combined = false)
{
    return apply_report(state, report.data(), report.size(), combined);
}

} // namespace

TEST_CASE("Separate reports carry translation and rotation", "[spacemouse][hid]")
{
    SpaceMouseState state;
    REQUIRE(apply_hid_report(state, axes_report(1, 175, -350, 0)));
    REQUIRE(apply_hid_report(state, axes_report(2, -175, 0, 350)));

    CHECK(state.translation[0] == Catch::Approx(0.5f));
    CHECK(state.translation[1] == Catch::Approx(-1.0f));
    CHECK(state.translation[2] == Catch::Approx(0.0f));
    CHECK(state.rotation[0] == Catch::Approx(-0.5f));
    CHECK(state.rotation[2] == Catch::Approx(1.0f));
}

TEST_CASE("Axes beyond full deflection are clamped", "[spacemouse][hid]")
{
    SpaceMouseState state;
    REQUIRE(apply_hid_report(state, axes_report(1, 700, -32768, 351)));

    CHECK(state.translation == std::array<float, 3>{ 1.0f, -1.0f, 1.0f });
}

TEST_CASE("A combined report carries both translation and rotation", "[spacemouse][hid]")
{
    std::vector<uint8_t> report = axes_report(1, 35, 70, 105);
    const auto rotation = axes_report(1, -35, -70, -105);
    report.insert(report.end(), rotation.begin() + 1, rotation.end());

    SpaceMouseState state;
    REQUIRE(apply_hid_report(state, report, true));

    CHECK(state.translation[2] == Catch::Approx(0.3f));
    CHECK(state.rotation[0] == Catch::Approx(-0.1f));
    CHECK(state.rotation[2] == Catch::Approx(-0.3f));
}

TEST_CASE("A combined-report device ignores separate rotation and short translation reports", "[spacemouse][hid]")
{
    SpaceMouseState state;
    CHECK_FALSE(apply_hid_report(state, axes_report(2, 350, 350, 350), true));
    CHECK_FALSE(apply_hid_report(state, axes_report(1, 350, 350, 350), true)); // 7 bytes, a combined report has 13
    CHECK(state.rotation == std::array<float, 3>{});
    CHECK(state.translation == std::array<float, 3>{});
}

TEST_CASE("The button report is a bitmask over every byte after the report ID", "[spacemouse][hid]")
{
    SpaceMouseState state;
    REQUIRE(apply_hid_report(state, { 3, 0b0000'0010, 0b0000'0001 }));

    REQUIRE(state.buttons.size() == 16);
    CHECK(state.buttons[0] == 0);
    CHECK(state.buttons[1] == 1);
    CHECK(state.buttons[8] == 1);

    REQUIRE(apply_hid_report(state, { 3, 0, 0 }));
    CHECK(state.buttons == std::vector<uint8_t>(16, 0));
}

TEST_CASE("A long button report is capped at kMaxButtons", "[spacemouse][hid]")
{
    std::vector<uint8_t> report(64, 0xFF);
    report[0] = 3;

    SpaceMouseState state;
    REQUIRE(apply_hid_report(state, report));

    CHECK(state.buttons == std::vector<uint8_t>(plugins::spacemouse::kMaxButtons, 1));
}

TEST_CASE("Unknown, empty and truncated reports change nothing", "[spacemouse][hid]")
{
    SpaceMouseState state;
    CHECK_FALSE(apply_hid_report(state, {}));
    CHECK_FALSE(apply_hid_report(state, { 9, 1, 2, 3, 4, 5, 6 }));
    CHECK_FALSE(apply_hid_report(state, { 1, 0xFF, 0xFF }));
    CHECK_FALSE(apply_hid_report(state, { 3 }));
    CHECK(state.translation == std::array<float, 3>{});
    CHECK(state.buttons.empty());
}

TEST_CASE("clear() leaves zero axes and released buttons", "[spacemouse][hid]")
{
    SpaceMouseState state;
    apply_hid_report(state, axes_report(1, 350, 350, 350));
    apply_hid_report(state, { 3, 0xFF });
    state.clear();

    CHECK(state.translation == std::array<float, 3>{});
    CHECK(state.buttons == std::vector<uint8_t>(8, 0));
}

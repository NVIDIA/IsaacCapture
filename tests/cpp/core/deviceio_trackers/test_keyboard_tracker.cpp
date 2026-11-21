// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The keyboard's key table: W3C KeyboardEvent.code <-> evdev code. Provider and session behavior
// is covered by tests/cpp/core/live_trackers/test_keyboard_session.cpp.

#include <catch2/catch_test_macros.hpp>
#include <deviceio_trackers/keyboard_tracker.hpp>

#include <cstdint>
#include <string_view>

namespace
{

constexpr uint16_t KEY_W = 17;

} // namespace

TEST_CASE("evdev_code_from_w3c maps standard keys", "[unit][keyboard]")
{
    CHECK(core::evdev_code_from_w3c("KeyW") == KEY_W);
    CHECK(core::evdev_code_from_w3c("ArrowUp") == uint16_t{ 103 });
    CHECK(core::evdev_code_from_w3c("Numpad8") == uint16_t{ 72 });
    CHECK_FALSE(core::evdev_code_from_w3c("NotAKey").has_value());
}

TEST_CASE("w3c_code_from_evdev names keys from Chromium's key table", "[unit][keyboard]")
{
    CHECK(core::w3c_code_from_evdev(KEY_W) == std::string_view("KeyW"));
    CHECK(core::w3c_code_from_evdev(127) == std::string_view("ContextMenu"));
    CHECK(core::w3c_code_from_evdev(86) == std::string_view("IntlBackslash"));
    CHECK_FALSE(core::w3c_code_from_evdev(0).has_value());
}

TEST_CASE("keyboard_key_codes: every key round-trips", "[unit][keyboard]")
{
    const auto& keys = core::keyboard_key_codes();
    REQUIRE(keys.size() > 100);
    for (const auto& key : keys)
    {
        CHECK(core::evdev_code_from_w3c(key.w3c_code) == key.evdev_code);
        const auto name = core::w3c_code_from_evdev(key.evdev_code);
        REQUIRE(name.has_value());
        CHECK(core::evdev_code_from_w3c(*name) == key.evdev_code);
        CHECK(key.evdev_code < core::kKeyboardKeyCodeCount);
    }
}

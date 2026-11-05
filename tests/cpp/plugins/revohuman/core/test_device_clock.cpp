// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <revohuman_plugin/device_clock.hpp>

using namespace plugins::revohuman;

TEST_CASE("tick unwrapper anchors on the first value", "[revohuman][clock]")
{
    TickUnwrapper u;
    REQUIRE_FALSE(u.anchored());
    REQUIRE(u.unwrap(1000) == 1000);
    REQUIRE(u.anchored());
    REQUIRE(u.unwrap(1005) == 1005);
}

TEST_CASE("tick unwrapper carries across the 32-bit wrap", "[revohuman][clock]")
{
    TickUnwrapper u;
    REQUIRE(u.unwrap(0xFFFFFFF0u) == 0xFFFFFFF0ll);
    REQUIRE(u.unwrap(0x00000010u) == 0x100000010ll);
    REQUIRE(u.unwrap(0x00000020u) == 0x100000020ll);
}

TEST_CASE("tick unwrapper follows a backward step instead of wrapping", "[revohuman][clock]")
{
    TickUnwrapper u;
    u.unwrap(100);
    REQUIRE(u.unwrap(90) == 90);
    REQUIRE(u.unwrap(95) == 95);
    u.reset();
    REQUIRE_FALSE(u.anchored());
    REQUIRE(u.unwrap(7) == 7);
}

TEST_CASE("tick conversions", "[revohuman][clock]")
{
    REQUIRE(ms_ticks_to_ns(5) == 5'000'000);
    REQUIRE(ms_ticks_to_ns(-2) == -2'000'000);
}

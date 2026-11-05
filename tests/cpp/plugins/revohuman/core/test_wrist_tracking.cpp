// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <revohuman_plugin/wrist_tracking.hpp>

using namespace plugins::revohuman;

namespace
{

constexpr std::int64_t kMs = 1'000'000;

} // namespace

TEST_CASE("injector is created on the first frame and kept while frames arrive", "[revohuman][wrist]")
{
    InjectorLifecycle life(50 * kMs);
    REQUIRE_FALSE(life.active());

    auto s = life.on_frame(0);
    REQUIRE(s.create);
    REQUIRE_FALSE(s.destroy);
    REQUIRE(s.reason == "first_frame");
    REQUIRE(life.active());

    s = life.on_frame(5 * kMs);
    REQUIRE_FALSE(s.create);
    REQUIRE_FALSE(s.destroy);
    REQUIRE(s.reason.empty());
    REQUIRE(life.active());
}

TEST_CASE("injector is destroyed when pushes stall and recreated by the next frame", "[revohuman][wrist]")
{
    InjectorLifecycle life(50 * kMs);
    REQUIRE_FALSE(life.on_tick(1000 * kMs).destroy);

    life.on_frame(100 * kMs);
    REQUIRE_FALSE(life.on_tick(150 * kMs).destroy);
    const auto s = life.on_tick(151 * kMs);
    REQUIRE(s.destroy);
    REQUIRE(s.reason == "stall");
    REQUIRE_FALSE(life.active());
    REQUIRE_FALSE(life.on_tick(300 * kMs).destroy);

    const auto again = life.on_frame(310 * kMs);
    REQUIRE(again.create);
    REQUIRE(again.reason == "first_frame");
    REQUIRE(life.active());
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <avatar/avatar_stream_health.hpp>
#include <catch2/catch_test_macros.hpp>

using plugins::avatar::DatasetFreshness;
using plugins::avatar::GloveWaitReminder;
using namespace std::chrono_literals;

TEST_CASE("Glove wait reminders continue per side until streaming", "[sharpa_avatar][lifecycle]")
{
    const auto start = GloveWaitReminder::Clock::time_point{};
    GloveWaitReminder left;
    GloveWaitReminder right;

    CHECK(left.observe(false, start, 60s));
    CHECK(right.observe(false, start, 60s));
    CHECK_FALSE(left.observe(true, start + 2s, 60s));
    CHECK_FALSE(right.observe(false, start + 2s, 60s));
    CHECK(right.observe(false, start + 60s, 60s));
    CHECK_FALSE(left.observe(true, start + 60s, 60s));
    CHECK_FALSE(right.observe(true, start + 61s, 60s));
    CHECK(right.observe(false, start + 62s, 60s));
}

TEST_CASE("Dataset output expires when its timestamp stops advancing", "[sharpa_avatar][health]")
{
    const auto start = DatasetFreshness::Clock::time_point{};
    DatasetFreshness dataset;
    CHECK_FALSE(dataset.fresh(start));
    dataset.observe(1, 0, start);
    CHECK(dataset.fresh(start + 2999ms));
    dataset.observe(1, 0, start + 3s);
    CHECK_FALSE(dataset.fresh(start + 3s));
    dataset.observe(1, 1, start + 3s);
    CHECK(dataset.fresh(start + 3s));
    dataset = {};
    CHECK_FALSE(dataset.fresh(start + 3s));
}

TEST_CASE("Fresh datasets do not extend a stalled dataset's lifetime", "[sharpa_avatar][health]")
{
    const auto start = DatasetFreshness::Clock::time_point{};
    DatasetFreshness human;
    DatasetFreshness raw;
    DatasetFreshness robot;
    human.observe(1, 0, start);
    raw.observe(1, 0, start);
    robot.observe(1, 0, start);
    human.observe(1, 0, start + 3s);
    raw.observe(2, 0, start + 3s);
    robot.observe(2, 0, start + 3s);
    CHECK_FALSE(human.fresh(start + 3s));
    CHECK(raw.fresh(start + 3s));
    CHECK(robot.fresh(start + 3s));
    human.observe(0, 0, start + 4s);
    CHECK(human.fresh(start + 4s));
}

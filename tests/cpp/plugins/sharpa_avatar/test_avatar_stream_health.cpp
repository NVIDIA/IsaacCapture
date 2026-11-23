// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <avatar/avatar_stream_health.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <memory>
#include <stdexcept>

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

namespace
{

struct OutputCounters
{
    int created = 0;
    int withdrawn = 0;
    bool active = false;
};

struct CapturingOutput
{
    explicit CapturingOutput(OutputCounters& counters) : counters(counters)
    {
        ++counters.created;
        counters.active = true;
    }

    ~CapturingOutput()
    {
        ++counters.withdrawn;
        counters.active = false;
    }

    OutputCounters& counters;
};

struct OutputHarness
{
    OutputCounters counters;
    DatasetFreshness freshness;
    plugins::avatar::StreamOutputRecovery recovery;
    std::unique_ptr<CapturingOutput> output;
    bool has_payload = true;
    bool ready = true;
    bool fail_setup = false;
    bool unknown_failure = false;
    int attempts = 0;

    CapturingOutput* update(DatasetFreshness::Clock::time_point now)
    {
        return recovery.update(output, ready && has_payload && freshness.fresh(now), now, 2s,
                               [this]
                               {
                                   ++attempts;
                                   if (unknown_failure)
                                   {
                                       throw 1;
                                   }
                                   if (fail_setup)
                                   {
                                       throw std::runtime_error("runtime unavailable");
                                   }
                                   return std::make_unique<CapturingOutput>(counters);
                               });
    }
};

} // namespace

TEST_CASE("Stale outputs are withdrawn and recreated only after a changed stamp", "[sharpa_avatar][lifecycle]")
{
    const auto start = DatasetFreshness::Clock::time_point{};
    OutputHarness stream;
    CHECK(stream.update(start) == nullptr);
    stream.freshness.observe(1, 0, start);
    REQUIRE(stream.update(start) != nullptr);
    CHECK(stream.counters.active);
    CHECK(stream.update(start + 2999ms) == stream.output.get());
    CHECK(stream.attempts == 1);

    stream.freshness.observe(1, 0, start + 3s);
    CHECK(stream.update(start + 3s) == nullptr);
    CHECK_FALSE(stream.counters.active);
    CHECK(stream.counters.withdrawn == 1);
    CHECK(stream.update(start + 4s) == nullptr);
    CHECK(stream.counters.withdrawn == 1);
    CHECK(stream.attempts == 1);

    stream.freshness.observe(2, 0, start + 4s);
    REQUIRE(stream.update(start + 4s) != nullptr);
    CHECK(stream.counters.active);
    CHECK(stream.counters.created == 2);
}

TEST_CASE("Output withdrawal is independent for each side and dataset", "[sharpa_avatar][lifecycle]")
{
    const auto start = DatasetFreshness::Clock::time_point{};
    std::array<std::array<OutputHarness, 3>, 2> streams;
    for (auto& side : streams)
    {
        for (auto& stream : side)
        {
            stream.freshness.observe(1, 0, start);
            REQUIRE(stream.update(start) != nullptr);
        }
    }
    for (size_t side = 0; side < streams.size(); ++side)
    {
        for (size_t category = 0; category < streams[side].size(); ++category)
        {
            auto& stream = streams[side][category];
            if (side != 0 || category == 1)
            {
                stream.freshness.observe(2, 0, start + 3s);
            }
            stream.update(start + 3s);
            CHECK(stream.counters.active == (side != 0 || category == 1));
        }
    }
    streams[0][0].freshness.observe(3, 0, start + 4s);
    REQUIRE(streams[0][0].update(start + 4s) != nullptr);
    CHECK(streams[0][2].counters.created == 1);
    CHECK(streams[0][1].counters.withdrawn == 0);
    CHECK(streams[1][0].counters.created == 1);
}

TEST_CASE("Missing payload or physical readiness withdraws a fresh output", "[sharpa_avatar][lifecycle]")
{
    const auto start = DatasetFreshness::Clock::time_point{};
    OutputHarness stream;
    stream.freshness.observe(1, 0, start);
    REQUIRE(stream.update(start) != nullptr);
    SECTION("empty or failed fetch")
    {
        stream.has_payload = false;
    }
    SECTION("physical disconnect")
    {
        stream.ready = false;
    }
    CHECK(stream.update(start + 1ms) == nullptr);
    CHECK_FALSE(stream.counters.active);
    CHECK(stream.counters.withdrawn == 1);
    stream.has_payload = true;
    stream.ready = true;
    REQUIRE(stream.update(start + 2ms) != nullptr);
    CHECK(stream.counters.created == 2);
}

TEST_CASE("Output recreation failures are isolated and retries are bounded", "[sharpa_avatar][lifecycle]")
{
    const auto start = DatasetFreshness::Clock::time_point{};
    OutputHarness failed;
    OutputHarness healthy;
    failed.freshness.observe(1, 0, start);
    healthy.freshness.observe(1, 0, start);
    SECTION("standard exception")
    {
        failed.fail_setup = true;
    }
    SECTION("vendor exception")
    {
        failed.unknown_failure = true;
    }
    CHECK(failed.update(start) == nullptr);
    REQUIRE(failed.recovery.error().has_value());
    REQUIRE(healthy.update(start) != nullptr);
    CHECK(healthy.counters.active);
    CHECK(failed.update(start + 1999ms) == nullptr);
    CHECK(failed.attempts == 1);

    failed.fail_setup = false;
    failed.unknown_failure = false;
    REQUIRE(failed.update(start + 2s) != nullptr);
    CHECK(failed.attempts == 2);
    CHECK_FALSE(failed.recovery.error().has_value());
    CHECK(healthy.counters.created == 1);
}

TEST_CASE("Intermittent missing data cannot bypass output setup backoff", "[sharpa_avatar][lifecycle]")
{
    const auto start = DatasetFreshness::Clock::time_point{};
    OutputHarness stream;
    stream.freshness.observe(1, 0, start);
    stream.fail_setup = true;
    CHECK(stream.update(start) == nullptr);
    stream.has_payload = false;
    CHECK(stream.update(start + 1ms) == nullptr);
    stream.has_payload = true;
    stream.fail_setup = false;
    CHECK(stream.update(start + 2ms) == nullptr);
    CHECK(stream.attempts == 1);
    REQUIRE(stream.update(start + 2s) != nullptr);
    CHECK(stream.attempts == 2);
}

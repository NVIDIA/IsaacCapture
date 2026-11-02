// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <live_trackers/schema_tracker_base.hpp>

#include <chrono>
#include <optional>

TEST_CASE("Tensor sample freshness uses the original source time", "[unit][live_trackers][haptic]")
{
    using namespace std::chrono_literals;

    core::SchemaTrackerBase::SampleResult sample;
    sample.timestamp = core::DeviceDataTimestamp(450'000'000, 100'000'000, 0);

    CHECK(core::SchemaTrackerBase::sample_is_fresh(sample, 299'999'999, 200ms));
    CHECK_FALSE(core::SchemaTrackerBase::sample_is_fresh(sample, 300'000'000, 200ms));
    CHECK(core::SchemaTrackerBase::sample_is_fresh(sample, 500'000'000, std::nullopt));
}

TEST_CASE("Tensor sample freshness falls back to arrival time", "[unit][live_trackers][haptic]")
{
    using namespace std::chrono_literals;

    core::SchemaTrackerBase::SampleResult sample;
    sample.timestamp = core::DeviceDataTimestamp(450'000'000, 0, 0);

    CHECK(core::SchemaTrackerBase::sample_is_fresh(sample, 500'000'000, 200ms));
    CHECK_FALSE(core::SchemaTrackerBase::sample_is_fresh(sample, 650'000'000, 200ms));
}

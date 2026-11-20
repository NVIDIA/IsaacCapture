// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <deviceio_trackers/haptic_command_reader_tracker.hpp>

#include <chrono>
#include <stdexcept>

TEST_CASE("Haptic command readers optionally bound source sample age", "[unit][haptic]")
{
    using namespace std::chrono_literals;

    const core::HapticCommandReaderTracker unbounded("haptic");
    CHECK_FALSE(unbounded.max_sample_age().has_value());

    const core::HapticCommandReaderTracker bounded(
        "haptic", core::HapticCommandReaderTracker::DEFAULT_MAX_PAYLOAD_SIZE, 200ms);
    REQUIRE(bounded.max_sample_age().has_value());
    CHECK(*bounded.max_sample_age() == 200ms);
}

TEST_CASE("Haptic command readers reject non-positive sample age limits", "[unit][haptic]")
{
    using namespace std::chrono_literals;

    CHECK_THROWS_AS(
        core::HapticCommandReaderTracker("haptic", core::HapticCommandReaderTracker::DEFAULT_MAX_PAYLOAD_SIZE, 0ms),
        std::invalid_argument);
    CHECK_THROWS_AS(
        core::HapticCommandReaderTracker("haptic", core::HapticCommandReaderTracker::DEFAULT_MAX_PAYLOAD_SIZE, -1ms),
        std::invalid_argument);
}

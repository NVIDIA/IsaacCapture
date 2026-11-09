// SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "frame_liveness.hpp"

#include <catch2/catch_test_macros.hpp>

#include <initializer_list>
#include <thread>

namespace
{
constexpr int64_t second = 1'000'000'000LL;
constexpr auto left = core::OrbbecEgoCameraStream_ColorLeft;
constexpr auto right = core::OrbbecEgoCameraStream_ColorRight;

void arrive(plugins::ego::VideoLiveness& liveness, int64_t now, bool has_left = true, bool has_right = true)
{
    liveness.observe(now, [=](auto stream) { return stream == left ? has_left : has_right; });
}
} // namespace

TEST_CASE("EGO video liveness follows callbacks while the consumer is stalled", "[ego][liveness]")
{
    plugins::ego::VideoLiveness liveness;
    const int64_t epoch = second;
    for (int64_t now = epoch; now <= 20 * second; now += second)
    {
        arrive(liveness, now);
        CHECK_FALSE(liveness.expired(left, now + second / 2, epoch));
        CHECK_FALSE(liveness.expired(right, now + second / 2, epoch));
    }
    CHECK_FALSE(liveness.expired(left, 25 * second, epoch));
    CHECK(liveness.expired(left, 25 * second + 1, epoch));
    CHECK(liveness.expired(right, 25 * second + 1, epoch));
}

TEST_CASE("EGO video liveness does not let one eye refresh a missing eye", "[ego][liveness]")
{
    plugins::ego::VideoLiveness liveness;
    const int64_t epoch = second;
    arrive(liveness, epoch);
    arrive(liveness, 7 * second, true, false);
    CHECK_FALSE(liveness.expired(left, 7 * second, epoch));
    CHECK(liveness.expired(right, 7 * second, epoch));

    arrive(liveness, 13 * second, false, false);
    CHECK(liveness.expired(left, 13 * second, epoch));
    CHECK(liveness.expired(right, 13 * second, epoch));
}

TEST_CASE("EGO video liveness gives unseen streams the current epoch grace period", "[ego][liveness]")
{
    plugins::ego::VideoLiveness liveness;
    const int64_t epoch = 30 * second;
    for (const auto stream : { left, right })
    {
        CHECK_FALSE(liveness.expired(stream, epoch, epoch));
        CHECK_FALSE(liveness.expired(stream, epoch + 5 * second, epoch));
        CHECK(liveness.expired(stream, epoch + 5 * second + 1, epoch));
    }
}

TEST_CASE("EGO video liveness discards old arrivals after reconnect", "[ego][liveness][reconnect]")
{
    plugins::ego::VideoLiveness liveness;
    arrive(liveness, second);
    liveness.reset();
    const int64_t epoch = 30 * second;
    CHECK_FALSE(liveness.expired(left, epoch, epoch));
    CHECK_FALSE(liveness.expired(right, epoch, epoch));
    arrive(liveness, epoch + 4 * second, false, true);
    CHECK(liveness.expired(left, epoch + 6 * second, epoch));
    CHECK_FALSE(liveness.expired(right, epoch + 6 * second, epoch));
}

TEST_CASE("EGO video liveness preserves the newest concurrently observed arrival", "[ego][liveness]")
{
    plugins::ego::VideoLiveness liveness;
    std::thread earlier([&] { arrive(liveness, second); });
    std::thread later([&] { arrive(liveness, 10 * second); });
    earlier.join();
    later.join();
    arrive(liveness, 2 * second);
    CHECK_FALSE(liveness.expired(left, 14 * second, second));
    CHECK_FALSE(liveness.expired(right, 14 * second, second));
}

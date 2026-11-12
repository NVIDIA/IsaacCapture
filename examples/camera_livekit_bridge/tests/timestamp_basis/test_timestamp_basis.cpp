// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <video_types/timestamp_basis.hpp>

#include <cstdint>
#include <iostream>

namespace
{

int g_failures = 0;

void check(bool condition, const char* what)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << what << "\n";
        ++g_failures;
    }
}

} // namespace

int main()
{
    using camera_bridge::TimestampBasis;
    using camera_bridge::user_timestamp_us;

    check(camera_bridge::parse_timestamp_basis("monotonic_us") == TimestampBasis::MonotonicUs, "parse monotonic_us");
    check(camera_bridge::parse_timestamp_basis("capture_monotonic_us") == TimestampBasis::CaptureMonotonicUs,
          "parse capture_monotonic_us");
    check(camera_bridge::parse_timestamp_basis("capture_unix_us") == TimestampBasis::CaptureUnixUs,
          "parse capture_unix_us");
    check(!camera_bridge::parse_timestamp_basis("processing_unix_us").has_value(), "reject an unsupported basis");
    check(!camera_bridge::parse_timestamp_basis("").has_value(), "reject an empty basis");

    const int64_t realtime = 1'800'000'000'000'000'000;
    const int64_t monotonic = 5'000'000'000'000;

    // Frame-ready time ignores the sensor time and is the monotonic clock in microseconds.
    check(user_timestamp_us(TimestampBasis::MonotonicUs, 0, realtime, monotonic) == 5'000'000'000,
          "monotonic_us without sensor time");
    check(user_timestamp_us(TimestampBasis::MonotonicUs, 123, realtime, monotonic) == 5'000'000'000,
          "monotonic_us ignores sensor time");

    // Capture time 80 ms ago on the sender's wall clock.
    const uint64_t sof = static_cast<uint64_t>(realtime - 80'000'000);
    check(user_timestamp_us(TimestampBasis::CaptureUnixUs, sof, realtime, monotonic) == sof / 1000,
          "capture_unix_us is the sensor time");

    // The adapter produces this basis; user_timestamp_us never does.
    for (const uint64_t sensor : { uint64_t{ 0 }, sof, static_cast<uint64_t>(realtime + 1'000'000) })
        check(user_timestamp_us(TimestampBasis::CaptureMonotonicUs, sensor, realtime, monotonic) == 0,
              "capture_monotonic_us gives 0");

    // No sensor time, a future one and an implausibly old one are not vouched for.
    {
        const auto basis = TimestampBasis::CaptureUnixUs;
        check(user_timestamp_us(basis, 0, realtime, monotonic) == 0, "no sensor time gives 0");
        check(user_timestamp_us(basis, static_cast<uint64_t>(realtime + 1'000'000), realtime, monotonic) == 0,
              "future sensor time gives 0");
        check(user_timestamp_us(basis, static_cast<uint64_t>(realtime - camera_bridge::kMaxCaptureAgeNs - 1), realtime,
                                monotonic) == 0,
              "too old sensor time gives 0");
        check(user_timestamp_us(
                  basis, static_cast<uint64_t>(realtime - camera_bridge::kMaxCaptureAgeNs), realtime, monotonic) != 0,
              "sensor time at the age limit is accepted");
    }

    if (g_failures == 0)
        std::cout << "timestamp_basis: all checks passed\n";
    return g_failures == 0 ? 0 : 1;
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace camera_bridge
{

/// What a published frame's application timestamp means; consumers must be told the same value.
enum class TimestampBasis
{
    /// The publisher's CLOCK_MONOTONIC when the frame is published (frame-ready time).
    MonotonicUs,
    /// The sensor's capture event on the publisher's CLOCK_MONOTONIC. The sensor time arrives as sender wall-clock
    /// time, so this assumes the sender's clock agrees with this host's wall clock.
    CaptureMonotonicUs,
    /// The sensor's capture event as Unix time on the sender's clock.
    CaptureUnixUs,
};

inline std::optional<TimestampBasis> parse_timestamp_basis(const std::string& name)
{
    if (name == "monotonic_us")
        return TimestampBasis::MonotonicUs;
    if (name == "capture_monotonic_us")
        return TimestampBasis::CaptureMonotonicUs;
    if (name == "capture_unix_us")
        return TimestampBasis::CaptureUnixUs;
    return std::nullopt;
}

/// A sensor time older than this, or in the future, is not a capture time this host can vouch for.
inline constexpr int64_t kMaxCaptureAgeNs = 5'000'000'000;

/**
 * @brief The frame's `user_timestamp` in microseconds on `basis`, or 0 when the publisher cannot vouch for one.
 *
 * `sensor_sof_ns` is the sensor start of frame as Unix nanoseconds on the sender's clock (0 when unknown);
 * `realtime_now_ns` and `monotonic_now_ns` are this host's CLOCK_REALTIME and CLOCK_MONOTONIC read together.
 * For the capture bases a missing, future or implausibly old sensor time gives 0, never a guess.
 */
inline uint64_t user_timestamp_us(TimestampBasis basis,
                                  uint64_t sensor_sof_ns,
                                  int64_t realtime_now_ns,
                                  int64_t monotonic_now_ns)
{
    if (basis == TimestampBasis::MonotonicUs)
        return monotonic_now_ns > 0 ? static_cast<uint64_t>(monotonic_now_ns) / 1000 : 0;
    if (sensor_sof_ns == 0)
        return 0;
    const int64_t age_ns = realtime_now_ns - static_cast<int64_t>(sensor_sof_ns);
    if (age_ns < 0 || age_ns > kMaxCaptureAgeNs)
        return 0;
    if (basis == TimestampBasis::CaptureUnixUs)
        return sensor_sof_ns / 1000;
    const int64_t capture_ns = monotonic_now_ns - age_ns;
    return capture_ns > 0 ? static_cast<uint64_t>(capture_ns) / 1000 : 0;
}

} // namespace camera_bridge

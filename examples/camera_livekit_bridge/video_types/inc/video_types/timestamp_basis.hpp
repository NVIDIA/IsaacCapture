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
    /// The sensor's capture event on the publisher's CLOCK_MONOTONIC, converted from the sender's CLOCK_REALTIME with
    /// chrony's synchronisation evidence for both hosts and refused when that evidence cannot vouch for it.
    CaptureMonotonicUs,
};

inline std::optional<TimestampBasis> parse_timestamp_basis(const std::string& name)
{
    if (name == "monotonic_us")
        return TimestampBasis::MonotonicUs;
    if (name == "capture_monotonic_us")
        return TimestampBasis::CaptureMonotonicUs;
    return std::nullopt;
}

/// A sensor time older than this, or in the future, is not a capture time this host can vouch for.
inline constexpr int64_t kMaxCaptureAgeNs = 5'000'000'000;

} // namespace camera_bridge

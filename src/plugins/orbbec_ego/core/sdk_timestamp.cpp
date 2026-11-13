// SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/ego_camera/sdk_timestamp.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace plugins::ego
{

int64_t sdk_timestamp_ns(uint64_t timestamp_us)
{
    // Reject before casting or multiplying; corrupt SDK values must retain
    // their original unsigned value in diagnostics instead of overflowing.
    constexpr auto maximum_us = static_cast<uint64_t>(std::numeric_limits<int64_t>::max() / 1000);
    if (timestamp_us > maximum_us)
        throw std::overflow_error("EGO SDK timestamp_us=" + std::to_string(timestamp_us) +
                                  " exceeds int64 nanosecond range");
    return static_cast<int64_t>(timestamp_us) * 1000;
}

} // namespace plugins::ego

// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/jmex/device_clock.hpp"

#include <cstdint>
#include <limits>

namespace plugins
{
namespace jmex
{

const char* device_clock_name(DeviceClock clock)
{
    return clock == DeviceClock::DEVICE ? "device" : "local";
}

int64_t raw_device_time_ns(DeviceClock clock, int64_t sample_time_ns, uint64_t player_time_ns)
{
    if (clock == DeviceClock::DEVICE && player_time_ns <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    {
        return static_cast<int64_t>(player_time_ns);
    }
    return sample_time_ns;
}

} // namespace jmex
} // namespace plugins

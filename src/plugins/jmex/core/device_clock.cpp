// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/jmex/device_clock.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace plugins
{
namespace jmex
{

namespace
{

DeviceClock parse_device_clock(const std::string& source, const std::string& value)
{
    if (value == "device")
    {
        return DeviceClock::DEVICE;
    }
    if (value == "local")
    {
        return DeviceClock::LOCAL;
    }
    throw std::runtime_error(source + " takes 'device' or 'local', got '" + value + "'");
}

} // namespace

DeviceClock choose_device_clock(const char* flag_value, const char* env_value)
{
    DeviceClock clock = DeviceClock::DEVICE;
    if (env_value != nullptr && *env_value != '\0')
    {
        clock = parse_device_clock("JMEX_DEVICE_CLOCK", env_value);
    }
    if (flag_value != nullptr)
    {
        clock = parse_device_clock("--device-clock=", flag_value);
    }
    return clock;
}

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

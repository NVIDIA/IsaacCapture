// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace plugins
{
namespace jmex
{

//! What goes in a sample's raw device clock.
enum class DeviceClock
{
    //! The time MOXI Player stamped on the frame.
    DEVICE,
    //! The local time the frame was picked up: the same value as the common clock.
    LOCAL,
};

//! The clock ``--device-clock=`` names, else ``JMEX_DEVICE_CLOCK``, else DEVICE. Null means not
//! given, and so does an empty variable. Any value but ``device`` or ``local`` throws
//! std::runtime_error naming where it came from, even one the flag overrides.
DeviceClock choose_device_clock(const char* flag_value, const char* env_value);

//! The clock's name as ``--device-clock=`` spells it, for the startup line.
const char* device_clock_name(DeviceClock clock);

//! What a sample carries as its raw device clock, given the local time it was picked up at and the
//! Player time on its frame.
int64_t raw_device_time_ns(DeviceClock clock, int64_t sample_time_ns, uint64_t player_time_ns);

} // namespace jmex
} // namespace plugins

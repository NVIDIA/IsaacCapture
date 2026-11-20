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
    //! The frame's Player time, FullBodyFrame::player_time_ns.
    DEVICE,
    //! The local time the frame was picked up: the same value as the common clock.
    LOCAL,
};

//! The clock's name as ``--device-clock=`` spells it, for the startup line.
const char* device_clock_name(DeviceClock clock);

//! What a sample carries as its raw device clock, given the local time it was picked up at and the
//! Player time on its frame. A Player time past int64 gives the sample time.
int64_t raw_device_time_ns(DeviceClock clock, int64_t sample_time_ns, uint64_t player_time_ns);

} // namespace jmex
} // namespace plugins

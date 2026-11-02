// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/revohuman_plugin/device_clock.hpp"

namespace plugins
{
namespace revohuman
{

std::int64_t TickUnwrapper::unwrap(std::uint32_t raw)
{
    if (!anchored_)
    {
        anchored_ = true;
        last_ = raw;
        extended_ = static_cast<std::int64_t>(raw);
        return extended_;
    }
    const auto delta = static_cast<std::int32_t>(raw - last_);
    last_ = raw;
    extended_ += delta;
    return extended_;
}

void TickUnwrapper::reset()
{
    anchored_ = false;
    last_ = 0;
    extended_ = 0;
}

std::int64_t ms_ticks_to_ns(std::int64_t ticks)
{
    return ticks * 1'000'000;
}

} // namespace revohuman
} // namespace plugins

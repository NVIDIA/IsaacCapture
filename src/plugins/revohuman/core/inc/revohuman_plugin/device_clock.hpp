// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace plugins
{
namespace revohuman
{

//! Unwraps a free-running uint32 device counter into a monotonic int64 by signed 32-bit deltas.
//! The first value anchors the output at the raw value itself, so the unwrapped count equals the
//! device count until the first wrap. A step backwards is kept as a step backwards.
class TickUnwrapper
{
public:
    std::int64_t unwrap(std::uint32_t raw);
    void reset();
    bool anchored() const
    {
        return anchored_;
    }

private:
    bool anchored_ = false;
    std::uint32_t last_ = 0;
    std::int64_t extended_ = 0;
};

//! Glove and tactile ticks are 1 ms.
std::int64_t ms_ticks_to_ns(std::int64_t ticks);

} // namespace revohuman
} // namespace plugins

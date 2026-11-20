// SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <schema/orbbec_ego_camera_generated.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace plugins::ego
{

class VideoLiveness
{
public:
    template <typename HasFrame>
    void observe(int64_t arrival_ns, HasFrame&& has_frame)
    {
        for (const auto stream : core::EnumValuesOrbbecEgoCameraStream())
        {
            if (!has_frame(stream))
                continue;
            auto& arrival = arrivals_[static_cast<size_t>(stream)];
            auto previous = arrival.load(std::memory_order_relaxed);
            // Concurrent callbacks can finish out of arrival order.
            while (previous < arrival_ns &&
                   !arrival.compare_exchange_weak(previous, arrival_ns, std::memory_order_relaxed))
            {
            }
        }
    }

    bool expired(core::OrbbecEgoCameraStream stream, int64_t now_ns, int64_t epoch_started_ns) const
    {
        const auto arrival = arrivals_.at(static_cast<size_t>(stream)).load(std::memory_order_relaxed);
        return now_ns - (arrival == 0 ? epoch_started_ns : arrival) > 5'000'000'000LL;
    }

    // Reset only after the old pipeline has stopped its callbacks.
    void reset()
    {
        for (auto& arrival : arrivals_)
            arrival.store(0, std::memory_order_relaxed);
    }

private:
    std::array<std::atomic<int64_t>, static_cast<size_t>(core::OrbbecEgoCameraStream_MAX) + 1> arrivals_{};
};

} // namespace plugins::ego

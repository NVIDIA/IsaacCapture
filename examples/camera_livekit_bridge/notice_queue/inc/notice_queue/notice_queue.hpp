// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ipc_client/cuda_ipc_client.hpp>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <utility>

namespace camera_bridge
{

// A notice with its position in the stream of notices read so far, so how many came after it is known even once
// others have been dropped.
struct QueuedNotice
{
    FrameNotice notice;
    uint64_t ordinal;
};

// Adds a notice just read to `queue`, oldest first, and drops what it makes unusable: an entry whose slot it reuses,
// and entries now as many notices old as the publisher's ring is deep (their slots are about to be, or already are,
// overwritten). The queue therefore never holds more than the ring's depth, however long the burst. Returns how many
// entries were dropped.
inline std::size_t enqueue_notice(std::deque<QueuedNotice>& queue,
                                  uint64_t& total_read,
                                  std::size_t ring_slots,
                                  FrameNotice&& notice)
{
    std::size_t dropped = 0;
    const uint64_t ordinal = ++total_read;
    for (auto it = queue.begin(); it != queue.end();)
    {
        if (it->notice.slot == notice.slot && it->notice.slot_generation < notice.slot_generation)
        {
            it = queue.erase(it);
            ++dropped;
        }
        else
            ++it;
    }
    queue.push_back({ std::move(notice), ordinal });
    while (ring_slots > 1 && !queue.empty() && total_read - queue.front().ordinal + 1 >= ring_slots)
    {
        queue.pop_front();
        ++dropped;
    }
    return dropped;
}

} // namespace camera_bridge

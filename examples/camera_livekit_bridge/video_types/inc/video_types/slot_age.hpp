// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace camera_bridge
{

/// How long after its notice a slot is still safe to read: the time the publisher needs to write the other slots of
/// its ring and come back to this one, at the stream's nominal rate. Zero when the rate or ring is unusable.
inline int64_t slot_read_deadline_ns(uint32_t fps, std::size_t slot_count)
{
    if (fps == 0 || slot_count < 2)
        return 0;
    return static_cast<int64_t>(slot_count - 1) * 1'000'000'000 / static_cast<int64_t>(fps);
}

enum class SlotAge
{
    Fresh, ///< the read finished within the deadline
    Stale, ///< the read finished after the deadline: the slot may have been overwritten during it
    Unknown, ///< the notice has no usable time (missing, or later than now): nothing can be said
};

/// Classify a read that finished at `now_ns` (CLOCK_MONOTONIC) of the slot announced at `notice_ns` (the publisher's
/// CLOCK_MONOTONIC on the same host). A deadline of zero disables the check.
inline SlotAge classify_slot_age(uint64_t notice_ns, int64_t now_ns, int64_t deadline_ns)
{
    if (deadline_ns <= 0 || notice_ns == 0 || static_cast<int64_t>(notice_ns) > now_ns)
        return SlotAge::Unknown;
    return now_ns - static_cast<int64_t>(notice_ns) > deadline_ns ? SlotAge::Stale : SlotAge::Fresh;
}

} // namespace camera_bridge

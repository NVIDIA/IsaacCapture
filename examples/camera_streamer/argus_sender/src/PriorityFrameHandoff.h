/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_PRIORITY_FRAME_HANDOFF_H
#define ARGUS_SENDER_PRIORITY_FRAME_HANDOFF_H

#include <cstdint>
#include <memory>
#include <utility>

namespace argus_sender
{

constexpr uint64_t kPriorityStartupDroughtNs = 10000000000ULL;
constexpr uint64_t kPrioritySteadyDroughtNs = 1000000000ULL;

inline bool priorityFrameDroughtExpired(uint64_t startNs, uint64_t nowNs, bool hasAcquiredFrame)
{
    const uint64_t limitNs = hasAcquiredFrame ? kPrioritySteadyDroughtNs : kPriorityStartupDroughtNs;
    return nowNs >= startNs && nowNs - startNs >= limitNs;
}

// Track progress by new frame identity, not by successful acquisition calls,
// so repeated frame numbers cannot hide a camera drought.
class PriorityFrameProgress
{
public:
    explicit PriorityFrameProgress(uint64_t startNs) : m_lastFreshFrameNs(startNs)
    {
    }

    void noteFreshFrame(uint64_t nowNs)
    {
        m_lastFreshFrameNs = nowNs;
        m_hasFreshFrame = true;
    }

    bool droughtExpired(uint64_t nowNs) const
    {
        return priorityFrameDroughtExpired(m_lastFreshFrameNs, nowNs, m_hasFreshFrame);
    }

    uint64_t droughtLimitNs() const
    {
        return m_hasFreshFrame ? kPrioritySteadyDroughtNs : kPriorityStartupDroughtNs;
    }

private:
    uint64_t m_lastFreshFrameNs;
    bool m_hasFreshFrame = false;
};

// The caller holds the domain mutex. Returning the superseded frame lets the
// caller release its Argus buffer after unlocking.
template <typename Frame>
class LatestFrameSlot
{
public:
    bool occupied() const
    {
        return static_cast<bool>(m_frame);
    }

    std::unique_ptr<Frame> replace(std::unique_ptr<Frame> frame)
    {
        std::unique_ptr<Frame> previous = std::move(m_frame);
        m_frame = std::move(frame);
        return previous;
    }

    std::unique_ptr<Frame> take()
    {
        return std::move(m_frame);
    }
    void reset()
    {
        m_frame.reset();
    }

private:
    std::unique_ptr<Frame> m_frame;
};

} // namespace argus_sender

#endif // ARGUS_SENDER_PRIORITY_FRAME_HANDOFF_H

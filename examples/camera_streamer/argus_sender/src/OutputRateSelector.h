/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_OUTPUT_RATE_SELECTOR_H
#define ARGUS_SENDER_OUTPUT_RATE_SELECTOR_H

#include <cstdint>

namespace argus_sender
{

// Holds a stream to its configured rate when the capture backend delivers faster. Sent frames are held to a grid one
// period apart: a frame is skipped while it is more than a quarter period earlier than the next grid time, so a 60 Hz
// source gives a 30 Hz stream, a 30 Hz source with a few milliseconds of jitter loses nothing, and a 30 Hz source at
// 24 fps keeps 24 frames a second. A gap longer than a period restarts the grid at the frame that ended it. A rate of
// 0, or a clock that steps back, never skips.
class OutputRateSelector
{
public:
    explicit OutputRateSelector(uint32_t fps) : m_periodNs(fps ? 1000000000ULL / fps : 0)
    {
    }

    bool shouldSkip(uint64_t frameNs) const
    {
        return m_periodNs != 0 && m_started && frameNs >= m_lastSentNs && frameNs + m_periodNs / 4 < m_nextDueNs;
    }

    void sent(uint64_t frameNs)
    {
        if (!m_started || frameNs < m_lastSentNs || frameNs > m_nextDueNs + m_periodNs)
            m_nextDueNs = frameNs + m_periodNs;
        else
            m_nextDueNs += m_periodNs;
        m_lastSentNs = frameNs;
        m_started = true;
    }

private:
    uint64_t m_periodNs;
    uint64_t m_nextDueNs = 0;
    uint64_t m_lastSentNs = 0;
    bool m_started = false;
};

} // namespace argus_sender

#endif

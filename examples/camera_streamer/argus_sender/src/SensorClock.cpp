/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "SensorClock.h"

#include <cstdlib>
#include <ctime>
#include <limits>
#include <utility>

namespace argus_sender
{
namespace
{

#if defined(__aarch64__)
int64_t monotonicNow()
{
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

int64_t readCounter()
{
    uint64_t value;
    __asm__ __volatile__("isb; mrs %0, cntvct_el0" : "=r"(value)::"memory");
    return static_cast<int64_t>(value);
}
#endif

} // namespace

bool readCounterPair(ClockPair* pair)
{
#if defined(__aarch64__)
    // The narrowest of several back-to-back reads: a descheduled read has a wide bracket and is discarded.
    const int kAttempts = 8;
    ClockPair best;
    best.bracketNs = std::numeric_limits<int64_t>::max();
    for (int i = 0; i < kAttempts; ++i)
    {
        const int64_t before = monotonicNow();
        const int64_t source = readCounter();
        const int64_t after = monotonicNow();
        if (after - before < best.bracketNs)
        {
            best.source = source;
            best.monotonicNs = before + (after - before) / 2;
            best.bracketNs = after - before;
        }
    }
    *pair = best;
    return true;
#else
    (void)pair;
    return false;
#endif
}

int64_t readCounterHz()
{
#if defined(__aarch64__)
    uint64_t hz;
    __asm__ __volatile__("mrs %0, cntfrq_el0" : "=r"(hz));
    return static_cast<int64_t>(hz);
#else
    return 0;
#endif
}

int64_t counterTicksToNs(int64_t ticks, int64_t hz)
{
    // Whole seconds and a remainder, so the product cannot overflow 64 bits.
    const int64_t seconds = ticks / hz;
    const int64_t remainder = ticks % hz;
    return seconds * 1000000000LL + remainder * 1000000000LL / hz;
}

SensorClockMap::SensorClockMap(const Config& config, PairReader reader)
    : m_config(config), m_reader(std::move(reader)), m_anchorTakenNs(0), m_haveAnchor(false), m_stepped(false)
{
}

void SensorClockMap::reset()
{
    m_haveAnchor = false;
    m_stepped = false;
}

bool SensorClockMap::refresh(int64_t nowMonotonicNs, std::string* reason)
{
    ClockPair next;
    if (!m_reader || !m_reader(&next))
    {
        *reason = "the sensor counter cannot be read on this platform";
        return false;
    }
    if (m_haveAnchor)
    {
        // The two anchors must agree on how far the counter and CLOCK_MONOTONIC moved between them.
        const int64_t counterNs = counterTicksToNs(next.source - m_anchor.source, m_config.counterHz);
        const int64_t monotonicDeltaNs = next.monotonicNs - m_anchor.monotonicNs;
        const int64_t disagreement = std::llabs(counterNs - monotonicDeltaNs);
        if (disagreement > m_config.stepThresholdNs + m_anchor.bracketNs + next.bracketNs)
            m_stepped = true;
        else
            m_stepped = false;
    }
    m_anchor = next;
    m_anchorTakenNs = nowMonotonicNs;
    m_haveAnchor = true;
    return true;
}

SensorTimeMapping SensorClockMap::map(uint64_t sensorTicks, int64_t readyMonotonicNs, int64_t nowMonotonicNs)
{
    SensorTimeMapping result;
    if (m_config.counterHz <= 0)
    {
        result.reason = "the sensor counter frequency is not known";
        return result;
    }
    if (!m_haveAnchor || nowMonotonicNs - m_anchorTakenNs >= m_config.refreshNs)
    {
        if (!refresh(nowMonotonicNs, &result.reason))
            return result;
    }
    result.anchorAgeNs = nowMonotonicNs - m_anchorTakenNs;
    if (m_stepped)
    {
        result.reason = "the sensor counter and CLOCK_MONOTONIC disagree between anchors: a clock step";
        return result;
    }
    // The mapping extrapolates from the anchor to the sensor event, in either direction, so the slew error accrues over
    // that distance and not over how old the anchor is by the time the frame is handled.
    const int64_t anchorToEventNs =
        counterTicksToNs(static_cast<int64_t>(sensorTicks) - m_anchor.source, m_config.counterHz);
    const int64_t capture = m_anchor.monotonicNs + anchorToEventNs;
    result.uncertaintyNs = m_anchor.bracketNs + std::llabs(anchorToEventNs) * m_config.slewPpm / 1000000;
    if (capture > readyMonotonicNs)
    {
        result.reason = "the sensor timestamp is after the frame was ready: the assumed counter or rate is wrong";
        return result;
    }
    if (readyMonotonicNs - capture > m_config.maxAgeNs)
    {
        result.reason =
            "the sensor timestamp is implausibly long before the frame was ready: the assumed counter or "
            "rate is wrong";
        return result;
    }
    result.valid = true;
    result.captureMonotonicNs = capture;
    return result;
}

} // namespace argus_sender

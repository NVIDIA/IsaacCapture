/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "SensorClock.h"

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <utility>

namespace argus_sender
{
namespace
{

using Int128 = __int128;

constexpr int64_t kInt64Max = std::numeric_limits<int64_t>::max();
constexpr int64_t kInt64Min = std::numeric_limits<int64_t>::min();

bool fitsInt64(Int128 value)
{
    return value >= kInt64Min && value <= kInt64Max;
}

// |value|, false for INT64_MIN, whose magnitude does not fit (llabs is undefined there).
bool absChecked(int64_t value, int64_t* out)
{
    if (value == kInt64Min)
        return false;
    *out = value < 0 ? -value : value;
    return true;
}

// `magnitude * ppm / 1e6`, widened so extreme inputs cannot wrap.
Int128 scaledByPpm(int64_t magnitude, int64_t ppm)
{
    return static_cast<Int128>(magnitude) * ppm / 1000000;
}

#if defined(__aarch64__)
// False when the clock cannot be read; the value is then meaningless.
bool monotonicNow(int64_t* ns)
{
    timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return false;
    *ns = static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
    return true;
}

// False when the clock cannot be read; the value is then meaningless.
bool realtimeNow(int64_t* ns)
{
    timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) != 0)
        return false;
    *ns = static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
    return true;
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
        int64_t before = 0;
        int64_t after = 0;
        int64_t realtime = 0;
        if (!monotonicNow(&before))
            continue;
        const int64_t source = readCounter();
        const bool realtimeOk = realtimeNow(&realtime);
        // A failed or backwards read cannot bracket the counter: this attempt is discarded.
        if (!monotonicNow(&after) || after < before)
            continue;
        if (after - before < best.bracketNs)
        {
            best.source = source;
            best.monotonicNs = before + (after - before) / 2;
            best.bracketNs = after - before;
            best.realtimeNs = realtime;
            // A failed read leaves the pair without a realtime reading, so no realtime capture time is claimed.
            best.hasRealtime = realtimeOk;
        }
    }
    if (best.bracketNs == std::numeric_limits<int64_t>::max())
        return false; // no attempt produced a usable bracket
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

bool counterTicksToNsChecked(int64_t ticks, int64_t hz, int64_t* ns)
{
    if (hz <= 0)
        return false;
    const Int128 value = static_cast<Int128>(ticks) * 1000000000LL / hz;
    if (!fitsInt64(value))
        return false;
    *ns = static_cast<int64_t>(value);
    return true;
}

int64_t counterTicksToNs(int64_t ticks, int64_t hz)
{
    int64_t ns = 0;
    if (counterTicksToNsChecked(ticks, hz, &ns))
        return ns;
    return ticks < 0 ? kInt64Min : kInt64Max;
}

int64_t monotonicNowNs()
{
    timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return 0;
    return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

SensorClockMap::SensorClockMap(const Config& config, PairReader reader)
    : m_config(config), m_reader(std::move(reader)), m_anchorTakenNs(0), m_haveAnchor(false), m_stepped(false)
{
}

void SensorClockMap::reset()
{
    m_haveAnchor = false;
    m_stepped = false;
    m_haveCutoff = false;
    m_cutoffSource = 0;
    m_counterWentBackwards = false;
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
        // The two anchors must agree on how far the counter and CLOCK_MONOTONIC moved between them. A value that
        // cannot be represented is not evidence of agreement: it counts as a step.
        int64_t sourceDelta = 0;
        int64_t counterNs = 0;
        int64_t counterSpacingNs = 0;
        int64_t monotonicDeltaNs = 0;
        const bool deltaOk = !__builtin_sub_overflow(next.source, m_anchor.source, &sourceDelta);
        const bool representable = deltaOk && counterTicksToNsChecked(sourceDelta, m_config.counterHz, &counterNs) &&
                                   absChecked(counterNs, &counterSpacingNs) &&
                                   !__builtin_sub_overflow(next.monotonicNs, m_anchor.monotonicNs, &monotonicDeltaNs);
        // Event order across a backwards counter cannot be trusted at all: refuse until reset().
        if (deltaOk && sourceDelta < 0)
            m_counterWentBackwards = true;
        const Int128 disagreement =
            representable ? (counterNs > monotonicDeltaNs ? static_cast<Int128>(counterNs) - monotonicDeltaNs :
                                                            static_cast<Int128>(monotonicDeltaNs) - counterNs) :
                            0;
        // A slew within slewPpm moves the two clocks apart by up to that fraction of the counter's own spacing, which
        // is not a step.
        const Int128 bounds = static_cast<Int128>(m_config.stepThresholdNs) + m_anchor.bracketNs + next.bracketNs;
        const Int128 allowed = bounds + scaledByPpm(counterSpacingNs, m_config.slewPpm);
        // CLOCK_REALTIME and CLOCK_MONOTONIC share frequency adjustment, so their difference moves only on a wall-clock
        // step. No slew term applies, or a step smaller than the slew allowance over long anchor spacing would pass.
        bool realtimeStep = false;
        if (m_anchor.hasRealtime && next.hasRealtime)
        {
            const Int128 previousOffset = static_cast<Int128>(m_anchor.realtimeNs) - m_anchor.monotonicNs;
            const Int128 nextOffset = static_cast<Int128>(next.realtimeNs) - next.monotonicNs;
            const Int128 moved = nextOffset > previousOffset ? nextOffset - previousOffset : previousOffset - nextOffset;
            realtimeStep = moved > bounds;
        }
        m_stepped = !representable || disagreement > allowed || realtimeStep;
        if (m_stepped)
        {
            // The step happened somewhere between the previous anchor and this one, so an event up to this anchor may
            // lie on either side of it. The counter is sampled before the realtime read that exposes a wall-clock step,
            // both inside the bracket, so the step can follow the sampled value by up to the bracket (grown by the
            // slew allowed over it). Events up to that upper edge stay invalid even once later anchors agree again.
            const Int128 bracketNs = std::max<int64_t>(next.bracketNs, 0);
            const Int128 widenedNs =
                bracketNs + scaledByPpm(next.bracketNs > 0 ? next.bracketNs : 0, m_config.slewPpm) + 1;
            const Int128 bracketTicks = (widenedNs * m_config.counterHz + 999999999) / 1000000000;
            const Int128 edge = std::min<Int128>(static_cast<Int128>(next.source) + bracketTicks, kInt64Max);
            const int64_t edgeSource = static_cast<int64_t>(edge);
            m_cutoffSource = m_haveCutoff ? std::max(m_cutoffSource, edgeSource) : edgeSource;
            m_haveCutoff = true;
        }
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
    // Widened: caller-supplied times can be arbitrarily far apart.
    const Int128 sinceAnchorNs = static_cast<Int128>(nowMonotonicNs) - m_anchorTakenNs;
    if (!m_haveAnchor || sinceAnchorNs >= m_config.refreshNs)
    {
        if (!refresh(nowMonotonicNs, &result.reason))
            return result;
    }
    {
        // The refresh above may have moved the anchor, so measure the age again, saturating.
        const Int128 age = static_cast<Int128>(nowMonotonicNs) - m_anchorTakenNs;
        result.anchorAgeNs = static_cast<int64_t>(std::max<Int128>(kInt64Min, std::min<Int128>(age, kInt64Max)));
    }
    if (m_stepped)
    {
        result.reason = "the sensor counter and CLOCK_MONOTONIC disagree between anchors: a clock step";
        return result;
    }
    if (m_counterWentBackwards)
    {
        result.reason = "the sensor counter went backwards between anchors: event order is lost until reset";
        return result;
    }
    if (sensorTicks > static_cast<uint64_t>(kInt64Max))
    {
        result.reason = "the sensor timestamp is out of range";
        return result;
    }
    const int64_t eventSource = static_cast<int64_t>(sensorTicks);
    if (m_haveCutoff && eventSource <= m_cutoffSource)
    {
        result.reason = "the sensor event is not after a detected clock step: its capture time is ambiguous";
        return result;
    }
    // The mapping extrapolates from the anchor to the sensor event, in either direction, so the slew error accrues over
    // that distance and not over how old the anchor is by the time the frame is handled.
    int64_t sourceDelta = 0;
    int64_t anchorToEventNs = 0;
    int64_t distanceNs = 0;
    int64_t capture = 0;
    if (__builtin_sub_overflow(eventSource, m_anchor.source, &sourceDelta) ||
        !counterTicksToNsChecked(sourceDelta, m_config.counterHz, &anchorToEventNs) ||
        !absChecked(anchorToEventNs, &distanceNs) ||
        __builtin_add_overflow(m_anchor.monotonicNs, anchorToEventNs, &capture))
    {
        result.reason = "the sensor event is too far from the anchor to be placed";
        return result;
    }
    const Int128 uncertainty = static_cast<Int128>(m_anchor.bracketNs) + scaledByPpm(distanceNs, m_config.slewPpm);
    // A refused mapping still reports the bound it computed over the distance.
    result.uncertaintyNs = static_cast<int64_t>(std::min<Int128>(uncertainty, kInt64Max));
    if (uncertainty > m_config.maxUncertaintyNs)
    {
        result.reason =
            "the sensor event is too far from the anchor for the mapping's uncertainty to stay within the limit";
        return result;
    }
    if (capture > readyMonotonicNs)
    {
        result.reason = "the sensor timestamp is after the frame was ready: the assumed counter or rate is wrong";
        return result;
    }
    if (static_cast<Int128>(readyMonotonicNs) - capture > m_config.maxAgeNs)
    {
        result.reason =
            "the sensor timestamp is implausibly long before the frame was ready: the assumed counter or "
            "rate is wrong";
        return result;
    }
    result.valid = true;
    result.captureMonotonicNs = capture;
    int64_t realtimeOffset = 0;
    int64_t captureRealtime = 0;
    if (m_anchor.hasRealtime && !__builtin_sub_overflow(m_anchor.realtimeNs, m_anchor.monotonicNs, &realtimeOffset) &&
        !__builtin_add_overflow(capture, realtimeOffset, &captureRealtime))
    {
        result.hasRealtime = true;
        result.captureRealtimeNs = captureRealtime;
    }
    return result;
}

namespace
{

SensorClockMap::Config nanosecondConfig(int64_t counterHz, SensorClockMap::Config config)
{
    // Everything inside the map is in nanoseconds of the platform counter, so one source unit is one nanosecond. A
    // counter of unknown frequency leaves it at 0, which the map refuses.
    config.counterHz = counterHz > 0 ? 1000000000LL : 0;
    return config;
}

SensorClockMap::PairReader nanosecondReader(int64_t counterHz, SensorClockMap::PairReader reader)
{
    return [counterHz, reader](ClockPair* pair)
    {
        if (counterHz <= 0 || !reader || !reader(pair))
            return false;
        // Only the counter value changes units; the monotonic, realtime and bracket fields are already in ns. A value
        // that does not fit is an unreadable anchor, never a wrapped one.
        int64_t ns = 0;
        if (!counterTicksToNsChecked(pair->source, counterHz, &ns))
            return false;
        pair->source = ns;
        return true;
    };
}

} // namespace

SensorTimeMapper::SensorTimeMapper(int64_t counterHz,
                                   const SensorClockMap::Config& config,
                                   SensorClockMap::PairReader reader)
    : m_counterHz(counterHz), m_map(nanosecondConfig(counterHz, config), nanosecondReader(counterHz, std::move(reader)))
{
}

void SensorTimeMapper::reset()
{
    m_map.reset();
}

SensorTimeMapping SensorTimeMapper::map(uint64_t timestamp,
                                        SensorTimestampUnit unit,
                                        int64_t readyMonotonicNs,
                                        int64_t nowMonotonicNs)
{
    if (m_counterHz <= 0)
        return m_map.map(0, readyMonotonicNs, nowMonotonicNs); // refused: the frequency is not known
    if (timestamp > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    {
        SensorTimeMapping refused;
        refused.reason = "the sensor timestamp is out of range";
        return refused;
    }
    int64_t ns = static_cast<int64_t>(timestamp);
    if (unit == SensorTimestampUnit::CounterTicks && !counterTicksToNsChecked(ns, m_counterHz, &ns))
    {
        SensorTimeMapping refused;
        refused.reason = "the sensor timestamp is out of range";
        return refused;
    }
    SensorTimeMapping result = m_map.map(static_cast<uint64_t>(ns), readyMonotonicNs, nowMonotonicNs);
    if (result.valid)
    {
        // One counter tick is the resolution of the timestamp and of the anchor read; add it, saturating.
        const int64_t tickNs = (1000000000LL + m_counterHz - 1) / m_counterHz;
        result.uncertaintyNs = result.uncertaintyNs > kInt64Max - tickNs ? kInt64Max : result.uncertaintyNs + tickNs;
    }
    return result;
}

void applyMappingToTiming(const SensorTimeMapping& mapping, FrameTiming* timing)
{
    timing->hasCaptureMonotonic = mapping.valid;
    timing->captureMonotonicNs = mapping.valid ? mapping.captureMonotonicNs : 0;
    timing->captureUncertaintyNs = mapping.valid ? mapping.uncertaintyNs : 0;
    timing->captureAnchorAgeNs = mapping.valid ? mapping.anchorAgeNs : 0;
    timing->captureRealtimeNs = mapping.valid && mapping.hasRealtime ? mapping.captureRealtimeNs : 0;
}

} // namespace argus_sender

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_SENSOR_CLOCK_H
#define ARGUS_SENDER_SENSOR_CLOCK_H

// Places a sensor timestamp that is a count of the platform counter's ticks on this machine's CLOCK_MONOTONIC. The
// mapping is only as good as the assumption that the sensor counts that counter, so every conversion is checked and
// an implausible one is reported as invalid, never replaced by a host-side stamp.

#include <cstdint>
#include <functional>
#include <string>

namespace argus_sender
{

// A source clock reading paired with CLOCK_MONOTONIC, read back to back.
struct ClockPair
{
    // The source clock's value (counter ticks) at the midpoint of the bracket.
    int64_t source = 0;
    // CLOCK_MONOTONIC ns at the same midpoint.
    int64_t monotonicNs = 0;
    // Width of the bracket the pair was read in; the pair is no more certain than this.
    int64_t bracketNs = 0;
};

// Reads the ARM generic counter and CLOCK_MONOTONIC back to back, keeping the narrowest of a few attempts. Returns
// false where the counter is not readable (not aarch64).
bool readCounterPair(ClockPair* pair);
// The generic counter's frequency in Hz; 0 where it is not readable.
int64_t readCounterHz();

// Counter ticks to nanoseconds without overflowing 64 bits; `hz` must be positive.
int64_t counterTicksToNs(int64_t ticks, int64_t hz);

// The outcome of placing one sensor timestamp.
struct SensorTimeMapping
{
    bool valid = false;
    // Why it is not valid; empty when it is.
    std::string reason;
    // CLOCK_MONOTONIC ns of the sensor's capture event; meaningful only when valid.
    int64_t captureMonotonicNs = 0;
    // The anchor pair's bracket plus the slew bound over the distance from the anchor to the sensor event: the mapping
    // is no more certain than this.
    int64_t uncertaintyNs = 0;
    // How long ago the anchor was read, at the time of this conversion: its freshness, separate from the uncertainty.
    int64_t anchorAgeNs = 0;
};

// One anchor pair is read through a supported clock API and refreshed at a low rate, never per frame; a frame's capture
// time is the anchor plus the counter difference. A capture time after the frame was ready, or more than maxAgeNs
// before it, means the assumed counter or rate is wrong. A refreshed anchor that disagrees with the old one by more
// than stepThresholdNs means a clock step: the mapping is invalid until reset() or the next consistent refresh.
//
// Not thread-safe; one instance per capture thread.
class SensorClockMap
{
public:
    using PairReader = std::function<bool(ClockPair*)>;

    struct Config
    {
        // Counter frequency in Hz.
        int64_t counterHz = 0;
        int64_t refreshNs = 1000000000;
        // Largest plausible gap between capture and the frame being ready.
        int64_t maxAgeNs = 2000000000;
        // Largest agreement error between consecutive anchors, on top of their brackets.
        int64_t stepThresholdNs = 2000000;
        // CLOCK_MONOTONIC is slewed by NTP against the counter, which Linux bounds at 500 ppm: the mapping's
        // uncertainty grows by this much per second of distance between the anchor and the sensor event.
        int64_t slewPpm = 500;
    };

    SensorClockMap(const Config& config, PairReader reader);

    SensorTimeMapping map(uint64_t sensorTicks, int64_t readyMonotonicNs, int64_t nowMonotonicNs);

    // Forget the anchor: call when capture restarts or the sensor domain changes.
    void reset();

private:
    bool refresh(int64_t nowMonotonicNs, std::string* reason);

    Config m_config;
    PairReader m_reader;
    ClockPair m_anchor;
    int64_t m_anchorTakenNs = 0;
    bool m_haveAnchor = false;
    bool m_stepped = false;
};

} // namespace argus_sender

#endif // ARGUS_SENDER_SENSOR_CLOCK_H

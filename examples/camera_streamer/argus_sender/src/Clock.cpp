// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "Clock.h"

#include <time.h>

namespace argus_sender
{
uint64_t readSenderClockNs(uint32_t* clockDomain)
{
    struct timespec ts;
    // Use the chrony-disciplined Unix wall clock for cross-host latency.
    // CLOCK_TAI offset is not guaranteed to be initialized consistently across
    // the Thor sender and x86 receiver hosts.
    if (clock_gettime(CLOCK_REALTIME, &ts) == 0)
    {
        if (clockDomain)
            *clockDomain = SENDER_CLOCK_REALTIME;
        return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
    }

    if (clockDomain)
        *clockDomain = SENDER_CLOCK_UNKNOWN;
    return 0;
}

uint64_t readTscNowNs()
{
#if defined(__aarch64__)
    // The ARM generic timer is the Tegra TSC domain that stamps VI
    // start-of-frame (Argus getSensorSofTimestampTsc). Scale by the reported
    // frequency (31.25 MHz on Orin) rather than a constant.
    uint64_t ticks = 0;
    uint64_t freq = 0;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(ticks));
    __asm__ __volatile__("mrs %0, cntfrq_el0" : "=r"(freq));
    if (freq == 0)
        return 0;
    const uint64_t kNsPerSecond = 1000000000ULL;
    return (ticks / freq) * kNsPerSecond + ((ticks % freq) * kNsPerSecond) / freq;
#else
    return 0;
#endif
}

}

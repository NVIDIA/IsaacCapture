/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_STEREO_TIMING_H
#define ARGUS_SENDER_STEREO_TIMING_H

#include "FrameTiming.h"

#include <Argus/Argus.h>
#include <EGLStream/EGLStream.h>

#include <cstdint>

namespace argus_sender
{

class StereoOutputRateLimiter
{
public:
    explicit StereoOutputRateLimiter(uint32_t targetFps);
    bool shouldOutput(const FrameTiming& left);

private:
    enum TimestampSource
    {
        TIMESTAMP_NONE,
        TIMESTAMP_TSC,
        TIMESTAMP_SENSOR,
        TIMESTAMP_FRAME,
        TIMESTAMP_STEADY
    };

    uint32_t m_targetFps;
    uint64_t m_intervalNs;
    uint64_t m_toleranceNs;
    TimestampSource m_source;
    uint64_t m_lastTimestampNs;
    uint64_t m_nextDeadlineNs;
    bool m_initialized;
};

bool readFrameTiming(EGLStream::Frame* frame, const char* label, FrameTiming* timing);
bool readFrameTiming(Argus::Buffer* buffer, const char* label, FrameTiming* timing);

} // namespace argus_sender

#endif // ARGUS_SENDER_STEREO_TIMING_H

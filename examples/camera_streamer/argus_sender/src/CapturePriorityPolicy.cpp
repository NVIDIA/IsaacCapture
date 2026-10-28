/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "CapturePriorityPolicy.h"

#include <algorithm>

namespace argus_sender
{
namespace
{

uint64_t percentage(uint64_t value, uint32_t percent)
{
    return (value / 100U) * percent + ((value % 100U) * percent) / 100U;
}

} // namespace

size_t selectPriorityStream(const AppConfig& config, const std::vector<uint32_t>& streamFps)
{
    if (streamFps.empty())
        return 0;
    size_t priority = 0;
    for (size_t i = 1; i < streamFps.size(); ++i)
    {
        if (streamFps[i] > streamFps[priority])
            priority = i;
    }
    const uint32_t baseFps = streamFps[priority];
    const size_t count = std::min(config.streams.size(), streamFps.size());
    for (size_t i = 0; i < count; ++i)
    {
        if (streamFps[i] == baseFps && config.streams[i].type == STREAM_TYPE_STEREO &&
            config.streams[i].output.codec == CODEC_HEVC)
        {
            return i;
        }
    }
    return priority;
}

uint64_t secondaryCaptureReleaseDelayNs(uint64_t framePeriodNs)
{
    return percentage(framePeriodNs, 42U);
}

uint64_t prioritySyncToleranceNs(uint64_t framePeriodNs)
{
    return framePeriodNs / 2U;
}

} // namespace argus_sender

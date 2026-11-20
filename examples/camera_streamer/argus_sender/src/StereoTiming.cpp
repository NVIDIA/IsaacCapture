/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "StereoTiming.h"

#include <Argus/Argus.h>
#include <Argus/Ext/SensorTimestampTsc.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <time.h>

namespace argus_sender
{

StereoOutputRateLimiter::StereoOutputRateLimiter(uint32_t targetFps)
    : m_targetFps(targetFps),
      m_intervalNs(targetFps ? (1000000000ULL / targetFps > 0 ? 1000000000ULL / targetFps : 1) : 0),
      m_toleranceNs(m_intervalNs > 0 ? std::min<uint64_t>(m_intervalNs / 10, 1000000ULL) : 0),
      m_source(TIMESTAMP_NONE),
      m_lastTimestampNs(0),
      m_nextDeadlineNs(0),
      m_initialized(false)
{
}

bool StereoOutputRateLimiter::shouldOutput(const FrameTiming& left)
{
    if (m_targetFps == 0)
        return true;

    TimestampSource source = TIMESTAMP_STEADY;
    uint64_t timestampNs = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
    if (left.hasSensorSofTimestampTsc)
    {
        source = TIMESTAMP_TSC;
        timestampNs = left.sensorSofTimestampTscNs;
    }
    else if (left.hasSensorTimestamp)
    {
        source = TIMESTAMP_SENSOR;
        timestampNs = left.sensorTimestampNs;
    }
    else if (left.frameTimeNs != 0)
    {
        source = TIMESTAMP_FRAME;
        timestampNs = left.frameTimeNs;
    }

    if (!m_initialized || source != m_source || timestampNs < m_lastTimestampNs)
    {
        m_source = source;
        m_lastTimestampNs = timestampNs;
        m_nextDeadlineNs = timestampNs;
        m_initialized = true;
    }
    else
    {
        m_lastTimestampNs = timestampNs;
    }

    if (timestampNs < m_nextDeadlineNs)
    {
        if (m_nextDeadlineNs - timestampNs > m_toleranceNs)
            return false;
        if (m_intervalNs > std::numeric_limits<uint64_t>::max() - m_nextDeadlineNs)
        {
            m_nextDeadlineNs = timestampNs;
        }
        else
        {
            m_nextDeadlineNs += m_intervalNs;
        }
        return true;
    }

    const uint64_t elapsedDeadlineNs = timestampNs - m_nextDeadlineNs;
    const uint64_t intervals = elapsedDeadlineNs / m_intervalNs + 1;
    if (intervals > (std::numeric_limits<uint64_t>::max() - m_nextDeadlineNs) / m_intervalNs)
    {
        m_nextDeadlineNs = timestampNs;
    }
    else
    {
        m_nextDeadlineNs += intervals * m_intervalNs;
    }
    return true;
}

int64_t signedDeltaNs(uint64_t rightNs, uint64_t leftNs)
{
    if (rightNs >= leftNs)
        return static_cast<int64_t>(rightNs - leftNs);
    return -static_cast<int64_t>(leftNs - rightNs);
}


namespace
{
bool applyCaptureMetadata(const Argus::CaptureMetadata* metadata, FrameTiming* timing)
{
    if (!metadata || !timing)
        return false;
    const Argus::ICaptureMetadata* iMetadata = Argus::interface_cast<const Argus::ICaptureMetadata>(metadata);
    if (iMetadata)
    {
        timing->sourceIndex = iMetadata->getSourceIndex();
        timing->sensorExposureTimeNs = iMetadata->getSensorExposureTime();
        timing->hasSensorExposureTime = true;
        timing->sensorAnalogGain = iMetadata->getSensorAnalogGain();
        timing->hasSensorAnalogGain = true;
        timing->ispDigitalGain = iMetadata->getIspDigitalGain();
        timing->hasIspDigitalGain = true;
        const Argus::AwbMode awbMode = iMetadata->getAwbMode();
        std::snprintf(timing->awbModeName, sizeof(timing->awbModeName), "%s", awbMode.getName());
        timing->hasAwbMode = true;
        timing->awbCct = iMetadata->getAwbCct();
        timing->hasAwbCct = true;
        const Argus::BayerTuple<float> awbGains = iMetadata->getAwbGains();
        timing->awbGainR = awbGains.r();
        timing->awbGainGEven = awbGains.gEven();
        timing->awbGainGOdd = awbGains.gOdd();
        timing->awbGainB = awbGains.b();
        timing->hasAwbGains = true;
        timing->hasSourceIndex = true;
        timing->sensorTimestampNs = iMetadata->getSensorTimestamp();
        timing->hasSensorTimestamp = true;
        if (timing->frameNumber == 0)
            timing->frameNumber = iMetadata->getCaptureId();
        if (timing->frameTimeNs == 0)
            timing->frameTimeNs = timing->sensorTimestampNs;
    }
    const Argus::Ext::ISensorTimestampTsc* iSensorTimestampTsc =
        Argus::interface_cast<const Argus::Ext::ISensorTimestampTsc>(metadata);
    if (iSensorTimestampTsc)
    {
        const uint64_t tsc0 = iSensorTimestampTsc->getSensorSofTimestampTsc();
        const uint64_t tsc1 = iSensorTimestampTsc->getSensorSofTimestampTsc2();
        const uint64_t selectedTsc = timing->hasSourceIndex && timing->sourceIndex == 1 ? tsc1 : tsc0;
        if (selectedTsc != 0)
        {
            timing->sensorSofTimestampTscNs = selectedTsc;
            timing->hasSensorSofTimestampTsc = true;
        }
    }
    return true;
}
} // namespace

bool readFrameTiming(EGLStream::Frame* frame, const char* label, FrameTiming* timing)
{
    if (timing)
        *timing = FrameTiming();

    EGLStream::IFrame* iFrame = Argus::interface_cast<EGLStream::IFrame>(frame);
    if (!iFrame)
    {
        std::fprintf(stderr, "Failed to get %s EGLStream IFrame interface\n", label);
        return false;
    }

    if (!timing)
        return true;

    timing->frameNumber = iFrame->getNumber();
    timing->frameTimeNs = iFrame->getTime();
    EGLStream::IArgusCaptureMetadata* iArgusCaptureMetadata =
        Argus::interface_cast<EGLStream::IArgusCaptureMetadata>(frame);
    return !iArgusCaptureMetadata || applyCaptureMetadata(iArgusCaptureMetadata->getMetadata(), timing);
}

bool readFrameTiming(Argus::Buffer* buffer, const char* label, FrameTiming* timing)
{
    if (timing)
        *timing = FrameTiming();
    Argus::IBuffer* iBuffer = Argus::interface_cast<Argus::IBuffer>(buffer);
    if (!iBuffer)
    {
        std::fprintf(stderr, "Failed to get %s Argus IBuffer interface\n", label);
        return false;
    }
    if (!timing)
        return true;
    const Argus::CaptureMetadata* metadata = iBuffer->getMetadata();
    if (!metadata)
    {
        std::fprintf(stderr, "Failed to get %s direct-buffer metadata\n", label);
        return false;
    }
    return applyCaptureMetadata(metadata, timing);
}

} // namespace argus_sender

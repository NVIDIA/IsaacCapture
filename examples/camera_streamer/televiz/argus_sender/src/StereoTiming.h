/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_STEREO_TIMING_H
#define ARGUS_SENDER_STEREO_TIMING_H

#include <Argus/Argus.h>
#include <EGLStream/EGLStream.h>

#include <cstdint>

namespace argus_sender
{

enum SenderClockDomain
{
    SENDER_CLOCK_UNKNOWN = 0,
    SENDER_CLOCK_TAI = 1,
    SENDER_CLOCK_REALTIME = 2
};

struct FrameTiming
{
    uint64_t frameNumber = 0;
    uint64_t frameNumberGap = 0;
    uint64_t frameTimeNs = 0;
    uint64_t acquiredSteadyNs = 0;
    // Tegra TSC sampled immediately after the Argus acquire call returned.
    // This shares the VI SOF timestamp's clock domain on Orin.
    uint64_t acquiredTscNs = 0;
    uint64_t sensorExposureTimeNs = 0;
    bool hasSensorExposureTime = false;
    float sensorAnalogGain = 0.0f;
    bool hasSensorAnalogGain = false;
    float ispDigitalGain = 0.0f;
    bool hasIspDigitalGain = false;
    char awbModeName[32] = {};
    bool hasAwbMode = false;
    uint32_t awbCct = 0;
    bool hasAwbCct = false;
    float awbGainR = 0.0f;
    float awbGainGEven = 0.0f;
    float awbGainGOdd = 0.0f;
    float awbGainB = 0.0f;
    bool hasAwbGains = false;
    uint32_t sourceIndex = 0;
    bool hasSourceIndex = false;
    uint64_t sensorTimestampNs = 0;
    bool hasSensorTimestamp = false;
    uint64_t sensorSofTimestampTscNs = 0;
    bool hasSensorSofTimestampTsc = false;
};

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

int64_t signedDeltaNs(uint64_t rightNs, uint64_t leftNs);
uint64_t readSenderClockNs(uint32_t* clockDomain = 0);
uint64_t readTscNowNs();
bool readFrameTiming(EGLStream::Frame* frame, const char* label, FrameTiming* timing);
bool readFrameTiming(Argus::Buffer* buffer, const char* label, FrameTiming* timing);

} // namespace argus_sender

#endif // ARGUS_SENDER_STEREO_TIMING_H

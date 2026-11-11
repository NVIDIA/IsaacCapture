/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_FRAME_TIMING_H
#define ARGUS_SENDER_FRAME_TIMING_H

// Per-frame timing and the sender's wall clock, with no camera SDK type: the encoder, the AVS1 assembler and every
// capture backend use it.

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
    // Tegra TSC sampled immediately after the frame was acquired.
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
    // The sensor's capture instant on the sender's CLOCK_MONOTONIC, with the mapping's uncertainty and anchor age;
    // present only when a checked mapping produced it.
    bool hasCaptureMonotonic = false;
    int64_t captureMonotonicNs = 0;
    int64_t captureUncertaintyNs = 0;
    int64_t captureAnchorAgeNs = 0;
    uint64_t sensorSofTimestampTscNs = 0;
    bool hasSensorSofTimestampTsc = false;
};

int64_t signedDeltaNs(uint64_t rightNs, uint64_t leftNs);
uint64_t readSenderClockNs(uint32_t* clockDomain = 0);
uint64_t readTscNowNs();
} // namespace argus_sender

#endif // ARGUS_SENDER_FRAME_TIMING_H

/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_AVS1_WIRE_H
#define ARGUS_SENDER_AVS1_WIRE_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace argus_sender
{

const uint16_t kAvs1Version = 1;
const size_t kAvs1LengthPrefixBytes = 4;
const size_t kAvs1HeaderBytes = 64;
const size_t kAvs1StreamDescriptorBytes = 40;
const size_t kAvs1SampleDescriptorBytes = 72;

enum Avs1BundleFlags
{
    AVS1_BUNDLE_SENDER_CLOCK_VALID = 1U << 0,
    AVS1_BUNDLE_SENDER_CLOCK_TAI = 1U << 1,
    AVS1_BUNDLE_SENDER_TSC_VALID = 1U << 2
};

enum Avs1StreamFlags
{
    AVS1_STREAM_PRESENT = 1U << 0,
    AVS1_STREAM_STEREO = 1U << 1
};

enum Avs1SampleFlags
{
    AVS1_SAMPLE_PRESENT = 1U << 0,
    AVS1_SAMPLE_SOF_TSC_VALID = 1U << 1,
    AVS1_SAMPLE_FRAME_TIMESTAMP_VALID = 1U << 2,
    AVS1_SAMPLE_SENDER_TIMING_VALID = 1U << 3
};

enum Avs1Codec
{
    AVS1_CODEC_RAW = 0,
    AVS1_CODEC_HEVC = 1
};

enum Avs1PixelFormat
{
    AVS1_PIXEL_FORMAT_NV12 = 1
};

enum Avs1SampleRole
{
    AVS1_ROLE_MONO = 0,
    AVS1_ROLE_LEFT = 1,
    AVS1_ROLE_RIGHT = 2
};

struct Avs1SampleRecord
{
    uint16_t flags = 0;
    uint8_t cameraIndex = 0;
    uint8_t role = AVS1_ROLE_MONO;
    uint32_t deviceIndex = 0;
    uint32_t sensorModeIndex = 0;
    uint64_t frameNumber = 0;
    uint64_t sofTscNs = 0;
    uint64_t frameTimestampNs = 0;
    uint64_t senderCaptureReadyNs = 0;
    uint64_t senderProcessSubmitNs = 0;
    uint64_t senderProcessDoneNs = 0;
    std::vector<uint8_t> payload;
};

struct Avs1StreamRecord
{
    uint16_t flags = 0;
    uint8_t codec = AVS1_CODEC_HEVC;
    uint8_t pixelFormat = AVS1_PIXEL_FORMAT_NV12;
    uint16_t rotation = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t fpsNumerator = 0;
    uint32_t fpsDenominator = 1;
    uint64_t sequence = 0;
    std::vector<Avs1SampleRecord> samples;
};

struct Avs1Bundle
{
    uint16_t flags = 0;
    uint64_t sequence = 0;
    uint64_t sessionId = 0;
    uint64_t senderCaptureReadyNs = 0;
    uint64_t senderTcpEnqueueNs = 0;
    uint64_t senderCaptureReadyTscNs = 0;
    std::vector<Avs1StreamRecord> streams;
};

bool buildAvs1WireRecord(const Avs1Bundle& bundle, std::vector<uint8_t>* out, const char** error = 0);
bool parseAvs1WireRecord(const uint8_t* data, size_t size, Avs1Bundle* bundle, const char** error = 0);

} // namespace argus_sender

#endif // ARGUS_SENDER_AVS1_WIRE_H

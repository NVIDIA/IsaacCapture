/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Avs1Wire.h"
#include "TestHarness.h"

#include <cstdint>
#include <vector>

namespace
{

argus_sender::Avs1SampleRecord sample(uint8_t camera, uint8_t role, uint32_t device, const std::vector<uint8_t>& payload)
{
    argus_sender::Avs1SampleRecord value;
    value.flags = argus_sender::AVS1_SAMPLE_PRESENT | argus_sender::AVS1_SAMPLE_SOF_TSC_VALID |
                  argus_sender::AVS1_SAMPLE_FRAME_TIMESTAMP_VALID | argus_sender::AVS1_SAMPLE_SENDER_TIMING_VALID;
    value.cameraIndex = camera;
    value.role = role;
    value.deviceIndex = device;
    value.sensorModeIndex = 2;
    value.frameNumber = 100 + camera;
    value.sofTscNs = 1000000 + camera * 7;
    value.frameTimestampNs = 2000000 + camera * 9;
    value.senderCaptureReadyNs = 3000000;
    value.senderProcessSubmitNs = 3000100;
    value.senderProcessDoneNs = 3000200 + camera;
    value.payload = payload;
    return value;
}

argus_sender::Avs1Bundle mixedBundle()
{
    argus_sender::Avs1Bundle bundle;
    bundle.flags = argus_sender::AVS1_BUNDLE_SENDER_CLOCK_VALID | argus_sender::AVS1_BUNDLE_SENDER_TSC_VALID;
    bundle.sequence = 9;
    bundle.sessionId = 0x1122334455667788ULL;
    bundle.senderCaptureReadyNs = 3000000;
    bundle.senderTcpEnqueueNs = 3000300;
    bundle.senderCaptureReadyTscNs = 1234567;

    argus_sender::Avs1StreamRecord stereo;
    stereo.flags = argus_sender::AVS1_STREAM_PRESENT | argus_sender::AVS1_STREAM_STEREO;
    stereo.codec = argus_sender::AVS1_CODEC_HEVC;
    stereo.width = 1280;
    stereo.height = 720;
    stereo.fpsNumerator = 60;
    stereo.sequence = 9;
    stereo.samples.push_back(sample(0, argus_sender::AVS1_ROLE_LEFT, 0, { 0x00, 0x00, 0x01 }));
    stereo.samples.push_back(sample(1, argus_sender::AVS1_ROLE_RIGHT, 1, { 0x26, 0x01 }));
    bundle.streams.push_back(stereo);

    argus_sender::Avs1StreamRecord mono;
    mono.codec = argus_sender::AVS1_CODEC_RAW;
    mono.width = 4;
    mono.height = 2;
    mono.fpsNumerator = 30;
    mono.samples.resize(1);
    mono.samples[0].cameraIndex = 0;
    mono.samples[0].role = argus_sender::AVS1_ROLE_MONO;
    mono.samples[0].deviceIndex = 2;
    bundle.streams.push_back(mono);
    return bundle;
}

void testStreamValidation(TestContext& test)
{
    const char* error = nullptr;
    argus_sender::Avs1Bundle parsed;
    std::vector<uint8_t> wire;
    for (uint16_t rotation : { 0, 180 })
    {
        auto bundle = mixedBundle();
        bundle.streams[0].rotation = rotation;
        TEST_CHECK(test, argus_sender::buildAvs1WireRecord(bundle, &wire, &error));
        TEST_CHECK(test, argus_sender::parseAvs1WireRecord(wire.data(), wire.size(), &parsed, &error));
    }
    auto invalid = mixedBundle();
    invalid.streams[0].rotation = 90;
    TEST_CHECK(test, !argus_sender::buildAvs1WireRecord(invalid, &wire, &error));
    for (size_t stream = 0; stream < 2; ++stream)
    {
        invalid = mixedBundle();
        invalid.streams[stream].sequence = stream == 0 ? 0 : 1;
        TEST_CHECK(test, !argus_sender::buildAvs1WireRecord(invalid, &wire, &error));
    }

    std::vector<uint8_t> valid;
    TEST_CHECK(test, argus_sender::buildAvs1WireRecord(mixedBundle(), &valid, &error));
    const size_t firstStream = argus_sender::kAvs1LengthPrefixBytes + argus_sender::kAvs1HeaderBytes;
    wire = valid;
    wire[firstStream + 7] = 90;
    TEST_CHECK(test, !argus_sender::parseAvs1WireRecord(wire.data(), wire.size(), &parsed, &error));
    wire = valid;
    wire[firstStream + 31] = 0;
    TEST_CHECK(test, !argus_sender::parseAvs1WireRecord(wire.data(), wire.size(), &parsed, &error));
    wire = valid;
    wire[firstStream + argus_sender::kAvs1StreamDescriptorBytes + 31] = 1;
    TEST_CHECK(test, !argus_sender::parseAvs1WireRecord(wire.data(), wire.size(), &parsed, &error));
}

void testBundleValidation(TestContext& test)
{
    const char* error = nullptr;
    argus_sender::Avs1Bundle parsed;
    auto bundle = mixedBundle();
    bundle.flags = argus_sender::AVS1_BUNDLE_SENDER_CLOCK_VALID;
    bundle.senderCaptureReadyTscNs = 0;
    std::vector<uint8_t> valid;
    TEST_CHECK(test, argus_sender::buildAvs1WireRecord(bundle, &valid, &error));
    // Mutate the on-wire fields independently of the builder's checks.
    for (uint8_t flags : { 0x80, 0x02 })
    {
        auto wire = valid;
        wire[4 + 7] = flags;
        TEST_CHECK(test, !argus_sender::parseAvs1WireRecord(wire.data(), wire.size(), &parsed, &error));
    }
    for (size_t field : { 16, 24 })
    {
        auto wire = valid;
        for (size_t byte = 0; byte < 8; ++byte)
            wire[4 + field + byte] = 0;
        TEST_CHECK(test, !argus_sender::parseAvs1WireRecord(wire.data(), wire.size(), &parsed, &error));
    }
}

void testPayloadCoverage(TestContext& test)
{
    const char* error = nullptr;
    argus_sender::Avs1Bundle parsed;
    std::vector<uint8_t> valid;
    TEST_CHECK(test, argus_sender::buildAvs1WireRecord(mixedBundle(), &valid, &error));
    const size_t secondSample = argus_sender::kAvs1LengthPrefixBytes + argus_sender::kAvs1HeaderBytes +
                                2 * argus_sender::kAvs1StreamDescriptorBytes + argus_sender::kAvs1SampleDescriptorBytes;
    // Two nonempty ranges still sum to five bytes when the second overlaps the first.
    for (uint8_t offset : { 0, 2, 4 })
    {
        auto wire = valid;
        wire[secondSample + 67] = offset;
        TEST_CHECK(test, !argus_sender::parseAvs1WireRecord(wire.data(), wire.size(), &parsed, &error));
    }
}

} // namespace

int main()
{
    TestContext test("avs1_wire");
    testStreamValidation(test);
    testBundleValidation(test);
    testPayloadCoverage(test);
    {
        const char* descriptorError = nullptr;
        std::vector<uint8_t> validRecord;
        argus_sender::Avs1Bundle result;
        TEST_CHECK(test, argus_sender::buildAvs1WireRecord(mixedBundle(), &validRecord, &descriptorError));
        const size_t firstStream = argus_sender::kAvs1LengthPrefixBytes + argus_sender::kAvs1HeaderBytes;
        // Oversized stereo count and overlapping mono range must fail at the descriptor,
        // before allocating sample vectors or reading sample payloads.
        for (size_t field : { firstStream + 35, firstStream + argus_sender::kAvs1StreamDescriptorBytes + 33 })
        {
            auto malformed = validRecord;
            malformed[field] = field == firstStream + 35 ? 3 : 0;
            TEST_CHECK(
                test, !argus_sender::parseAvs1WireRecord(malformed.data(), malformed.size(), &result, &descriptorError));
            TEST_CHECK_EQ(test, std::string(descriptorError), std::string("invalid AVS1 stream descriptor index"));
        }
    }
    argus_sender::Avs1Bundle input = mixedBundle();
    std::vector<uint8_t> wire;
    const char* error = 0;
    TEST_CHECK(test, argus_sender::buildAvs1WireRecord(input, &wire, &error));
    TEST_CHECK_EQ(test, wire.size(), static_cast<size_t>(369));
    TEST_CHECK_EQ(test, wire[0], static_cast<uint8_t>(0));
    TEST_CHECK_EQ(test, wire[1], static_cast<uint8_t>(0));
    TEST_CHECK_EQ(test, wire[2], static_cast<uint8_t>(1));
    TEST_CHECK_EQ(test, wire[3], static_cast<uint8_t>(109));
    TEST_CHECK_EQ(test, wire[4], static_cast<uint8_t>('A'));
    TEST_CHECK_EQ(test, wire[7], static_cast<uint8_t>('1'));

    argus_sender::Avs1Bundle parsed;
    TEST_CHECK(test, argus_sender::parseAvs1WireRecord(wire.data(), wire.size(), &parsed, &error));
    TEST_CHECK_EQ(test, parsed.sequence, 9ULL);
    TEST_CHECK_EQ(test, parsed.streams.size(), static_cast<size_t>(2));
    TEST_CHECK_EQ(test, parsed.streams[0].samples[1].sofTscNs, 1000007ULL);
    TEST_CHECK_EQ(test, parsed.streams[0].samples[0].payload.size(), static_cast<size_t>(3));
    TEST_CHECK(test, !(parsed.streams[1].flags & argus_sender::AVS1_STREAM_PRESENT));
    TEST_CHECK(test, parsed.streams[1].samples[0].payload.empty());
    // Header offset 56 carries sender_capture_ready_tsc_ns (1234567 = 0x12D687).
    TEST_CHECK_EQ(test, wire[4 + 56 + 4], static_cast<uint8_t>(0x00));
    TEST_CHECK_EQ(test, wire[4 + 56 + 5], static_cast<uint8_t>(0x12));
    TEST_CHECK_EQ(test, wire[4 + 56 + 6], static_cast<uint8_t>(0xD6));
    TEST_CHECK_EQ(test, wire[4 + 56 + 7], static_cast<uint8_t>(0x87));
    TEST_CHECK_EQ(test, parsed.flags,
                  static_cast<uint16_t>(argus_sender::AVS1_BUNDLE_SENDER_CLOCK_VALID |
                                        argus_sender::AVS1_BUNDLE_SENDER_TSC_VALID));
    TEST_CHECK_EQ(test, parsed.senderCaptureReadyTscNs, 1234567ULL);

    argus_sender::Avs1Bundle raw = mixedBundle();
    raw.streams.erase(raw.streams.begin());
    raw.streams[0].flags = argus_sender::AVS1_STREAM_PRESENT;
    raw.streams[0].sequence = 1;
    raw.streams[0].samples[0] = sample(0, argus_sender::AVS1_ROLE_MONO, 2, std::vector<uint8_t>(12, 0x80));
    TEST_CHECK(test, argus_sender::buildAvs1WireRecord(raw, &wire, &error));
    TEST_CHECK(test, argus_sender::parseAvs1WireRecord(wire.data(), wire.size(), &parsed, &error));
    TEST_CHECK_EQ(test, parsed.streams[0].samples[0].payload.size(), static_cast<size_t>(12));

    wire[4] = 'X';
    TEST_CHECK(test, !argus_sender::parseAvs1WireRecord(wire.data(), wire.size(), &parsed, &error));

    argus_sender::Avs1Bundle invalid = mixedBundle();
    invalid.streams[0].samples[1].payload.clear();
    TEST_CHECK(test, !argus_sender::buildAvs1WireRecord(invalid, &wire, &error));

    // Without SENDER_TSC_VALID offset 56 stays the legacy reserved zero.
    argus_sender::Avs1Bundle legacy = mixedBundle();
    legacy.flags = argus_sender::AVS1_BUNDLE_SENDER_CLOCK_VALID;
    legacy.senderCaptureReadyTscNs = 0;
    TEST_CHECK(test, argus_sender::buildAvs1WireRecord(legacy, &wire, &error));
    for (size_t i = 0; i < 8; ++i)
        TEST_CHECK_EQ(test, wire[4 + 56 + i], static_cast<uint8_t>(0));
    TEST_CHECK(test, argus_sender::parseAvs1WireRecord(wire.data(), wire.size(), &parsed, &error));
    TEST_CHECK(test, !(parsed.flags & argus_sender::AVS1_BUNDLE_SENDER_TSC_VALID));
    TEST_CHECK_EQ(test, parsed.senderCaptureReadyTscNs, 0ULL);

    // Builder rejects flag-without-field, field-without-flag, TSC-without-clock.
    argus_sender::Avs1Bundle tscFlagOnly = mixedBundle();
    tscFlagOnly.senderCaptureReadyTscNs = 0;
    TEST_CHECK(test, !argus_sender::buildAvs1WireRecord(tscFlagOnly, &wire, &error));
    argus_sender::Avs1Bundle tscFieldOnly = mixedBundle();
    tscFieldOnly.flags = argus_sender::AVS1_BUNDLE_SENDER_CLOCK_VALID;
    TEST_CHECK(test, !argus_sender::buildAvs1WireRecord(tscFieldOnly, &wire, &error));
    argus_sender::Avs1Bundle tscWithoutClock = mixedBundle();
    tscWithoutClock.flags = argus_sender::AVS1_BUNDLE_SENDER_TSC_VALID;
    TEST_CHECK(test, !argus_sender::buildAvs1WireRecord(tscWithoutClock, &wire, &error));

    // Parser rejects the same three shapes, patched into a valid record.
    // Bundle flags are the u16 at record offset 6; wire[4 + 7] is its low byte.
    std::vector<uint8_t> valid;
    TEST_CHECK(test, argus_sender::buildAvs1WireRecord(mixedBundle(), &valid, &error));
    TEST_CHECK(test, argus_sender::parseAvs1WireRecord(valid.data(), valid.size(), &parsed, &error));
    wire = valid;
    for (size_t i = 0; i < 8; ++i)
        wire[4 + 56 + i] = 0;
    TEST_CHECK(test, !argus_sender::parseAvs1WireRecord(wire.data(), wire.size(), &parsed, &error));
    wire = valid;
    wire[4 + 7] = static_cast<uint8_t>(wire[4 + 7] & ~argus_sender::AVS1_BUNDLE_SENDER_TSC_VALID);
    TEST_CHECK(test, !argus_sender::parseAvs1WireRecord(wire.data(), wire.size(), &parsed, &error));
    wire = valid;
    wire[4 + 7] = static_cast<uint8_t>(wire[4 + 7] & ~argus_sender::AVS1_BUNDLE_SENDER_CLOCK_VALID);
    TEST_CHECK(test, !argus_sender::parseAvs1WireRecord(wire.data(), wire.size(), &parsed, &error));
    return test.finish();
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Avs1Wire.h"
#include "TestHarness.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
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

} // namespace

namespace
{

// One stream, one sample, a three-byte payload, with the timing extension; the receiver's parser test uses the same
// bytes.
const char kGoldenHex[] =
    "000000cb415653310001000d000000400001000100000000000000050102030405060708000000003b9aca0000000000"
    "3ba26b2000000088000000030000000077359400000000010101000000000040000000300000001e0000000100000000"
    "00000005000000010000000000000000000300000000000000000000000000000000004d000000007604670000000000"
    "000000000000000000000000000000000000000000000000000000000000000000000003000000003b7c458000000bb8"
    "0ee6b2800001000100000000000001";

} // namespace

int main()
{
    TestContext test("avs1_wire");
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
    // The timing extension: one 24-byte record per sample, after the sample descriptors, counted in descriptor_bytes.
    {
        argus_sender::Avs1Bundle timed = mixedBundle();
        timed.flags |= argus_sender::AVS1_BUNDLE_TIMING_EXT;
        argus_sender::Avs1TimingRecord good;
        good.flags = argus_sender::AVS1_TIMING_VALID;
        good.clockDomain = argus_sender::AVS1_TIMING_CLOCK_MONOTONIC;
        good.sensorCaptureNs = 40505284984832ULL;
        good.uncertaintyNs = 4321;
        good.anchorAgeNs = 987654;
        timed.streams[0].samples[0].timing = good; // the other samples carry no capture claim (all zero, VALID clear)
        std::vector<uint8_t> timedWire;
        TEST_CHECK(test, argus_sender::buildAvs1WireRecord(timed, &timedWire, &error));
        TEST_CHECK_EQ(test, timedWire.size(), wire.size() + 3 * argus_sender::kAvs1TimingRecordBytes);
        argus_sender::Avs1Bundle timedParsed;
        TEST_CHECK(test, argus_sender::parseAvs1WireRecord(timedWire.data(), timedWire.size(), &timedParsed, &error));
        TEST_CHECK_EQ(test, timedParsed.streams[0].samples[0].timing.sensorCaptureNs, good.sensorCaptureNs);
        TEST_CHECK_EQ(test, timedParsed.streams[0].samples[0].timing.uncertaintyNs, good.uncertaintyNs);
        TEST_CHECK_EQ(test, timedParsed.streams[0].samples[0].timing.anchorAgeNs, good.anchorAgeNs);
        TEST_CHECK_EQ(test, timedParsed.streams[0].samples[0].timing.clockDomain, good.clockDomain);
        TEST_CHECK_EQ(test, timedParsed.streams[0].samples[1].timing.flags, static_cast<uint16_t>(0));
        TEST_CHECK_EQ(test, timedParsed.streams[1].samples[0].timing.sensorCaptureNs, static_cast<uint64_t>(0));
        // Without the extension flag the layout is unchanged and a timing record is refused.
        argus_sender::Avs1Bundle claimed = mixedBundle();
        claimed.streams[0].samples[0].timing = good;
        TEST_CHECK(test, !argus_sender::buildAvs1WireRecord(claimed, &timedWire, &error));
        // The realtime domain round-trips like the monotonic one.
        {
            argus_sender::Avs1Bundle realtime = timed;
            realtime.streams[0].samples[0].timing.clockDomain = argus_sender::AVS1_TIMING_CLOCK_REALTIME;
            realtime.streams[0].samples[0].timing.sensorCaptureNs = 1762900000123456789ULL;
            std::vector<uint8_t> realtimeWire;
            TEST_CHECK(test, argus_sender::buildAvs1WireRecord(realtime, &realtimeWire, &error));
            argus_sender::Avs1Bundle realtimeParsed;
            TEST_CHECK(test, argus_sender::parseAvs1WireRecord(
                                 realtimeWire.data(), realtimeWire.size(), &realtimeParsed, &error));
            const argus_sender::Avs1TimingRecord& back = realtimeParsed.streams[0].samples[0].timing;
            TEST_CHECK_EQ(test, back.clockDomain, static_cast<uint16_t>(argus_sender::AVS1_TIMING_CLOCK_REALTIME));
            TEST_CHECK_EQ(test, back.sensorCaptureNs, 1762900000123456789ULL);
            TEST_CHECK_EQ(test, back.flags, static_cast<uint16_t>(argus_sender::AVS1_TIMING_VALID));
            // Domain 2 without VALID and a non-zero field is refused.
            argus_sender::Avs1Bundle noValid = timed;
            noValid.streams[0].samples[0].timing.flags = 0;
            noValid.streams[0].samples[0].timing.clockDomain = argus_sender::AVS1_TIMING_CLOCK_REALTIME;
            TEST_CHECK(test, !argus_sender::buildAvs1WireRecord(noValid, &realtimeWire, &error));
            // An unknown domain with VALID is refused, on build and on parse.
            const uint16_t unknownDomains[] = { 0, 3, 0xFFFF };
            for (size_t i = 0; i < 3; ++i)
            {
                argus_sender::Avs1Bundle unknown = timed;
                unknown.streams[0].samples[0].timing.clockDomain = unknownDomains[i];
                TEST_CHECK(test, !argus_sender::buildAvs1WireRecord(unknown, &realtimeWire, &error));
            }
            // Patch a built record: clock_domain is the u16 at timing-record offset 18.
            const size_t domainOffset = 4 + 64 + 2 * 40 + 3 * 72 + 18;
            std::vector<uint8_t> goodWire;
            TEST_CHECK(test, argus_sender::buildAvs1WireRecord(timed, &goodWire, &error));
            for (size_t i = 0; i < 3; ++i)
            {
                std::vector<uint8_t> patched = goodWire;
                patched[domainOffset] = static_cast<uint8_t>(unknownDomains[i] >> 8);
                patched[domainOffset + 1] = static_cast<uint8_t>(unknownDomains[i] & 0xff);
                TEST_CHECK(
                    test, !argus_sender::parseAvs1WireRecord(patched.data(), patched.size(), &realtimeParsed, &error));
            }
            std::vector<uint8_t> patched = goodWire;
            patched[domainOffset + 1] = 2;
            TEST_CHECK(test, argus_sender::parseAvs1WireRecord(patched.data(), patched.size(), &realtimeParsed, &error));
        }
        // VALID needs the monotonic or realtime domain and a nonzero time; without VALID everything must be zero.
        argus_sender::Avs1Bundle bad = timed;
        bad.streams[0].samples[0].timing.clockDomain = argus_sender::AVS1_TIMING_CLOCK_NONE;
        TEST_CHECK(test, !argus_sender::buildAvs1WireRecord(bad, &timedWire, &error));
        bad = timed;
        bad.streams[0].samples[0].timing.sensorCaptureNs = 0;
        TEST_CHECK(test, !argus_sender::buildAvs1WireRecord(bad, &timedWire, &error));
        bad = timed;
        bad.streams[0].samples[1].timing.uncertaintyNs = 5;
        TEST_CHECK(test, !argus_sender::buildAvs1WireRecord(bad, &timedWire, &error));
        bad = timed;
        bad.streams[0].samples[0].timing.flags = 0x8001;
        TEST_CHECK(test, !argus_sender::buildAvs1WireRecord(bad, &timedWire, &error));
        // A parser refuses a nonzero reserved field and a record count that does not match descriptor_bytes.
        std::vector<uint8_t> tampered = timedWire;
        timedWire.clear();
        TEST_CHECK(test, argus_sender::buildAvs1WireRecord(timed, &timedWire, &error));
        tampered = timedWire;
        const size_t timingStart = 4 + 64 + 2 * 40 + 3 * 72;
        tampered[timingStart + 23] = 1; // first record's reserved word
        TEST_CHECK(test, !argus_sender::parseAvs1WireRecord(tampered.data(), tampered.size(), &timedParsed, &error));
        tampered = timedWire;
        tampered[4 + 48 + 3] = static_cast<uint8_t>(tampered[4 + 48 + 3] - 24); // descriptor_bytes short by one record
        TEST_CHECK(test, !argus_sender::parseAvs1WireRecord(tampered.data(), tampered.size(), &timedParsed, &error));

        // Golden vector shared with the receiver's parser: one stream, one sample, a three-byte payload.
        argus_sender::Avs1Bundle golden;
        golden.flags = argus_sender::AVS1_BUNDLE_SENDER_CLOCK_VALID | argus_sender::AVS1_BUNDLE_SENDER_TSC_VALID |
                       argus_sender::AVS1_BUNDLE_TIMING_EXT;
        golden.sequence = 5;
        golden.sessionId = 0x0102030405060708ULL;
        golden.senderCaptureReadyNs = 1000000000ULL;
        golden.senderTcpEnqueueNs = 1000500000ULL;
        golden.senderCaptureReadyTscNs = 2000000000ULL;
        argus_sender::Avs1StreamRecord stream;
        stream.flags = argus_sender::AVS1_STREAM_PRESENT;
        stream.codec = argus_sender::AVS1_CODEC_HEVC;
        stream.width = 64;
        stream.height = 48;
        stream.fpsNumerator = 30;
        stream.sequence = 5;
        argus_sender::Avs1SampleRecord one;
        one.flags = argus_sender::AVS1_SAMPLE_PRESENT | argus_sender::AVS1_SAMPLE_SOF_TSC_VALID;
        one.frameNumber = 77;
        one.sofTscNs = 1980000000ULL;
        one.payload = { 0x00, 0x00, 0x01 };
        one.timing.flags = argus_sender::AVS1_TIMING_VALID;
        one.timing.clockDomain = argus_sender::AVS1_TIMING_CLOCK_MONOTONIC;
        one.timing.sensorCaptureNs = 998000000ULL;
        one.timing.uncertaintyNs = 3000;
        one.timing.anchorAgeNs = 250000000U;
        stream.samples.push_back(one);
        golden.streams.push_back(stream);
        std::vector<uint8_t> goldenWire;
        TEST_CHECK(test, argus_sender::buildAvs1WireRecord(golden, &goldenWire, &error));
        std::string hex;
        char pair[3];
        for (size_t i = 0; i < goldenWire.size(); ++i)
        {
            std::snprintf(pair, sizeof(pair), "%02x", goldenWire[i]);
            hex += pair;
        }
        if (std::getenv("AVS1_PRINT_GOLDEN"))
            std::printf("%s\n", hex.c_str());
        TEST_CHECK_EQ(test, hex, std::string(kGoldenHex));
    }

    return test.finish();
}

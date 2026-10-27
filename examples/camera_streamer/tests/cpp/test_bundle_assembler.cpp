/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "EncodedFrameSink.h"
#include "FrameBundleAssembler.h"
#include "PacketSink.h"
#include "TestHarness.h"

#include <chrono>
#include <cstdint>
#include <utility>
#include <vector>

namespace
{

class FakePacketSink : public argus_sender::PacketSink
{
public:
    bool enqueueFramed(std::vector<uint8_t>&& packet) override
    {
        packets.push_back(std::move(packet));
        return true;
    }

    bool enqueueFramedWithTiming(std::vector<uint8_t>&& packet, const argus_sender::PacketTiming& timing) override
    {
        timings.push_back(timing);
        packets.push_back(std::move(packet));
        return true;
    }

    bool hasError() const override
    {
        return false;
    }

    std::vector<std::vector<uint8_t>> packets;
    std::vector<argus_sender::PacketTiming> timings;
};

std::vector<argus_sender::Avs1StreamRecord> templates()
{
    std::vector<argus_sender::Avs1StreamRecord> result(2);
    result[0].flags = argus_sender::AVS1_STREAM_STEREO;
    result[0].codec = argus_sender::AVS1_CODEC_HEVC;
    result[0].width = 1280;
    result[0].height = 720;
    result[0].fpsNumerator = 60;
    result[0].samples.resize(2);
    result[0].samples[0].cameraIndex = 0;
    result[0].samples[0].role = argus_sender::AVS1_ROLE_LEFT;
    result[0].samples[1].cameraIndex = 1;
    result[0].samples[1].role = argus_sender::AVS1_ROLE_RIGHT;

    result[1].codec = argus_sender::AVS1_CODEC_RAW;
    result[1].width = 4;
    result[1].height = 2;
    result[1].fpsNumerator = 30;
    result[1].samples.resize(1);
    result[1].samples[0].cameraIndex = 0;
    result[1].samples[0].role = argus_sender::AVS1_ROLE_MONO;
    return result;
}

argus_sender::FrameTiming timing(uint64_t frame, uint64_t tsc)
{
    argus_sender::FrameTiming result;
    result.frameNumber = frame;
    result.frameTimeNs = tsc + 100;
    result.sensorSofTimestampTscNs = tsc;
    result.hasSensorSofTimestampTsc = true;
    return result;
}

} // namespace

int main()
{
    TestContext test("bundle_assembler");
    FakePacketSink sink;
    argus_sender::FrameBundleAssembler assembler(&sink, 77, templates());

    std::vector<argus_sender::BundleStreamInput> bundle1(2);
    bundle1[0].present = true;
    bundle1[0].sequence = 1;
    bundle1[0].timings.push_back(timing(10, 1000));
    bundle1[0].timings.push_back(timing(11, 1004));
    const uint64_t captureStartSteadyNs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                                    std::chrono::steady_clock::now().time_since_epoch())
                                                                    .count()) -
                                          2000000ULL;
    bundle1[0].timings[0].acquiredSteadyNs = captureStartSteadyNs;
    bundle1[0].timings[1].acquiredSteadyNs = captureStartSteadyNs + 1000000ULL;
    TEST_CHECK(test, assembler.beginBundle(1, bundle1, 2000, 2100, argus_sender::SENDER_CLOCK_REALTIME, 1300));
    TEST_CHECK(test, assembler.recordCopyDone(0, 1, captureStartSteadyNs + 1400000ULL));
    TEST_CHECK(test, assembler.recordProcessStart(0, 1, captureStartSteadyNs + 1500000ULL));
    TEST_CHECK(test, assembler.recordProcessStart(0, 1, captureStartSteadyNs + 1700000ULL));

    std::vector<argus_sender::BundleStreamInput> bundle2(2);
    bundle2[1].present = true;
    bundle2[1].sequence = 1;
    bundle2[1].timings.push_back(timing(20, 3000));
    const uint64_t rawCaptureStartSteadyNs = captureStartSteadyNs + 500000ULL;
    bundle2[1].timings[0].acquiredSteadyNs = rawCaptureStartSteadyNs;
    TEST_CHECK(test, assembler.beginBundle(2, bundle2, 4000, 4100, argus_sender::SENDER_CLOCK_REALTIME, 0));
    TEST_CHECK(test, assembler.recordCopyDone(1, 2, rawCaptureStartSteadyNs + 200000ULL));
    TEST_CHECK(test, assembler.recordProcessStart(1, 2, rawCaptureStartSteadyNs + 300000ULL));
    TEST_CHECK(
        test, assembler.submitRaw(1, 0, 2, std::vector<uint8_t>(12, 0x80), 4200, argus_sender::SENDER_CLOCK_REALTIME));
    TEST_CHECK(test, sink.packets.empty());

    const uint8_t right[] = { 0, 0, 1, 0x26 };
    TEST_CHECK(test, assembler.encodedSink(0, 1)->enqueueEncoded(
                         right, sizeof(right), 1, 2200, argus_sender::SENDER_CLOCK_REALTIME));
    TEST_CHECK(test, sink.packets.empty());
    const uint8_t left[] = { 0, 0, 1, 0x20 };
    TEST_CHECK(test, assembler.encodedSink(0, 0)->enqueueEncoded(
                         left, sizeof(left), 1, 2201, argus_sender::SENDER_CLOCK_REALTIME));
    TEST_CHECK_EQ(test, sink.packets.size(), static_cast<size_t>(2));
    TEST_CHECK_EQ(test, sink.timings.size(), static_cast<size_t>(2));
    TEST_CHECK_EQ(test, sink.timings[0].captureStartSteadyNs, captureStartSteadyNs);
    TEST_CHECK_EQ(test, sink.timings[0].acquireSpreadNs, 1000000ULL);
    TEST_CHECK_EQ(test, sink.timings[0].captureToEncodeStartNs, 1500000ULL);
    TEST_CHECK(test, sink.timings[0].captureToProcessDoneNs != 0);
    TEST_CHECK_EQ(test, sink.timings[0].streams.size(), static_cast<size_t>(2));
    TEST_CHECK(test, sink.timings[0].streams[0].present);
    TEST_CHECK(test, sink.timings[0].streams[0].encoded);
    TEST_CHECK_EQ(test, sink.timings[0].samples.size(), static_cast<size_t>(3));
    TEST_CHECK(test, sink.timings[0].samples[0].present);
    TEST_CHECK_EQ(test, sink.timings[0].samples[0].sensorSofTscNs, 1000ULL);
    TEST_CHECK(test, sink.timings[0].samples[1].present);
    TEST_CHECK_EQ(test, sink.timings[0].samples[1].sensorSofTscNs, 1004ULL);
    TEST_CHECK(test, !sink.timings[0].samples[2].present);
    TEST_CHECK_EQ(test, sink.timings[0].streams[0].captureStartSteadyNs, captureStartSteadyNs);
    TEST_CHECK_EQ(test, sink.timings[0].streams[0].acquireSpreadNs, 1000000ULL);
    TEST_CHECK_EQ(test, sink.timings[0].streams[0].copyDoneSteadyNs, captureStartSteadyNs + 1400000ULL);
    TEST_CHECK_EQ(test, sink.timings[0].streams[0].processStartSteadyNs, captureStartSteadyNs + 1500000ULL);
    TEST_CHECK(test, sink.timings[0].streams[0].processDoneSteadyNs >= sink.timings[0].streams[0].processStartSteadyNs);
    TEST_CHECK(test, sink.timings[1].streams[1].present);
    TEST_CHECK(test, !sink.timings[1].streams[1].encoded);
    TEST_CHECK_EQ(test, sink.timings[1].samples.size(), static_cast<size_t>(3));
    TEST_CHECK(test, !sink.timings[1].samples[0].present);
    TEST_CHECK(test, !sink.timings[1].samples[1].present);
    TEST_CHECK(test, sink.timings[1].samples[2].present);
    TEST_CHECK_EQ(test, sink.timings[1].samples[2].sensorSofTscNs, 3000ULL);
    TEST_CHECK_EQ(test, sink.timings[1].streams[1].captureStartSteadyNs, rawCaptureStartSteadyNs);
    TEST_CHECK_EQ(test, sink.timings[1].streams[1].copyDoneSteadyNs, rawCaptureStartSteadyNs + 200000ULL);
    TEST_CHECK_EQ(test, sink.timings[1].streams[1].processStartSteadyNs, rawCaptureStartSteadyNs + 300000ULL);

    argus_sender::Avs1Bundle parsed;
    const char* error = 0;
    TEST_CHECK(test, argus_sender::parseAvs1WireRecord(sink.packets[0].data(), sink.packets[0].size(), &parsed, &error));
    TEST_CHECK_EQ(test, parsed.sequence, 1ULL);
    TEST_CHECK(test, parsed.flags & argus_sender::AVS1_BUNDLE_SENDER_TSC_VALID);
    TEST_CHECK_EQ(test, parsed.senderCaptureReadyTscNs, 1300ULL);
    TEST_CHECK(test, parsed.streams[0].flags & argus_sender::AVS1_STREAM_PRESENT);
    TEST_CHECK(test, !(parsed.streams[1].flags & argus_sender::AVS1_STREAM_PRESENT));

    TEST_CHECK(test, argus_sender::parseAvs1WireRecord(sink.packets[1].data(), sink.packets[1].size(), &parsed, &error));
    TEST_CHECK_EQ(test, parsed.sequence, 2ULL);
    TEST_CHECK(test, !(parsed.flags & argus_sender::AVS1_BUNDLE_SENDER_TSC_VALID));
    TEST_CHECK_EQ(test, parsed.senderCaptureReadyTscNs, 0ULL);
    TEST_CHECK(test, parsed.streams[1].flags & argus_sender::AVS1_STREAM_PRESENT);

    // A TSC without a valid sender clock must not set SENDER_TSC_VALID.
    std::vector<argus_sender::BundleStreamInput> bundle3(2);
    bundle3[1].present = true;
    bundle3[1].sequence = 2;
    bundle3[1].timings.push_back(timing(30, 5000));
    bundle3[1].timings[0].acquiredSteadyNs = rawCaptureStartSteadyNs;
    TEST_CHECK(test, assembler.beginBundle(3, bundle3, 6000, 6100, argus_sender::SENDER_CLOCK_UNKNOWN, 6300));
    TEST_CHECK(
        test, assembler.submitRaw(1, 0, 3, std::vector<uint8_t>(12, 0x80), 6200, argus_sender::SENDER_CLOCK_REALTIME));
    TEST_CHECK_EQ(test, sink.packets.size(), static_cast<size_t>(3));
    TEST_CHECK(test, argus_sender::parseAvs1WireRecord(sink.packets[2].data(), sink.packets[2].size(), &parsed, &error));
    TEST_CHECK_EQ(test, parsed.sequence, 3ULL);
    TEST_CHECK(test, !(parsed.flags & argus_sender::AVS1_BUNDLE_SENDER_CLOCK_VALID));
    TEST_CHECK(test, !(parsed.flags & argus_sender::AVS1_BUNDLE_SENDER_TSC_VALID));
    TEST_CHECK_EQ(test, parsed.senderCaptureReadyTscNs, 0ULL);
    return test.finish();
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "EncodedFrameSink.h"
#include "FrameBundleAssembler.h"
#include "PacketSink.h"
#include "TestHarness.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
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

class BlockingPacketSink : public argus_sender::PacketSink
{
public:
    bool enqueueFramed(std::vector<uint8_t>&& packet) override
    {
        argus_sender::Avs1Bundle bundle;
        const char* error = NULL;
        if (!argus_sender::parseAvs1WireRecord(packet.data(), packet.size(), &bundle, &error))
            return false;
        std::unique_lock<std::mutex> lock(m_mutex);
        if (bundle.sequence == 1)
        {
            m_firstEntered = true;
            m_cv.notify_all();
            m_cv.wait(lock, [this] { return m_releaseFirst; });
        }
        sequences.push_back(bundle.sequence);
        m_cv.notify_all();
        return true;
    }

    bool hasError() const override
    {
        return false;
    }

    bool waitForFirst()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_cv.wait_for(lock, std::chrono::seconds(2), [this] { return m_firstEntered; });
    }

    bool waitForOvertake()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_cv.wait_for(lock, std::chrono::milliseconds(100), [this] { return !sequences.empty(); });
    }

    void releaseFirst()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_releaseFirst = true;
        m_cv.notify_all();
    }

    std::vector<uint64_t> sequences;

private:
    std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_firstEntered = false;
    bool m_releaseFirst = false;
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

void checkConcurrentEnqueueOrder(TestContext& test, bool emptyBundles)
{
    BlockingPacketSink sink;
    std::vector<argus_sender::Avs1StreamRecord> streams = templates();
    streams[1].codec = argus_sender::AVS1_CODEC_HEVC;
    argus_sender::FrameBundleAssembler assembler(&sink, 77, streams);
    std::vector<argus_sender::BundleStreamInput> inputs(2);
    const uint8_t payload[] = { 0, 0, 1, 0x26 };
    if (!emptyBundles)
    {
        inputs[0].present = true;
        inputs[0].sequence = 1;
        inputs[0].timings.push_back(timing(10, 1000));
        inputs[0].timings.push_back(timing(11, 1004));
        inputs[1].present = true;
        inputs[1].sequence = 1;
        inputs[1].timings.push_back(timing(20, 1000));
        TEST_CHECK(test, assembler.beginBundle(1, inputs, 2000, 2100, argus_sender::SENDER_CLOCK_REALTIME, 0));
        inputs[0].sequence = 2;
        inputs[1].present = false;
        inputs[1].sequence = 0;
        TEST_CHECK(test, assembler.beginBundle(2, inputs, 3000, 3100, argus_sender::SENDER_CLOCK_REALTIME, 0));
        for (size_t camera = 0; camera < 2; ++camera)
        {
            TEST_CHECK(test, assembler.encodedSink(0, camera)->enqueueEncoded(
                                 payload, sizeof(payload), 1, 2200, argus_sender::SENDER_CLOCK_REALTIME));
        }
    }

    bool firstOk = false;
    bool secondOk = false;
    std::thread first(
        [&]
        {
            firstOk = emptyBundles ?
                          assembler.beginBundle(1, inputs, 2000, 2100, argus_sender::SENDER_CLOCK_REALTIME, 0) :
                          assembler.encodedSink(1, 0)->enqueueEncoded(
                              payload, sizeof(payload), 1, 2200, argus_sender::SENDER_CLOCK_REALTIME);
        });
    TEST_CHECK(test, sink.waitForFirst());
    std::thread second(
        [&]
        {
            if (emptyBundles)
            {
                secondOk = assembler.beginBundle(2, inputs, 3000, 3100, argus_sender::SENDER_CLOCK_REALTIME, 0);
            }
            else
            {
                secondOk = assembler.encodedSink(0, 0)->enqueueEncoded(
                    payload, sizeof(payload), 2, 3200, argus_sender::SENDER_CLOCK_REALTIME);
                secondOk = assembler.encodedSink(0, 1)->enqueueEncoded(
                               payload, sizeof(payload), 2, 3200, argus_sender::SENDER_CLOCK_REALTIME) &&
                           secondOk;
            }
        });
    TEST_CHECK(test, !sink.waitForOvertake());
    sink.releaseFirst();
    first.join();
    second.join();
    TEST_CHECK(test, firstOk);
    TEST_CHECK(test, secondOk);
    TEST_CHECK_EQ(test, sink.sequences.size(), static_cast<size_t>(2));
    if (sink.sequences.size() == 2)
    {
        TEST_CHECK_EQ(test, sink.sequences[0], 1ULL);
        TEST_CHECK_EQ(test, sink.sequences[1], 2ULL);
    }
}

} // namespace

// With the timing extension a bundle carries a record per sample: valid for a frame whose capture time was mapped,
// all zero for one whose was not.
void checkTimingExtension(TestContext& test)
{
    FakePacketSink sink;
    std::vector<argus_sender::Avs1StreamRecord> streams = templates();
    streams[1].codec = argus_sender::AVS1_CODEC_HEVC;
    argus_sender::FrameBundleAssembler assembler(&sink, 88, streams, true);
    std::vector<argus_sender::BundleStreamInput> inputs(2);
    inputs[0].present = true;
    inputs[0].sequence = 1;
    inputs[0].timings.push_back(timing(10, 1000));
    inputs[0].timings.push_back(timing(11, 1004));
    inputs[0].timings[0].hasCaptureMonotonic = true;
    inputs[0].timings[0].captureMonotonicNs = 123456789;
    inputs[0].timings[0].captureUncertaintyNs = 2500;
    inputs[0].timings[0].captureAnchorAgeNs = 400000000;
    // A second sample whose uncertainty does not fit the 32-bit field makes no capture claim.
    inputs[0].timings[1].hasCaptureMonotonic = true;
    inputs[0].timings[1].captureMonotonicNs = 123456790;
    inputs[0].timings[1].captureUncertaintyNs = 5000000000LL;
    TEST_CHECK(test, assembler.beginBundle(1, inputs, 2000, 2100, argus_sender::SENDER_CLOCK_REALTIME, 0));
    const uint8_t payload[] = { 0, 0, 1, 0x26 };
    for (size_t camera = 0; camera < 2; ++camera)
    {
        TEST_CHECK(test, assembler.encodedSink(0, camera)->enqueueEncoded(
                             payload, sizeof(payload), 1, 2200, argus_sender::SENDER_CLOCK_REALTIME));
    }
    TEST_CHECK_EQ(test, sink.packets.size(), static_cast<size_t>(1));
    if (sink.packets.size() != 1)
        return;
    argus_sender::Avs1Bundle bundle;
    const char* error = NULL;
    TEST_CHECK(test, argus_sender::parseAvs1WireRecord(sink.packets[0].data(), sink.packets[0].size(), &bundle, &error));
    TEST_CHECK(test, (bundle.flags & argus_sender::AVS1_BUNDLE_TIMING_EXT) != 0);
    const argus_sender::Avs1TimingRecord& mapped = bundle.streams[0].samples[0].timing;
    TEST_CHECK_EQ(test, mapped.flags, static_cast<uint16_t>(argus_sender::AVS1_TIMING_VALID));
    TEST_CHECK_EQ(test, mapped.sensorCaptureNs, static_cast<uint64_t>(123456789));
    TEST_CHECK_EQ(test, mapped.uncertaintyNs, static_cast<uint32_t>(2500));
    TEST_CHECK_EQ(test, mapped.anchorAgeNs, static_cast<uint32_t>(400000000));
    TEST_CHECK_EQ(test, bundle.streams[0].samples[1].timing.flags, static_cast<uint16_t>(0));
    TEST_CHECK_EQ(test, bundle.streams[0].samples[1].timing.sensorCaptureNs, static_cast<uint64_t>(0));
    // The absent stream carries an all-zero record.
    TEST_CHECK_EQ(test, bundle.streams[1].samples[0].timing.flags, static_cast<uint16_t>(0));
}

// Emits one two-camera stereo bundle with the timing extension and the given clock; the first sample carries
// monotonic and realtime capture times, `realtimeNs` for the latter.
bool emitTimed(bool setClock, argus_sender::Avs1TimingClockDomain clock, int64_t realtimeNs, argus_sender::Avs1Bundle* out)
{
    FakePacketSink sink;
    std::vector<argus_sender::Avs1StreamRecord> streams = templates();
    streams[1].codec = argus_sender::AVS1_CODEC_HEVC;
    argus_sender::FrameBundleAssembler assembler =
        setClock ? argus_sender::FrameBundleAssembler(&sink, 89, streams, true, clock) :
                   argus_sender::FrameBundleAssembler(&sink, 89, streams, true);
    std::vector<argus_sender::BundleStreamInput> inputs(2);
    inputs[0].present = true;
    inputs[0].sequence = 1;
    inputs[0].timings.push_back(timing(10, 1000));
    inputs[0].timings.push_back(timing(11, 1004));
    inputs[0].timings[0].hasCaptureMonotonic = true;
    inputs[0].timings[0].captureMonotonicNs = 123456789;
    inputs[0].timings[0].captureRealtimeNs = realtimeNs;
    inputs[0].timings[0].captureUncertaintyNs = 2500;
    inputs[0].timings[0].captureAnchorAgeNs = 400000000;
    if (!assembler.beginBundle(1, inputs, 2000, 2100, argus_sender::SENDER_CLOCK_REALTIME, 0))
        return false;
    const uint8_t payload[] = { 0, 0, 1, 0x26 };
    for (size_t camera = 0; camera < 2; ++camera)
    {
        if (!assembler.encodedSink(0, camera)->enqueueEncoded(
                payload, sizeof(payload), 1, 2200, argus_sender::SENDER_CLOCK_REALTIME))
            return false;
    }
    const char* error = NULL;
    return sink.packets.size() == 1 &&
           argus_sender::parseAvs1WireRecord(sink.packets[0].data(), sink.packets[0].size(), out, &error);
}

void checkTimingClock(TestContext& test)
{
    const int64_t realtimeNs = 1762900000123456789LL;
    argus_sender::Avs1Bundle bundle;

    TEST_CHECK(test, emitTimed(true, argus_sender::AVS1_TIMING_CLOCK_REALTIME, realtimeNs, &bundle));
    argus_sender::Avs1TimingRecord record = bundle.streams[0].samples[0].timing;
    TEST_CHECK_EQ(test, record.flags, static_cast<uint16_t>(argus_sender::AVS1_TIMING_VALID));
    TEST_CHECK_EQ(test, record.clockDomain, static_cast<uint16_t>(argus_sender::AVS1_TIMING_CLOCK_REALTIME));
    TEST_CHECK_EQ(test, record.sensorCaptureNs, static_cast<uint64_t>(realtimeNs));
    TEST_CHECK_EQ(test, record.uncertaintyNs, static_cast<uint32_t>(2500));

    // Realtime requested but no realtime time: no claim, even though a monotonic time exists.
    TEST_CHECK(test, emitTimed(true, argus_sender::AVS1_TIMING_CLOCK_REALTIME, 0, &bundle));
    record = bundle.streams[0].samples[0].timing;
    TEST_CHECK_EQ(test, record.flags, static_cast<uint16_t>(0));
    TEST_CHECK_EQ(test, record.clockDomain, static_cast<uint16_t>(0));
    TEST_CHECK_EQ(test, record.sensorCaptureNs, static_cast<uint64_t>(0));

    // Default and explicit monotonic emit domain 1 with the monotonic time, whatever the realtime value.
    TEST_CHECK(test, emitTimed(false, argus_sender::AVS1_TIMING_CLOCK_MONOTONIC, realtimeNs, &bundle));
    record = bundle.streams[0].samples[0].timing;
    TEST_CHECK_EQ(test, record.flags, static_cast<uint16_t>(argus_sender::AVS1_TIMING_VALID));
    TEST_CHECK_EQ(test, record.clockDomain, static_cast<uint16_t>(argus_sender::AVS1_TIMING_CLOCK_MONOTONIC));
    TEST_CHECK_EQ(test, record.sensorCaptureNs, static_cast<uint64_t>(123456789));
    TEST_CHECK(test, emitTimed(true, argus_sender::AVS1_TIMING_CLOCK_MONOTONIC, 0, &bundle));
    record = bundle.streams[0].samples[0].timing;
    TEST_CHECK_EQ(test, record.clockDomain, static_cast<uint16_t>(argus_sender::AVS1_TIMING_CLOCK_MONOTONIC));
    TEST_CHECK_EQ(test, record.sensorCaptureNs, static_cast<uint64_t>(123456789));
}

int main()
{
    TestContext test("bundle_assembler");
    checkTimingExtension(test);
    checkTimingClock(test);
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
    checkConcurrentEnqueueOrder(test, false);
    checkConcurrentEnqueueOrder(test, true);
    return test.finish();
}

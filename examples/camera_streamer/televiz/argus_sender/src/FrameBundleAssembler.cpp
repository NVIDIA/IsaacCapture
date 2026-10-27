/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "FrameBundleAssembler.h"

#include "EncodedFrameSink.h"
#include "PacketSink.h"
#include "Runtime.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <map>
#include <mutex>
#include <utility>

namespace argus_sender
{

namespace
{

uint64_t steadyNowNs()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

} // namespace

class FrameBundleAssembler::Impl
{
public:
    class SampleSink : public EncodedFrameSink
    {
    public:
        SampleSink(Impl* parent, size_t streamIndex, size_t cameraIndex)
            : m_parent(parent), m_streamIndex(streamIndex), m_cameraIndex(cameraIndex)
        {
        }

        bool enqueueEncoded(const uint8_t* data,
                            size_t size,
                            uint64_t frameSeq,
                            uint64_t encodeDoneNs,
                            uint32_t senderClockDomain) override
        {
            return m_parent && m_parent->submitEncoded(
                                   m_streamIndex, m_cameraIndex, frameSeq, data, size, encodeDoneNs, senderClockDomain);
        }

        bool hasError() const override
        {
            return !m_parent || m_parent->hasError();
        }

    private:
        Impl* m_parent;
        size_t m_streamIndex;
        size_t m_cameraIndex;
    };

    Impl(PacketSink* packetSink, uint64_t sessionId, const std::vector<Avs1StreamRecord>& streamTemplates)
        : m_packetSink(packetSink),
          m_sessionId(sessionId),
          m_streamTemplates(streamTemplates),
          m_nextSendSequence(1),
          m_error(false)
    {
        m_sinks.resize(streamTemplates.size());
        for (size_t stream = 0; stream < streamTemplates.size(); ++stream)
        {
            for (size_t camera = 0; camera < streamTemplates[stream].samples.size(); ++camera)
            {
                m_sinks[stream].push_back(std::unique_ptr<SampleSink>(new SampleSink(this, stream, camera)));
            }
        }
    }

    EncodedFrameSink* encodedSink(size_t streamIndex, size_t cameraIndex)
    {
        if (streamIndex >= m_sinks.size() || cameraIndex >= m_sinks[streamIndex].size())
        {
            return NULL;
        }
        return m_sinks[streamIndex][cameraIndex].get();
    }

    bool beginBundle(uint64_t bundleSequence,
                     const std::vector<BundleStreamInput>& inputs,
                     uint64_t senderCaptureReadyNs,
                     uint64_t senderProcessSubmitNs,
                     uint32_t senderClockDomain,
                     uint64_t senderCaptureReadyTscNs)
    {
        if (bundleSequence == 0 || inputs.size() != m_streamTemplates.size())
            return setError("invalid AVS1 bundle registration");

        PendingBundle pending;
        pending.captureReadySteadyNs = steadyNowNs();
        pending.timing.captureStartSteadyNs = pending.captureReadySteadyNs;
        pending.timing.streams.resize(inputs.size());
        uint64_t earliestAcquiredSteadyNs = 0;
        uint64_t latestAcquiredSteadyNs = 0;
        pending.bundle.sequence = bundleSequence;
        pending.bundle.sessionId = m_sessionId;
        pending.bundle.senderCaptureReadyNs = senderCaptureReadyNs;
        if (senderClockDomain != SENDER_CLOCK_UNKNOWN && senderCaptureReadyNs != 0 && senderProcessSubmitNs != 0)
        {
            pending.bundle.flags |= AVS1_BUNDLE_SENDER_CLOCK_VALID;
            if (senderClockDomain == SENDER_CLOCK_TAI)
                pending.bundle.flags |= AVS1_BUNDLE_SENDER_CLOCK_TAI;
            pending.senderClockDomain = senderClockDomain;
            if (senderCaptureReadyTscNs != 0)
            {
                pending.bundle.flags |= AVS1_BUNDLE_SENDER_TSC_VALID;
                pending.bundle.senderCaptureReadyTscNs = senderCaptureReadyTscNs;
            }
        }
        pending.bundle.streams = m_streamTemplates;
        for (size_t streamIndex = 0; streamIndex < inputs.size(); ++streamIndex)
        {
            Avs1StreamRecord& stream = pending.bundle.streams[streamIndex];
            const BundleStreamInput& input = inputs[streamIndex];
            PacketStreamTiming& streamTiming = pending.timing.streams[streamIndex];
            streamTiming.present = input.present;
            streamTiming.encoded = stream.codec == AVS1_CODEC_HEVC;
            uint64_t streamEarliestAcquiredSteadyNs = 0;
            uint64_t streamLatestAcquiredSteadyNs = 0;
            stream.flags &= ~AVS1_STREAM_PRESENT;
            stream.sequence = input.present ? input.sequence : 0;
            if (input.present)
                stream.flags |= AVS1_STREAM_PRESENT;
            if (input.present && input.timings.size() != stream.samples.size())
                return setError("AVS1 present stream timing count mismatch");
            for (size_t cameraIndex = 0; cameraIndex < stream.samples.size(); ++cameraIndex)
            {
                PacketSampleTiming sampleTiming;
                Avs1SampleRecord& sample = stream.samples[cameraIndex];
                sample.flags = 0;
                sample.payload.clear();
                sample.frameNumber = 0;
                sample.sofTscNs = 0;
                sample.frameTimestampNs = 0;
                sample.senderCaptureReadyNs = 0;
                sample.senderProcessSubmitNs = 0;
                sample.senderProcessDoneNs = 0;
                if (!input.present)
                {
                    pending.timing.samples.push_back(sampleTiming);
                    continue;
                }
                const FrameTiming& timing = input.timings[cameraIndex];
                if (timing.acquiredSteadyNs != 0)
                {
                    if (earliestAcquiredSteadyNs == 0 || timing.acquiredSteadyNs < earliestAcquiredSteadyNs)
                    {
                        earliestAcquiredSteadyNs = timing.acquiredSteadyNs;
                    }
                    if (timing.acquiredSteadyNs > latestAcquiredSteadyNs)
                        latestAcquiredSteadyNs = timing.acquiredSteadyNs;
                    if (streamEarliestAcquiredSteadyNs == 0 || timing.acquiredSteadyNs < streamEarliestAcquiredSteadyNs)
                    {
                        streamEarliestAcquiredSteadyNs = timing.acquiredSteadyNs;
                    }
                    if (timing.acquiredSteadyNs > streamLatestAcquiredSteadyNs)
                    {
                        streamLatestAcquiredSteadyNs = timing.acquiredSteadyNs;
                    }
                }
                sample.flags |= AVS1_SAMPLE_PRESENT;
                sample.frameNumber = timing.frameNumber;
                if (timing.hasSensorSofTimestampTsc)
                {
                    sample.flags |= AVS1_SAMPLE_SOF_TSC_VALID;
                    sample.sofTscNs = timing.sensorSofTimestampTscNs;
                    sampleTiming.present = true;
                    sampleTiming.sensorSofTscNs = timing.sensorSofTimestampTscNs;
                }
                pending.timing.samples.push_back(sampleTiming);
                if (timing.frameTimeNs != 0)
                {
                    sample.flags |= AVS1_SAMPLE_FRAME_TIMESTAMP_VALID;
                    sample.frameTimestampNs = timing.frameTimeNs;
                }
                if (pending.bundle.flags & AVS1_BUNDLE_SENDER_CLOCK_VALID)
                {
                    sample.flags |= AVS1_SAMPLE_SENDER_TIMING_VALID;
                    sample.senderCaptureReadyNs = senderCaptureReadyNs;
                    sample.senderProcessSubmitNs = senderProcessSubmitNs;
                }
                ++pending.expectedPayloads;
            }
            if (input.present)
            {
                streamTiming.captureStartSteadyNs =
                    streamEarliestAcquiredSteadyNs != 0 ? streamEarliestAcquiredSteadyNs : pending.captureReadySteadyNs;
                if (streamEarliestAcquiredSteadyNs != 0)
                {
                    streamTiming.acquireSpreadNs = streamLatestAcquiredSteadyNs - streamEarliestAcquiredSteadyNs;
                }
            }
        }
        if (earliestAcquiredSteadyNs != 0)
        {
            pending.timing.captureStartSteadyNs = earliestAcquiredSteadyNs;
            pending.timing.acquireSpreadNs = latestAcquiredSteadyNs - earliestAcquiredSteadyNs;
        }

        std::vector<ReadyBundle> ready;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_pending.count(bundleSequence) != 0)
                return setError("duplicate AVS1 bundle registration");
            m_pending[bundleSequence] = std::move(pending);
            collectReadyLocked(&ready);
        }
        return sendReady(&ready);
    }

    bool recordCopyDone(size_t streamIndex, uint64_t bundleSequence, uint64_t copyDoneSteadyNs)
    {
        if (bundleSequence == 0 || copyDoneSteadyNs == 0)
            return setError("invalid AVS1 copy-done timestamp");
        std::lock_guard<std::mutex> lock(m_mutex);
        std::map<uint64_t, PendingBundle>::iterator it = m_pending.find(bundleSequence);
        if (it == m_pending.end())
            return setError("AVS1 copy done has no registered bundle");
        if (streamIndex >= it->second.timing.streams.size())
            return setError("AVS1 copy done stream is out of range");
        PacketStreamTiming& stream = it->second.timing.streams[streamIndex];
        if (!stream.present || copyDoneSteadyNs < stream.captureStartSteadyNs)
        {
            return setError("invalid AVS1 per-stream copy-done timestamp");
        }
        stream.copyDoneSteadyNs = std::max(stream.copyDoneSteadyNs, copyDoneSteadyNs);
        return true;
    }

    bool recordProcessStart(size_t streamIndex, uint64_t bundleSequence, uint64_t processStartSteadyNs)
    {
        if (bundleSequence == 0 || processStartSteadyNs == 0)
            return setError("invalid AVS1 process-start timestamp");
        std::lock_guard<std::mutex> lock(m_mutex);
        std::map<uint64_t, PendingBundle>::iterator it = m_pending.find(bundleSequence);
        if (it == m_pending.end())
            return setError("AVS1 process start has no registered bundle");
        PacketTiming& timing = it->second.timing;
        if (streamIndex >= timing.streams.size())
            return setError("AVS1 process start stream is out of range");
        PacketStreamTiming& stream = timing.streams[streamIndex];
        if (!stream.present || processStartSteadyNs < stream.captureStartSteadyNs ||
            processStartSteadyNs < timing.captureStartSteadyNs)
        {
            return setError("invalid AVS1 per-stream process-start timestamp");
        }
        if (stream.processStartSteadyNs == 0 || processStartSteadyNs < stream.processStartSteadyNs)
        {
            stream.processStartSteadyNs = processStartSteadyNs;
        }
        const uint64_t latencyNs = processStartSteadyNs - timing.captureStartSteadyNs;
        if (timing.captureToEncodeStartNs == 0 || latencyNs < timing.captureToEncodeStartNs)
        {
            timing.captureToEncodeStartNs = latencyNs;
        }
        return true;
    }

    bool submitRaw(size_t streamIndex,
                   size_t cameraIndex,
                   uint64_t bundleSequence,
                   std::vector<uint8_t>&& payload,
                   uint64_t processDoneNs,
                   uint32_t senderClockDomain)
    {
        return submitOwned(
            streamIndex, cameraIndex, bundleSequence, std::move(payload), processDoneNs, senderClockDomain);
    }

    bool submitEncoded(size_t streamIndex,
                       size_t cameraIndex,
                       uint64_t bundleSequence,
                       const uint8_t* data,
                       size_t size,
                       uint64_t processDoneNs,
                       uint32_t senderClockDomain)
    {
        if (!data || size == 0)
            return setError("empty AVS1 encoded payload");
        std::vector<uint8_t> owned(data, data + size);
        return submitOwned(streamIndex, cameraIndex, bundleSequence, std::move(owned), processDoneNs, senderClockDomain);
    }

    bool hasError() const
    {
        return m_error.load() || !m_packetSink || m_packetSink->hasError();
    }

private:
    struct ReadyBundle
    {
        Avs1Bundle bundle;
        PacketTiming timing;
    };

    struct PendingBundle
    {
        Avs1Bundle bundle;
        PacketTiming timing;
        uint64_t captureReadySteadyNs = 0;
        uint32_t senderClockDomain = SENDER_CLOCK_UNKNOWN;
        size_t expectedPayloads = 0;
        size_t receivedPayloads = 0;
    };

    bool submitOwned(size_t streamIndex,
                     size_t cameraIndex,
                     uint64_t bundleSequence,
                     std::vector<uint8_t>&& payload,
                     uint64_t processDoneNs,
                     uint32_t senderClockDomain)
    {
        if (payload.empty())
            return setError("empty AVS1 sample payload");
        const uint64_t payloadDoneSteadyNs = steadyNowNs();
        std::vector<ReadyBundle> ready;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            std::map<uint64_t, PendingBundle>::iterator it = m_pending.find(bundleSequence);
            if (it == m_pending.end())
                return setError("AVS1 payload has no registered bundle");
            PendingBundle& pending = it->second;
            if (streamIndex >= pending.bundle.streams.size() ||
                cameraIndex >= pending.bundle.streams[streamIndex].samples.size())
            {
                return setError("AVS1 payload index is out of range");
            }
            if (streamIndex >= pending.timing.streams.size())
                return setError("AVS1 payload timing stream is out of range");
            PacketStreamTiming& streamTiming = pending.timing.streams[streamIndex];
            if (!streamTiming.present || payloadDoneSteadyNs < streamTiming.captureStartSteadyNs)
            {
                return setError("invalid AVS1 per-stream payload-ready timestamp");
            }
            streamTiming.processDoneSteadyNs = std::max(streamTiming.processDoneSteadyNs, payloadDoneSteadyNs);
            Avs1SampleRecord& sample = pending.bundle.streams[streamIndex].samples[cameraIndex];
            if (!(sample.flags & AVS1_SAMPLE_PRESENT) || !sample.payload.empty())
                return setError("duplicate or unexpected AVS1 sample payload");
            sample.payload = std::move(payload);
            sample.senderProcessDoneNs = processDoneNs;
            uint64_t latencyNs = 0;
            if (senderClockDomain == pending.senderClockDomain && pending.bundle.senderCaptureReadyNs != 0 &&
                processDoneNs >= pending.bundle.senderCaptureReadyNs &&
                pending.captureReadySteadyNs >= pending.timing.captureStartSteadyNs)
            {
                latencyNs = pending.captureReadySteadyNs - pending.timing.captureStartSteadyNs + processDoneNs -
                            pending.bundle.senderCaptureReadyNs;
            }
            else if (payloadDoneSteadyNs >= pending.timing.captureStartSteadyNs)
            {
                latencyNs = payloadDoneSteadyNs - pending.timing.captureStartSteadyNs;
            }
            if (latencyNs > pending.timing.captureToProcessDoneNs)
                pending.timing.captureToProcessDoneNs = latencyNs;
            if (senderClockDomain != pending.senderClockDomain || processDoneNs == 0)
            {
                pending.bundle.flags &= ~AVS1_BUNDLE_SENDER_CLOCK_VALID;
                pending.bundle.flags &= ~AVS1_BUNDLE_SENDER_CLOCK_TAI;
                pending.bundle.flags &= ~AVS1_BUNDLE_SENDER_TSC_VALID;
                pending.bundle.senderCaptureReadyTscNs = 0;
                for (size_t s = 0; s < pending.bundle.streams.size(); ++s)
                {
                    for (size_t c = 0; c < pending.bundle.streams[s].samples.size(); ++c)
                    {
                        pending.bundle.streams[s].samples[c].flags &= ~AVS1_SAMPLE_SENDER_TIMING_VALID;
                    }
                }
            }
            ++pending.receivedPayloads;
            collectReadyLocked(&ready);
        }
        return sendReady(&ready);
    }

    void collectReadyLocked(std::vector<ReadyBundle>* ready)
    {
        while (true)
        {
            std::map<uint64_t, PendingBundle>::iterator it = m_pending.find(m_nextSendSequence);
            if (it == m_pending.end() || it->second.receivedPayloads != it->second.expectedPayloads)
            {
                break;
            }
            ReadyBundle completed;
            completed.bundle = std::move(it->second.bundle);
            completed.timing = it->second.timing;
            ready->push_back(std::move(completed));
            m_pending.erase(it);
            ++m_nextSendSequence;
        }
    }

    bool sendReady(std::vector<ReadyBundle>* ready)
    {
        for (size_t i = 0; i < ready->size(); ++i)
        {
            uint32_t clockDomain = SENDER_CLOCK_UNKNOWN;
            (*ready)[i].bundle.senderTcpEnqueueNs = readSenderClockNs(&clockDomain);
            if (clockDomain == SENDER_CLOCK_UNKNOWN || (*ready)[i].bundle.senderTcpEnqueueNs == 0)
            {
                (*ready)[i].bundle.flags &= ~AVS1_BUNDLE_SENDER_CLOCK_VALID;
                (*ready)[i].bundle.flags &= ~AVS1_BUNDLE_SENDER_CLOCK_TAI;
                (*ready)[i].bundle.flags &= ~AVS1_BUNDLE_SENDER_TSC_VALID;
                (*ready)[i].bundle.senderCaptureReadyTscNs = 0;
            }
            std::vector<uint8_t> packet;
            const char* wireError = NULL;
            if (!buildAvs1WireRecord((*ready)[i].bundle, &packet, &wireError) || !m_packetSink ||
                !m_packetSink->enqueueFramedWithTiming(std::move(packet), (*ready)[i].timing))
            {
                if (!stopRequested())
                {
                    std::fprintf(stderr, "Failed to emit AVS1 bundle: %s\n", wireError ? wireError : "packet sink error");
                    m_error = true;
                }
                return false;
            }
        }
        return true;
    }

    bool setError(const char* message)
    {
        std::fprintf(stderr, "Frame bundle assembler: %s\n", message);
        m_error = true;
        return false;
    }

    PacketSink* m_packetSink;
    uint64_t m_sessionId;
    std::vector<Avs1StreamRecord> m_streamTemplates;
    std::vector<std::vector<std::unique_ptr<SampleSink>>> m_sinks;
    mutable std::mutex m_mutex;
    std::map<uint64_t, PendingBundle> m_pending;
    uint64_t m_nextSendSequence;
    std::atomic<bool> m_error;
};

FrameBundleAssembler::FrameBundleAssembler(PacketSink* packetSink,
                                           uint64_t sessionId,
                                           const std::vector<Avs1StreamRecord>& streamTemplates)
    : m_impl(new Impl(packetSink, sessionId, streamTemplates))
{
}

FrameBundleAssembler::~FrameBundleAssembler()
{
}

EncodedFrameSink* FrameBundleAssembler::encodedSink(size_t streamIndex, size_t cameraIndex)
{
    return m_impl->encodedSink(streamIndex, cameraIndex);
}

bool FrameBundleAssembler::beginBundle(uint64_t bundleSequence,
                                       const std::vector<BundleStreamInput>& streams,
                                       uint64_t senderCaptureReadyNs,
                                       uint64_t senderProcessSubmitNs,
                                       uint32_t senderClockDomain,
                                       uint64_t senderCaptureReadyTscNs)
{
    return m_impl->beginBundle(bundleSequence, streams, senderCaptureReadyNs, senderProcessSubmitNs, senderClockDomain,
                               senderCaptureReadyTscNs);
}

bool FrameBundleAssembler::recordCopyDone(size_t streamIndex, uint64_t bundleSequence, uint64_t copyDoneSteadyNs)
{
    return m_impl->recordCopyDone(streamIndex, bundleSequence, copyDoneSteadyNs);
}

bool FrameBundleAssembler::recordProcessStart(size_t streamIndex, uint64_t bundleSequence, uint64_t processStartSteadyNs)
{
    return m_impl->recordProcessStart(streamIndex, bundleSequence, processStartSteadyNs);
}

bool FrameBundleAssembler::submitRaw(size_t streamIndex,
                                     size_t cameraIndex,
                                     uint64_t bundleSequence,
                                     std::vector<uint8_t>&& payload,
                                     uint64_t processDoneNs,
                                     uint32_t senderClockDomain)
{
    return m_impl->submitRaw(
        streamIndex, cameraIndex, bundleSequence, std::move(payload), processDoneNs, senderClockDomain);
}

bool FrameBundleAssembler::hasError() const
{
    return m_impl->hasError();
}

} // namespace argus_sender

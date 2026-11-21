/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "CaptureSender.h"

#include "Avs1Wire.h"
#include "CameraCapture.h"
#include "CaptureFrame.h"
#include "Config.h"
#include "EncoderInputSlots.h"
#include "FrameBundleAssembler.h"
#include "FrameTiming.h"
#include "HevcEncoder.h"
#include "OutputRateSelector.h"
#include "Runtime.h"
#include "TcpTransport.h"

#include <linux/videodev2.h>

#include <NvVideoEncoder.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

namespace argus_sender
{
namespace
{

const uint32_t kTcpQueueDepth = 1;

uint64_t steadyNowNs()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct Stream
{
    std::string label;
    CaptureSenderStream source;
    uint64_t sequence = 0;
    uint64_t dropped = 0;
    OutputRateSelector rate{ 0 };
    std::unique_ptr<DirectHevcEncoder> encoder;
    EncoderInputSlots<DmaBuffer> slots{ kEncoderOutputPlaneBuffers };
};

void initializeV4l2Buffer(uint32_t index, struct v4l2_buffer* buffer, struct v4l2_plane planes[MAX_PLANES])
{
    std::memset(buffer, 0, sizeof(*buffer));
    std::memset(planes, 0, sizeof(struct v4l2_plane) * MAX_PLANES);
    buffer->index = index;
    buffer->m.planes = planes;
}

// Returns the buffer of one input the encoder has finished reading. With `wait` it blocks until one finishes;
// without, it asks once (the encoder's own dequeue timeout bounds the delay) and returns false if none is ready.
bool releaseOne(Stream* stream, bool wait)
{
    if (stream->slots.inFlight() == 0)
        return false;
    struct v4l2_buffer completed;
    struct v4l2_plane planes[MAX_PLANES];
    initializeV4l2Buffer(0, &completed, planes);
    DmaBuffer* buffer = NULL;
    if (wait ? !waitForEncoderInput(stream->encoder.get(), &completed, &buffer) :
               !stream->encoder->dequeueInput(completed, &buffer))
        return false;
    return stream->slots.complete(completed.index);
}

} // namespace

uint64_t createCaptureSessionId()
{
    return (steadyNowNs() << 16) ^ static_cast<uint64_t>(getpid()) ^ 0x4156533100000000ULL;
}

bool runCaptureSender(const AppConfig& config,
                      std::unique_ptr<CameraCapture> capture,
                      const std::vector<CaptureSenderStream>& sources,
                      uint64_t sessionId,
                      const char* backendName,
                      bool* receiverDisconnected,
                      bool appendDump)
{
    if (receiverDisconnected)
        *receiverDisconnected = false;
    if (!capture || sources.size() != config.streams.size())
        return false;

    std::vector<Stream> streams(config.streams.size());
    std::vector<Avs1StreamRecord> templates(config.streams.size());
    for (size_t i = 0; i < config.streams.size(); ++i)
    {
        const StreamConfig& configStream = config.streams[i];
        streams[i].label = configStream.id;
        streams[i].source = sources[i];
        streams[i].rate = OutputRateSelector(sources[i].fps);

        Avs1StreamRecord& record = templates[i];
        record.codec = AVS1_CODEC_HEVC;
        record.pixelFormat = AVS1_PIXEL_FORMAT_NV12;
        record.rotation = static_cast<uint16_t>(configStream.output.rotation);
        record.width = sources[i].width;
        record.height = sources[i].height;
        record.fpsNumerator = sources[i].fps;
        record.fpsDenominator = 1;
        record.samples.resize(1);
        record.samples[0].cameraIndex = 0;
        record.samples[0].role = AVS1_ROLE_MONO;
        record.samples[0].deviceIndex = sources[i].cameraId;
        record.samples[0].sensorModeIndex = 0;
    }

    std::printf("Listening for AVS1 receiver on %s:%u\n", config.network.bindHost.c_str(), config.network.port);
    int fd = acceptTcpClient(config.network.bindHost, config.network.port, config.network.reconnectDelayMs);
    if (fd < 0)
        return false;

    TcpPacketSenderSettings transportSettings;
    transportSettings.port = config.network.port;
    transportSettings.dumpWirePath = config.network.dumpWirePath;
    transportSettings.sendBufferBytes = config.network.sendBufferKb * 1024U;
    for (size_t i = 0; i < config.streams.size(); ++i)
    {
        transportSettings.streamLabels.push_back(config.streams[i].id);
        transportSettings.sampleLabels.push_back(config.streams[i].id);
    }
    transportSettings.kpi = config.runtime.kpi;
    transportSettings.logIntervalSeconds = config.runtime.logIntervalSeconds;
    TcpPacketSender tcpSender(&fd, transportSettings, kTcpQueueDepth, config.network.port, "avs1", appendDump);
    if (!tcpSender.start())
    {
        close(fd);
        return false;
    }

    FrameBundleAssembler assembler(&tcpSender, sessionId, templates);
    HevcEncoderSettings encoderSettings;
    encoderSettings.kpi = config.runtime.kpi;
    encoderSettings.logIntervalSeconds = config.runtime.logIntervalSeconds;
    encoderSettings.benchEncode = config.runtime.benchEncode;
    bool ok = true;
    for (size_t i = 0; ok && i < streams.size(); ++i)
    {
        Stream& stream = streams[i];
        encoderSettings.bitrateMbps = config.streams[i].output.bitrateMbps;
        stream.encoder.reset(new DirectHevcEncoder(assembler.encodedSink(i, 0), encoderSettings));
        const uint32_t gop = config.streams[i].output.gop ? config.streams[i].output.gop : stream.source.fps;
        if (!stream.encoder->initialize(stream.source.width, stream.source.height, stream.source.fps, gop))
            ok = false;
    }

    uint64_t bundleSequence = 0;
    try
    {
        if (ok)
            capture->start();
        uint64_t frames = 0;
        while (ok && !stopRequested() && !tcpSender.hasError() && !assembler.hasError() &&
               (config.runtime.frames == 0 || frames < config.runtime.frames))
        {
            if (!capture->waitForFrame(100))
            {
                // With few capture buffers all of them can sit in encoder slots the encoder has already finished with;
                // the backend then has nothing to fill and no frame ever arrives, so give those buffers back here.
                for (Stream& stream : streams)
                {
                    if (stream.encoder && stream.slots.inFlight() > 0)
                        releaseOne(&stream, false);
                }
                continue;
            }
            for (size_t i = 0; ok && i < streams.size(); ++i)
            {
                Stream& stream = streams[i];
                if (stream.encoder->hasError())
                {
                    ok = false;
                    break;
                }
                CaptureFrame frame;
                if (!capture->acquireLatest(stream.source.cameraId, &frame))
                    continue;
                // The stream carries its configured rate, not the sensor's; a skipped frame's lease returns it to the
                // backend on leaving this scope.
                if (stream.rate.shouldSkip(frame.readyMonotonicNs))
                    continue;
                // A frame arriving with no free encoder input is dropped, not waited for: the newest frame is the
                // one worth sending and the encoder is already a frame behind.
                if (!stream.slots.hasFree())
                    releaseOne(&stream, false);
                if (!stream.slots.hasFree())
                {
                    ++stream.dropped;
                    continue;
                }
                const uint32_t index = stream.slots.nextFree();
                std::unique_ptr<DmaBuffer> wrapper(DmaBuffer::createExternal(frame.surface.fd));
                if (!wrapper)
                {
                    ok = false;
                    break;
                }

                stream.rate.sent(frame.readyMonotonicNs);
                ++stream.sequence;
                ++bundleSequence;
                std::vector<BundleStreamInput> inputs(streams.size());
                inputs[i].present = true;
                inputs[i].sequence = stream.sequence;
                FrameTiming timing;
                timing.frameNumber = frame.sequence;
                timing.acquiredSteadyNs = frame.readyMonotonicNs;
                // Only a TSC in nanoseconds may go into the AVS1 TSC fields, and only a frame whose capture time
                // passed the backend's mapping check carries one.
                if (frame.sensorClock == SENSOR_CLOCK_TSC_NS)
                {
                    timing.hasSensorSofTimestampTsc = true;
                    timing.sensorSofTimestampTscNs = frame.sensorTime;
                }
                if (frame.captureMapped)
                {
                    timing.hasCaptureMonotonic = true;
                    timing.captureMonotonicNs = frame.captureMonotonicNs;
                    timing.captureUncertaintyNs = frame.captureUncertaintyNs;
                    timing.captureAnchorAgeNs = frame.captureAnchorAgeNs;
                    timing.captureRealtimeNs = frame.captureRealtimeNs;
                }
                inputs[i].timings.push_back(timing);
                uint32_t readyDomain = SENDER_CLOCK_UNKNOWN;
                const uint64_t readyNs = readSenderClockNs(&readyDomain);
                // The TSC at the same instant as the wall clock reading, so a receiver can place the sensor's
                // start-of-frame on the wall clock. Left at 0 (the bundle's TSC anchor is then not marked valid) unless
                // it is after the start-of-frame and within a plausible lag.
                const uint64_t tscNowNs = readTscNowNs();
                const uint64_t kMaxSofLagNs = 500000000ULL;
                const uint64_t readyTscNs =
                    (timing.hasSensorSofTimestampTsc && tscNowNs > timing.sensorSofTimestampTscNs &&
                     tscNowNs - timing.sensorSofTimestampTscNs < kMaxSofLagNs) ?
                        tscNowNs :
                        0;
                uint32_t submitDomain = SENDER_CLOCK_UNKNOWN;
                const uint64_t submitNs = readSenderClockNs(&submitDomain);
                if (readyDomain != submitDomain)
                    readyDomain = SENDER_CLOCK_UNKNOWN;
                if (!assembler.beginBundle(bundleSequence, inputs, readyNs, submitNs, readyDomain, readyTscNs) ||
                    !assembler.recordCopyDone(i, bundleSequence, frame.readyMonotonicNs) ||
                    !assembler.recordProcessStart(i, bundleSequence, steadyNowNs()))
                {
                    ok = false;
                    break;
                }

                struct v4l2_buffer v4l2Buffer;
                struct v4l2_plane planes[MAX_PLANES];
                initializeV4l2Buffer(index, &v4l2Buffer, planes);
                if (!stream.encoder->queueInput(v4l2Buffer, wrapper.get(), bundleSequence))
                {
                    ok = false;
                    break;
                }
                stream.slots.occupy(std::move(wrapper), std::move(frame));
                ++frames;
            }
        }
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "%s capture failed: %s\n", backendName, e.what());
        ok = false;
    }

    // Finish the encoder before any captured buffer goes back: it may still be reading a queued input.
    const bool disconnected = tcpSender.disconnected() && !stopRequested();
    bool encoderError = false;
    for (size_t i = 0; i < streams.size(); ++i)
    {
        Stream& stream = streams[i];
        if (!stream.encoder)
            continue;
        if (!disconnected)
        {
            while (stream.slots.inFlight() > 0 && releaseOne(&stream, true))
            {
            }
            struct v4l2_buffer eos;
            struct v4l2_plane planes[MAX_PLANES];
            initializeV4l2Buffer(stream.slots.hasFree() ? stream.slots.nextFree() : 0, &eos, planes);
            stream.encoder->sendEos(eos);
        }
        encoderError = encoderError || stream.encoder->hasError();
        stream.encoder->shutdown();
        if (stream.dropped > 0)
        {
            std::printf("Stream %s: %llu frame(s) dropped with no free encoder input\n", stream.label.c_str(),
                        static_cast<unsigned long long>(stream.dropped));
        }
    }
    // Every lease must be dropped before capture stops.
    for (size_t i = 0; i < streams.size(); ++i)
        streams[i].slots.releaseAll();
    capture->stop();
    tcpSender.shutdown(ok && !encoderError && !stopRequested());
    if (fd >= 0)
        close(fd);
    capture.reset();
    if (receiverDisconnected)
        *receiverDisconnected = disconnected;
    return disconnected || (ok && !encoderError && !assembler.hasError());
}

} // namespace argus_sender

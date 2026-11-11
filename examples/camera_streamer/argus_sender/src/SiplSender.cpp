/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "SiplSender.h"

#include "Avs1Wire.h"
#include "CaptureFrame.h"
#include "Config.h"
#include "FrameBundleAssembler.h"
#include "FrameTiming.h"
#include "HevcEncoder.h"
#include "Runtime.h"
#include "SiplCapture.h"
#include "TcpTransport.h"

#include <linux/videodev2.h>

#include <NvVideoEncoder.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
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

uint64_t createSessionId()
{
    return (steadyNowNs() << 16) ^ static_cast<uint64_t>(getpid()) ^ 0x4156533100000000ULL;
}

// One encoder input slot: the DMA-BUF wrapper the encoder reads and the frame lease that keeps the SIPL buffer out of
// the ISP's hands until the encoder has dequeued it.
struct InputSlot
{
    DmaBuffer* wrapper = NULL;
    CaptureFrame frame;
};

struct Stream
{
    std::string label;
    uint32_t cameraId = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t fps = 0;
    uint64_t sequence = 0;
    uint64_t dropped = 0;
    std::unique_ptr<DirectHevcEncoder> encoder;
    InputSlot slots[kEncoderOutputPlaneBuffers];
    std::deque<uint32_t> freeIndices;
    uint32_t inFlight = 0;
};

void initializeV4l2Buffer(uint32_t index, struct v4l2_buffer* buffer, struct v4l2_plane planes[MAX_PLANES])
{
    std::memset(buffer, 0, sizeof(*buffer));
    std::memset(planes, 0, sizeof(struct v4l2_plane) * MAX_PLANES);
    buffer->index = index;
    buffer->m.planes = planes;
}

// Returns the SIPL buffer of one input the encoder has finished reading. With `wait` it blocks until one finishes;
// without, it asks once (the encoder's own dequeue timeout bounds the delay) and returns false if none is ready.
bool releaseOne(Stream* stream, bool wait)
{
    if (stream->inFlight == 0)
        return false;
    struct v4l2_buffer completed;
    struct v4l2_plane planes[MAX_PLANES];
    initializeV4l2Buffer(0, &completed, planes);
    DmaBuffer* buffer = NULL;
    if (wait ? !waitForEncoderInput(stream->encoder.get(), &completed, &buffer) :
               !stream->encoder->dequeueInput(completed, &buffer))
        return false;
    if (completed.index >= kEncoderOutputPlaneBuffers)
        return false;
    InputSlot& slot = stream->slots[completed.index];
    slot.frame.lease.reset();
    delete slot.wrapper;
    slot.wrapper = NULL;
    stream->freeIndices.push_back(completed.index);
    --stream->inFlight;
    return true;
}

} // namespace

bool runSiplSender(const AppConfig& config, bool* receiverDisconnected, bool appendDump)
{
    if (receiverDisconnected)
        *receiverDisconnected = false;

    SiplConfig siplConfig;
    siplConfig.platformConfigJson = config.sipl.platformConfigJson;
    siplConfig.platformConfigName = config.sipl.platformConfigName;
    siplConfig.linkMasks = config.sipl.linkMasks;
    siplConfig.nitoDir = config.sipl.nitoDir;
    siplConfig.isp0Buffers = config.sipl.isp0Buffers;
    const uint64_t sessionId = createSessionId();
    siplConfig.sessionEpoch = sessionId;

    std::unique_ptr<SiplCapture> capture;
    try
    {
        capture.reset(new SiplCapture(siplConfig));
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "SIPL capture setup failed: %s\n", e.what());
        return false;
    }

    // Streams name SIPL pipeline indices; geometry and rate come from the platform configuration, never from the
    // command line.
    std::vector<Stream> streams(config.streams.size());
    std::vector<Avs1StreamRecord> templates(config.streams.size());
    for (size_t i = 0; i < config.streams.size(); ++i)
    {
        const StreamConfig& configStream = config.streams[i];
        Stream& stream = streams[i];
        stream.label = configStream.id;
        stream.cameraId = configStream.cameras[0].device;
        const CameraDescriptor* found = NULL;
        for (size_t c = 0; c < capture->cameras().size(); ++c)
        {
            if (capture->cameras()[c].id == stream.cameraId)
                found = &capture->cameras()[c];
        }
        if (!found)
        {
            std::string available;
            for (size_t c = 0; c < capture->cameras().size(); ++c)
                available += " " + std::to_string(capture->cameras()[c].id);
            std::fprintf(stderr,
                         "Stream %s: device %u is not a SIPL pipeline index in '%s'. Available:%s (a pipeline index is "
                         "not the GMSL link index)\n",
                         stream.label.c_str(), stream.cameraId, config.sipl.platformConfigName.c_str(),
                         available.c_str());
            return false;
        }
        stream.width = found->width;
        stream.height = found->height;
        if ((configStream.output.width != 0 && configStream.output.width != stream.width) ||
            (configStream.output.height != 0 && configStream.output.height != stream.height))
        {
            std::fprintf(stderr,
                         "Stream %s: the sipl backend encodes the sensor's %ux%u and does not scale; remove "
                         "output.width/height or set them to the sensor size\n",
                         stream.label.c_str(), stream.width, stream.height);
            return false;
        }
        stream.fps = configStream.output.fps ? configStream.output.fps : static_cast<uint32_t>(std::lround(found->fps));
        if (stream.fps == 0)
        {
            std::fprintf(stderr, "Stream %s resolved to zero FPS\n", stream.label.c_str());
            return false;
        }
        std::printf("Stream %s: SIPL pipeline %u (%s) %ux%u @ %u FPS\n", stream.label.c_str(), stream.cameraId,
                    found->name.c_str(), stream.width, stream.height, stream.fps);

        Avs1StreamRecord& record = templates[i];
        record.codec = AVS1_CODEC_HEVC;
        record.pixelFormat = AVS1_PIXEL_FORMAT_NV12;
        record.rotation = static_cast<uint16_t>(configStream.output.rotation);
        record.width = stream.width;
        record.height = stream.height;
        record.fpsNumerator = stream.fps;
        record.fpsDenominator = 1;
        record.samples.resize(1);
        record.samples[0].cameraIndex = 0;
        record.samples[0].role = AVS1_ROLE_MONO;
        record.samples[0].deviceIndex = stream.cameraId;
        record.samples[0].sensorModeIndex = 0;
    }

    std::printf("Listening for AVS1 receiver on %s:%u\n", config.network.bindHost.c_str(), config.network.port);
    int fd = acceptTcpClient(config.network.bindHost, config.network.port, config.network.reconnectDelayMs);
    if (fd < 0)
        return false;

    TcpPacketSenderSettings transportSettings;
    transportSettings.port = config.network.port;
    transportSettings.dumpWirePath = config.network.dumpWirePath;
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
        const uint32_t gop = config.streams[i].output.gop ? config.streams[i].output.gop : stream.fps;
        if (!stream.encoder->initialize(stream.width, stream.height, stream.fps, gop))
            ok = false;
        for (uint32_t index = 0; index < kEncoderOutputPlaneBuffers; ++index)
            stream.freeIndices.push_back(index);
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
                continue;
            for (size_t i = 0; ok && i < streams.size(); ++i)
            {
                Stream& stream = streams[i];
                if (stream.encoder->hasError())
                {
                    ok = false;
                    break;
                }
                CaptureFrame frame;
                if (!capture->acquireLatest(stream.cameraId, &frame))
                    continue;
                // A frame arriving with no free encoder input is dropped, not waited for: the newest frame is the
                // one worth sending and the encoder is already a frame behind.
                if (stream.freeIndices.empty())
                    releaseOne(&stream, false);
                if (stream.freeIndices.empty())
                {
                    ++stream.dropped;
                    continue;
                }
                const uint32_t index = stream.freeIndices.front();
                DmaBuffer* wrapper = DmaBuffer::createExternal(frame.surface.fd);
                if (!wrapper)
                {
                    ok = false;
                    break;
                }

                ++stream.sequence;
                ++bundleSequence;
                std::vector<BundleStreamInput> inputs(streams.size());
                inputs[i].present = true;
                inputs[i].sequence = stream.sequence;
                FrameTiming timing;
                timing.frameNumber = frame.sequence;
                timing.acquiredSteadyNs = frame.readyMonotonicNs;
                // Only a TSC in nanoseconds may go into the AVS1 TSC fields; a raw backend counter stays out
                // until its units are declared.
                inputs[i].timings.push_back(timing);
                uint32_t readyDomain = SENDER_CLOCK_UNKNOWN;
                const uint64_t readyNs = readSenderClockNs(&readyDomain);
                uint32_t submitDomain = SENDER_CLOCK_UNKNOWN;
                const uint64_t submitNs = readSenderClockNs(&submitDomain);
                if (readyDomain != submitDomain)
                    readyDomain = SENDER_CLOCK_UNKNOWN;
                if (!assembler.beginBundle(bundleSequence, inputs, readyNs, submitNs, readyDomain, 0) ||
                    !assembler.recordCopyDone(i, bundleSequence, frame.readyMonotonicNs) ||
                    !assembler.recordProcessStart(i, bundleSequence, steadyNowNs()))
                {
                    delete wrapper;
                    ok = false;
                    break;
                }

                struct v4l2_buffer v4l2Buffer;
                struct v4l2_plane planes[MAX_PLANES];
                initializeV4l2Buffer(index, &v4l2Buffer, planes);
                if (!stream.encoder->queueInput(v4l2Buffer, wrapper, bundleSequence))
                {
                    delete wrapper;
                    ok = false;
                    break;
                }
                stream.freeIndices.pop_front();
                stream.slots[index].wrapper = wrapper;
                stream.slots[index].frame = std::move(frame);
                ++stream.inFlight;
                ++frames;
            }
        }
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "SIPL capture failed: %s\n", e.what());
        ok = false;
    }

    // Finish the encoder before any SIPL buffer goes back: it may still be reading a queued input.
    const bool disconnected = tcpSender.disconnected() && !stopRequested();
    bool encoderError = false;
    for (size_t i = 0; i < streams.size(); ++i)
    {
        Stream& stream = streams[i];
        if (!stream.encoder)
            continue;
        if (!disconnected)
        {
            while (stream.inFlight > 0 && releaseOne(&stream, true))
            {
            }
            struct v4l2_buffer eos;
            struct v4l2_plane planes[MAX_PLANES];
            initializeV4l2Buffer(stream.freeIndices.empty() ? 0 : stream.freeIndices.front(), &eos, planes);
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
    {
        for (uint32_t index = 0; index < kEncoderOutputPlaneBuffers; ++index)
        {
            streams[i].slots[index].frame.lease.reset();
            delete streams[i].slots[index].wrapper;
            streams[i].slots[index].wrapper = NULL;
        }
    }
    if (capture)
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

/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "StreamProcessor.h"

#include "CaptureManager.h"
#include "FrameBundleAssembler.h"
#include "FrameCopyWorker.h"
#include "HevcEncoder.h"
#include "NvRemap.h"
#include "StereoTiming.h"

#include <linux/videodev2.h>

#include <NvVideoEncoder.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <nvbufsurface.h>
#include <string>
#include <vector>

namespace argus_sender
{
namespace
{

const uint32_t kProcessorBuffers = 2;

uint64_t steadyNowNs()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

void initializeV4l2Buffer(uint32_t index, struct v4l2_buffer* buffer, struct v4l2_plane planes[MAX_PLANES])
{
    std::memset(buffer, 0, sizeof(*buffer));
    std::memset(planes, 0, sizeof(struct v4l2_plane) * MAX_PLANES);
    buffer->index = index;
    buffer->m.planes = planes;
}

} // namespace

class StreamProcessor::Impl
{
public:
    struct CameraState
    {
        std::string label;
        std::vector<DmaBuffer*> buffers;
        std::deque<uint32_t> freeIndices;
        std::unique_ptr<DirectHevcEncoder> encoder;
        std::unique_ptr<FrameCopyWorker> copyWorker;
        // Raw wire payload produced on the copy worker's thread (tight pack
        // or remap); collected by finishSubmit after wait().
        std::vector<uint8_t> pendingPayload;
        DmaBuffer* pendingInput = NULL;
        uint32_t inFlight = 0;
        uint32_t selectedIndex = 0;
        uint64_t directReadySteadyNs = 0;
        bool copySubmitted = false;
    };

    Impl(const StreamConfig& config,
         const ResolvedOutputGeometry& geometry,
         size_t streamIndex,
         bool kpi,
         uint32_t logIntervalSeconds,
         bool benchEncode)
        : m_config(config),
          m_geometry(geometry),
          m_streamIndex(streamIndex),
          m_assembler(NULL),
          m_bundleSequence(0),
          m_copyPending(false)
    {
        m_encoderSettings.bitrateMbps = config.output.bitrateMbps;
        m_encoderSettings.kpi = kpi;
        m_encoderSettings.logIntervalSeconds = logIntervalSeconds;
        m_encoderSettings.benchEncode = benchEncode;
        m_cameras.resize(config.cameras.size());
        for (size_t i = 0; i < m_cameras.size(); ++i)
        {
            m_cameras[i].label =
                config.id + "." + (config.type == STREAM_TYPE_STEREO ? (i == 0 ? "left" : "right") : "mono");
            if (config.output.codec == CODEC_RAW_NV12)
            {
                m_cameras[i].copyWorker.reset(new FrameCopyWorker(m_cameras[i].label.c_str()));
            }
        }
    }

    ~Impl()
    {
        shutdown();
        for (size_t camera = 0; camera < m_cameras.size(); ++camera)
        {
            for (size_t i = 0; i < m_cameras[camera].buffers.size(); ++i)
                delete m_cameras[camera].buffers[i];
        }
    }

    bool start(FrameBundleAssembler* assembler, uint32_t encoderFps)
    {
        if (!assembler || encoderFps == 0)
            return false;
        m_assembler = assembler;
        if (m_config.output.remap.enabled())
        {
            std::string remapError;
            if (!loadNvRemapTable(m_config.output.remap.file, m_geometry.captureWidth, m_geometry.captureHeight,
                                  m_geometry.width, m_geometry.height, &m_remapTable, &remapError))
            {
                std::fprintf(stderr, "Stream %s remap: %s\n", m_config.id.c_str(), remapError.c_str());
                return false;
            }
            std::printf(
                "Stream %s remap: %s (%ux%u source -> %ux%u via %ux "
                "supersampled LUT)\n",
                m_config.id.c_str(), m_config.output.remap.file.c_str(), m_geometry.captureWidth,
                m_geometry.captureHeight, m_geometry.width, m_geometry.height, m_remapTable.supersample);
        }
        const Argus::Size2D<uint32_t> size(m_geometry.captureWidth, m_geometry.captureHeight);
        for (size_t camera = 0; camera < m_cameras.size(); ++camera)
        {
            CameraState& state = m_cameras[camera];
            if (m_config.output.codec == CODEC_RAW_NV12)
            {
                for (uint32_t i = 0; i < kProcessorBuffers; ++i)
                {
                    DmaBuffer* buffer = DmaBuffer::create(size, NVBUF_COLOR_FORMAT_NV12, NVBUF_LAYOUT_PITCH);
                    if (!buffer)
                    {
                        std::fprintf(stderr, "Failed to allocate %s processing DMABUF %u\n", state.label.c_str(), i);
                        return false;
                    }
                    state.buffers.push_back(buffer);
                    state.freeIndices.push_back(i);
                }
                if (!state.copyWorker || !state.copyWorker->start())
                    return false;
            }
            else
            {
                for (uint32_t i = 0; i < kEncoderOutputPlaneBuffers; ++i)
                    state.freeIndices.push_back(i);
                state.encoder.reset(
                    new DirectHevcEncoder(assembler->encodedSink(m_streamIndex, camera), m_encoderSettings));
                const uint32_t gop = m_config.output.gop ? m_config.output.gop : encoderFps;
                if (!state.encoder->initialize(m_geometry.width, m_geometry.height, encoderFps, gop))
                {
                    return false;
                }
            }
        }
        return true;
    }

    bool waitForInputBuffers()
    {
        for (size_t camera = 0; camera < m_cameras.size(); ++camera)
        {
            CameraState& state = m_cameras[camera];
            while (state.freeIndices.empty() && state.inFlight > 0)
            {
                if (!releaseOne(&state))
                    return false;
            }
            if (state.freeIndices.empty())
                return false;
        }
        return true;
    }

    bool submitCopies(CapturedFrameSet* frames, uint64_t bundleSequence)
    {
        if (!frames || frames->frames.size() != m_cameras.size() || frames->dmaBuffers.size() != m_cameras.size() ||
            m_copyPending || bundleSequence == 0)
        {
            return false;
        }
        for (size_t camera = 0; camera < m_cameras.size(); ++camera)
        {
            CameraState& state = m_cameras[camera];
            const bool raw = m_config.output.codec == CODEC_RAW_NV12;
            if (state.freeIndices.empty() || (raw && (!frames->frames[camera].get() || !state.copyWorker)) ||
                (!raw && !frames->dmaBuffers[camera].get()))
            {
                return false;
            }
        }
        m_bundleSequence = bundleSequence;
        for (size_t camera = 0; camera < m_cameras.size(); ++camera)
        {
            CameraState& state = m_cameras[camera];
            state.selectedIndex = state.freeIndices.front();
            const bool raw = m_config.output.codec == CODEC_RAW_NV12;
            if (raw)
            {
                RawPackJob pack;
                pack.width = m_geometry.width;
                pack.height = m_geometry.height;
                pack.remap = m_remapTable.loaded() ? &m_remapTable : nullptr;
                pack.payload = &state.pendingPayload;
                state.copySubmitted =
                    state.copyWorker->submit(frames->frames[camera].get(), state.buffers[state.selectedIndex], &pack);
            }
            else
            {
                state.pendingInput = frames->dmaBuffers[camera].release();
                state.directReadySteadyNs = frames->timings[camera].acquiredSteadyNs;
                state.copySubmitted = state.pendingInput != NULL;
            }
            if (!state.copySubmitted)
            {
                std::fprintf(stderr, "Failed to submit %s frame processing; draining previously submitted work\n",
                             state.label.c_str());
                discardPendingCopies();
                return false;
            }
        }
        m_copyPending = true;
        return true;
    }

    bool finishSubmit()
    {
        if (!m_copyPending)
            return false;
        bool ok = true;
        uint64_t copyDoneSteadyNs = 0;
        for (size_t camera = 0; camera < m_cameras.size(); ++camera)
        {
            CameraState& state = m_cameras[camera];
            uint64_t cameraCopyDoneSteadyNs = 0;
            if (!state.copySubmitted)
                ok = false;
            else if (m_config.output.codec == CODEC_RAW_NV12)
            {
                if (!state.copyWorker->wait(&cameraCopyDoneSteadyNs))
                    ok = false;
            }
            else
            {
                cameraCopyDoneSteadyNs = state.directReadySteadyNs;
            }
            copyDoneSteadyNs = std::max(copyDoneSteadyNs, cameraCopyDoneSteadyNs);
            state.copySubmitted = false;
        }
        if (!ok)
        {
            releasePendingDirectInputs();
            m_copyPending = false;
            return false;
        }

        if (!m_assembler->recordCopyDone(m_streamIndex, m_bundleSequence, copyDoneSteadyNs))
        {
            releasePendingDirectInputs();
            m_copyPending = false;
            return false;
        }
        for (size_t camera = 0; camera < m_cameras.size(); ++camera)
        {
            CameraState& state = m_cameras[camera];
            const uint32_t index = state.selectedIndex;
            const uint64_t encodeStartSteadyNs = steadyNowNs();
            if (!m_assembler->recordProcessStart(m_streamIndex, m_bundleSequence, encodeStartSteadyNs))
            {
                ok = false;
                break;
            }
            if (m_config.output.codec == CODEC_RAW_NV12)
            {
                // The tight pack (and remap, when configured) already ran on
                // the copy worker's thread; the payload is ready to submit.
                if (state.pendingPayload.empty())
                {
                    ok = false;
                    break;
                }
                uint32_t clockDomain = SENDER_CLOCK_UNKNOWN;
                const uint64_t processDoneNs = readSenderClockNs(&clockDomain);
                if (!m_assembler->submitRaw(m_streamIndex, camera, m_bundleSequence, std::move(state.pendingPayload),
                                            processDoneNs, clockDomain))
                {
                    ok = false;
                    break;
                }
                state.pendingPayload.clear();
            }
            else
            {
                state.freeIndices.pop_front();
                DmaBuffer* input = state.pendingInput;
                state.pendingInput = NULL;
                struct v4l2_buffer v4l2Buffer;
                struct v4l2_plane planes[MAX_PLANES];
                initializeV4l2Buffer(index, &v4l2Buffer, planes);
                if (!state.encoder->queueInput(v4l2Buffer, input, m_bundleSequence))
                {
                    state.freeIndices.push_front(index);
                    if (input && !input->releaseToArgus())
                    {
                        std::fprintf(stderr, "Failed to return %s direct input after encoder queue failure\n",
                                     state.label.c_str());
                    }
                    ok = false;
                    break;
                }
                ++state.inFlight;
                state.directReadySteadyNs = 0;
            }
        }
        if (!ok)
            releasePendingDirectInputs();
        m_copyPending = false;
        return ok;
    }

    bool discardPendingCopies()
    {
        bool ok = true;
        bool hadPending = m_copyPending;
        for (size_t camera = 0; camera < m_cameras.size(); ++camera)
        {
            CameraState& state = m_cameras[camera];
            if (!state.copySubmitted && !state.pendingInput)
                continue;
            hadPending = true;
            if (m_config.output.codec == CODEC_RAW_NV12)
            {
                uint64_t ignoredCompletionSteadyNs = 0;
                if (!state.copyWorker || !state.copyWorker->wait(&ignoredCompletionSteadyNs))
                {
                    ok = false;
                }
            }
            else if (state.pendingInput)
            {
                if (!state.pendingInput->releaseToArgus())
                    ok = false;
                state.pendingInput = NULL;
            }
            state.copySubmitted = false;
            state.directReadySteadyNs = 0;
        }
        m_copyPending = false;
        m_bundleSequence = 0;
        if (hadPending && !ok)
        {
            std::fprintf(stderr, "Failed while draining abandoned frame-copy work for stream %s\n", m_config.id.c_str());
        }
        return ok;
    }

    bool drain()
    {
        bool ok = true;
        for (size_t camera = 0; camera < m_cameras.size(); ++camera)
        {
            while (m_cameras[camera].inFlight > 0)
            {
                if (!releaseOne(&m_cameras[camera]))
                {
                    ok = false;
                    break;
                }
            }
        }
        return ok;
    }

    bool sendEos()
    {
        bool ok = true;
        for (size_t camera = 0; camera < m_cameras.size(); ++camera)
        {
            CameraState& state = m_cameras[camera];
            if (!state.encoder)
                continue;
            struct v4l2_buffer buffer;
            struct v4l2_plane planes[MAX_PLANES];
            const uint32_t index = state.freeIndices.empty() ? 0 : state.freeIndices.front();
            initializeV4l2Buffer(index, &buffer, planes);
            if (!state.encoder->sendEos(buffer))
                ok = false;
        }
        return ok;
    }

    void shutdown()
    {
        releasePendingDirectInputs();
        // A receiver disconnect may skip the coordinator's normal drain. The
        // output-plane dequeue owns the final reference to each direct Argus
        // buffer, so reclaim those buffers before destroying encoder or Argus
        // stream resources.
        if (m_config.output.codec == CODEC_HEVC)
            drain();
        for (size_t camera = 0; camera < m_cameras.size(); ++camera)
        {
            if (m_cameras[camera].encoder)
                m_cameras[camera].encoder->shutdown();
            if (m_cameras[camera].copyWorker)
                m_cameras[camera].copyWorker->shutdown();
        }
    }

    bool hasError() const
    {
        for (size_t camera = 0; camera < m_cameras.size(); ++camera)
        {
            if (m_cameras[camera].encoder && m_cameras[camera].encoder->hasError())
            {
                return true;
            }
        }
        return false;
    }

private:
    void releasePendingDirectInputs()
    {
        if (m_config.output.codec != CODEC_HEVC)
            return;
        for (size_t camera = 0; camera < m_cameras.size(); ++camera)
        {
            CameraState& state = m_cameras[camera];
            if (state.pendingInput)
            {
                if (!state.pendingInput->releaseToArgus())
                {
                    std::fprintf(stderr, "Failed to release pending %s direct input to Argus\n", state.label.c_str());
                }
                state.pendingInput = NULL;
            }
        }
    }

    bool releaseOne(CameraState* state)
    {
        if (!state || !state->encoder || state->inFlight == 0)
            return false;
        struct v4l2_buffer completed;
        struct v4l2_plane planes[MAX_PLANES];
        initializeV4l2Buffer(0, &completed, planes);
        DmaBuffer* buffer = NULL;
        if (!waitForEncoderInput(state->encoder.get(), &completed, &buffer))
            return false;
        if (completed.index >= kEncoderOutputPlaneBuffers || !buffer)
            return false;
        if (buffer->isArgusBound() && !buffer->releaseToArgus())
        {
            std::fprintf(stderr, "Failed to release encoded direct DMABUF to Argus\n");
            // The encoder slot is still dequeued even if returning the backing
            // surface to Argus failed. Account it before reporting the error so
            // shutdown cannot spin forever on an already-dequeued slot.
            state->freeIndices.push_back(completed.index);
            --state->inFlight;
            return false;
        }
        state->freeIndices.push_back(completed.index);
        --state->inFlight;
        return true;
    }

    StreamConfig m_config;
    ResolvedOutputGeometry m_geometry;
    size_t m_streamIndex;
    FrameBundleAssembler* m_assembler;
    HevcEncoderSettings m_encoderSettings;
    std::vector<CameraState> m_cameras;
    NvRemapTable m_remapTable;
    uint64_t m_bundleSequence;
    bool m_copyPending;
};

StreamProcessor::StreamProcessor(const StreamConfig& config,
                                 const ResolvedOutputGeometry& geometry,
                                 size_t streamIndex,
                                 bool kpi,
                                 uint32_t logIntervalSeconds,
                                 bool benchEncode)
    : m_impl(new Impl(config, geometry, streamIndex, kpi, logIntervalSeconds, benchEncode))
{
}
StreamProcessor::~StreamProcessor()
{
}
bool StreamProcessor::start(FrameBundleAssembler* assembler, uint32_t fps)
{
    return m_impl->start(assembler, fps);
}
bool StreamProcessor::waitForInputBuffers()
{
    return m_impl->waitForInputBuffers();
}
bool StreamProcessor::submitCopies(CapturedFrameSet* frames, uint64_t sequence)
{
    return m_impl->submitCopies(frames, sequence);
}
bool StreamProcessor::finishSubmit()
{
    return m_impl->finishSubmit();
}
bool StreamProcessor::discardPendingCopies()
{
    return m_impl->discardPendingCopies();
}
bool StreamProcessor::drain()
{
    return m_impl->drain();
}
bool StreamProcessor::sendEos()
{
    return m_impl->sendEos();
}
void StreamProcessor::shutdown()
{
    m_impl->shutdown();
}
bool StreamProcessor::hasError() const
{
    return m_impl->hasError();
}

} // namespace argus_sender

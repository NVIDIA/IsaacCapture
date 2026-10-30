/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "HevcEncoder.h"

#include "EncodedFrameSink.h"
#include "Error.h"
#include "Runtime.h"
#include "StereoTiming.h"

#include <NvVideoEncoder.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <thread>
#include <v4l2_nv_extensions.h>

namespace argus_sender
{
namespace
{

const uint32_t kEncoderCapturePlaneBuffers = 4;
const uint32_t kEncodedBufferBytes = 16 * 1024 * 1024;

} // namespace

DmaBuffer* DmaBuffer::create(const Argus::Size2D<uint32_t>& size,
                             NvBufSurfaceColorFormat colorFormat,
                             NvBufSurfaceLayout layout)
{
    DmaBuffer* buffer = new DmaBuffer();
    if (!buffer)
        return NULL;

    NvBufSurf::NvCommonAllocateParams params;
    std::memset(&params, 0, sizeof(params));
    params.memtag = NvBufSurfaceTag_CAMERA;
    params.width = size.width();
    params.height = size.height();
    params.colorFormat = colorFormat;
    params.layout = layout;
    params.memType = NVBUF_MEM_SURFACE_ARRAY;

    if (NvBufSurf::NvAllocate(&params, 1, &buffer->m_fd) != 0)
    {
        delete buffer;
        return NULL;
    }

    if (NvBufSurfaceFromFd(buffer->m_fd, reinterpret_cast<void**>(&buffer->m_surface)) != 0 || !buffer->m_surface)
    {
        delete buffer;
        return NULL;
    }

    buffer->planes[0].fd = buffer->m_fd;
    buffer->planes[0].bytesused = 1;
    return buffer;
}

int DmaBuffer::getFd() const
{
    return m_fd;
}

DmaBuffer::DmaBuffer()
    : NvBuffer(0, 0),
      m_fd(-1),
      m_surface(NULL),
      m_eglImageMapped(false),
      m_argusStream(NULL),
      m_argusBuffer(NULL),
      m_argusAcquired(false)
{
}

DmaBuffer::~DmaBuffer()
{
    if (m_fd < 0)
        return;
    if (m_eglImageMapped && m_surface)
        NvBufSurfaceUnMapEglImage(m_surface, 0);
    if (m_surface)
        NvBufSurfaceDestroy(m_surface);
    m_surface = NULL;
    m_fd = -1;
}

bool DmaBuffer::mapEglImage()
{
    if (!m_surface)
        return false;
    if (m_eglImageMapped)
        return getEglImage() != EGL_NO_IMAGE_KHR;
    if (NvBufSurfaceMapEglImage(m_surface, 0) != 0)
        return false;
    m_eglImageMapped = true;
    return getEglImage() != EGL_NO_IMAGE_KHR;
}

EGLImageKHR DmaBuffer::getEglImage() const
{
    if (!m_surface || !m_eglImageMapped)
        return EGL_NO_IMAGE_KHR;
    return m_surface->surfaceList[0].mappedAddr.eglImage;
}

bool DmaBuffer::bindArgusBuffer(Argus::IBufferOutputStream* stream, Argus::Buffer* buffer)
{
    if (!stream || !buffer || m_argusStream || m_argusBuffer)
        return false;
    m_argusStream = stream;
    m_argusBuffer = buffer;
    m_argusAcquired.store(true);
    return true;
}

bool DmaBuffer::markArgusAcquired(Argus::Buffer* buffer)
{
    if (!m_argusStream || buffer != m_argusBuffer)
        return false;
    bool expected = false;
    return m_argusAcquired.compare_exchange_strong(expected, true);
}

bool DmaBuffer::releaseToArgus()
{
    if (!m_argusStream || !m_argusBuffer)
        return false;
    bool expected = true;
    if (!m_argusAcquired.compare_exchange_strong(expected, false))
        return false;
    if (m_argusStream->releaseBuffer(m_argusBuffer) != Argus::STATUS_OK)
    {
        m_argusAcquired.store(true);
        return false;
    }
    return true;
}

bool DmaBuffer::isArgusBound() const
{
    return m_argusStream && m_argusBuffer;
}

Argus::Buffer* DmaBuffer::getArgusBuffer() const
{
    return m_argusBuffer;
}

class DirectHevcEncoder::Impl
{
public:
    Impl(EncodedFrameSink* sink, const HevcEncoderSettings& settings)
        : m_sink(sink),
          m_settings(settings),
          m_encoder(NULL),
          m_totalFrames(0),
          m_intervalFrames(0),
          m_intervalBytes(0),
          m_benchSamples(0),
          m_benchSumMs(0.0),
          m_benchMinMs(std::numeric_limits<double>::max()),
          m_benchMaxMs(0.0),
          m_benchSummaryPrinted(false),
          m_error(false)
    {
    }

    ~Impl()
    {
        shutdown();
    }

    bool initialize(uint32_t width, uint32_t height, uint32_t fps, uint32_t gop)
    {
        m_statsStart = std::chrono::steady_clock::now();
        m_encoder = NvVideoEncoder::createVideoEncoder("argus_sender_hevc");
        if (!m_encoder)
            ORIGINATE_ERROR("Could not create HEVC NvVideoEncoder");

        if (m_encoder->setCapturePlaneFormat(V4L2_PIX_FMT_H265, width, height, kEncodedBufferBytes) < 0)
            ORIGINATE_ERROR("Could not set HEVC encoder capture plane format");
        if (m_encoder->setOutputPlaneFormat(V4L2_PIX_FMT_NV12M, width, height, V4L2_COLORSPACE_REC709) < 0)
            ORIGINATE_ERROR("Could not set HEVC encoder output plane format");
        if (m_encoder->setBitrate(m_settings.bitrateMbps * 1000000U) < 0)
            ORIGINATE_ERROR("Could not set HEVC bitrate");
        if (m_encoder->setProfile(V4L2_MPEG_VIDEO_H265_PROFILE_MAIN) < 0)
            ORIGINATE_ERROR("Could not set HEVC profile");
        if (m_encoder->setRateControlMode(V4L2_MPEG_VIDEO_BITRATE_MODE_CBR) < 0)
            ORIGINATE_ERROR("Could not set HEVC CBR mode");
        if (m_encoder->setIFrameInterval(gop) < 0)
            ORIGINATE_ERROR("Could not set HEVC I-frame interval");
        if (m_encoder->setFrameRate(fps, 1) < 0)
            ORIGINATE_ERROR("Could not set HEVC encoder frame rate");

        setOptionalControl("max performance mode", m_encoder->setMaxPerfMode(1));
        setOptionalControl("ultrafast hardware preset", m_encoder->setHWPresetType(V4L2_ENC_HW_PRESET_ULTRAFAST));
        setOptionalControl("IDR interval", m_encoder->setIDRInterval(gop));
        setOptionalControl("disable B-frames", m_encoder->setNumBFrames(0));
        setOptionalControl("single reference frame", m_encoder->setNumReferenceFrames(1));
        setOptionalControl("insert SPS/PPS at IDR", m_encoder->setInsertSpsPpsAtIdrEnabled(true));
        setOptionalControl("insert AUD", m_encoder->setInsertAudEnabled(true));

        if (m_encoder->output_plane.setupPlane(V4L2_MEMORY_DMABUF, kEncoderOutputPlaneBuffers, true, false) < 0)
            ORIGINATE_ERROR("Could not setup HEVC encoder output plane");
        if (m_encoder->capture_plane.setupPlane(V4L2_MEMORY_MMAP, kEncoderCapturePlaneBuffers, true, false) < 0)
            ORIGINATE_ERROR("Could not setup HEVC encoder capture plane");

        if (m_encoder->output_plane.setStreamStatus(true) < 0)
            ORIGINATE_ERROR("Failed to stream on HEVC output plane");
        if (m_encoder->capture_plane.setStreamStatus(true) < 0)
            ORIGINATE_ERROR("Failed to stream on HEVC capture plane");

        m_encoder->capture_plane.setDQThreadCallback(capturePlaneCallback);
        if (m_encoder->capture_plane.startDQThread(this) < 0)
            ORIGINATE_ERROR("Failed to start HEVC capture-plane dequeue thread");

        for (uint32_t i = 0; i < m_encoder->capture_plane.getNumBuffers(); ++i)
        {
            struct v4l2_buffer v4l2Buf;
            struct v4l2_plane planes[MAX_PLANES];
            std::memset(&v4l2Buf, 0, sizeof(v4l2Buf));
            std::memset(planes, 0, sizeof(planes));
            v4l2Buf.index = i;
            v4l2Buf.m.planes = planes;
            if (m_encoder->capture_plane.qBuffer(v4l2Buf, NULL) < 0)
                ORIGINATE_ERROR("Failed to queue HEVC capture-plane buffer");
        }

        return true;
    }

    bool queueInput(struct v4l2_buffer& v4l2Buf, DmaBuffer* buffer, uint64_t frameSeq = 0)
    {
        if (!m_encoder || !buffer)
            return false;
        const auto queueStart = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lock(m_benchMutex);
            m_pendingSeqs.push_back(frameSeq);
            if (m_settings.benchEncode)
                m_benchQueueTimes.push_back(queueStart);
        }
        if (m_encoder->output_plane.qBuffer(v4l2Buf, buffer) < 0)
        {
            forgetQueuedFrameMetadata();
            m_error = true;
            return false;
        }
        return true;
    }

    bool dequeueInput(struct v4l2_buffer& v4l2Buf, DmaBuffer** buffer)
    {
        NvBuffer* sharedBuffer = NULL;
        if (!m_encoder)
            return false;
        if (m_encoder->output_plane.dqBuffer(v4l2Buf, NULL, &sharedBuffer, 10) < 0)
        {
            if (errno != EAGAIN && errno != EINTR)
                m_error = true;
            return false;
        }
        if (buffer)
            *buffer = static_cast<DmaBuffer*>(sharedBuffer);
        return true;
    }

    bool sendEos(struct v4l2_buffer& v4l2Buf)
    {
        if (!m_encoder)
            return true;
        v4l2Buf.m.planes[0].m.fd = -1;
        v4l2Buf.m.planes[0].bytesused = 0;
        if (m_encoder->output_plane.qBuffer(v4l2Buf, NULL) < 0)
        {
            m_error = true;
            return false;
        }
        return true;
    }

    bool hasError() const
    {
        return m_error || (m_sink && m_sink->hasError());
    }

    void shutdown()
    {
        if (!m_encoder)
            return;
        m_encoder->capture_plane.waitForDQThread(2000);
        printEncodeBenchSummary();
        m_encoder->output_plane.setStreamStatus(false);
        m_encoder->capture_plane.setStreamStatus(false);
        delete m_encoder;
        m_encoder = NULL;
    }

private:
    void setOptionalControl(const char* name, int ret)
    {
        if (ret < 0)
            std::fprintf(stderr, "Warning: HEVC encoder did not accept %s control; continuing\n", name);
    }

    static bool capturePlaneCallback(struct v4l2_buffer* v4l2Buf, NvBuffer* buffer, NvBuffer*, void* arg)
    {
        Impl* self = static_cast<Impl*>(arg);
        return self->onCapturePlaneBuffer(v4l2Buf, buffer);
    }

    bool onCapturePlaneBuffer(struct v4l2_buffer* v4l2Buf, NvBuffer* buffer)
    {
        if (!v4l2Buf || !buffer)
        {
            m_error = true;
            return false;
        }

        const size_t encodedBytes = buffer->planes[0].bytesused;
        if (encodedBytes == 0)
            return false;

        const uint64_t frameSeq = popPendingFrameSeq();
        recordEncodeLatency(encodedBytes);

        uint32_t senderClockDomain = SENDER_CLOCK_UNKNOWN;
        const uint64_t encodeDoneNs = readSenderClockNs(&senderClockDomain);
        if (!m_sink ||
            !m_sink->enqueueEncoded(buffer->planes[0].data, encodedBytes, frameSeq, encodeDoneNs, senderClockDomain))
        {
            if (!stopRequested())
                m_error = true;
            return false;
        }

        ++m_totalFrames;
        ++m_intervalFrames;
        m_intervalBytes += encodedBytes;

        if (m_settings.kpi && m_settings.logIntervalSeconds > 0)
        {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - m_statsStart).count();
            if (elapsed >= static_cast<double>(m_settings.logIntervalSeconds))
            {
                const double fpsActual = static_cast<double>(m_intervalFrames) / elapsed;
                const double mbps = static_cast<double>(m_intervalBytes) * 8.0 / elapsed / 1.0e6;
                if (verboseLoggingEnabled())
                {
                    std::printf("argus_sender[hevc-direct]: %.2f FPS, %.2f Mbps, total=%llu\n", fpsActual, mbps,
                                static_cast<unsigned long long>(m_totalFrames));
                }
                m_intervalFrames = 0;
                m_intervalBytes = 0;
                m_statsStart = now;
            }
        }

        if (m_encoder->capture_plane.qBuffer(*v4l2Buf, NULL) < 0)
        {
            m_error = true;
            return false;
        }
        return true;
    }

    uint64_t popPendingFrameSeq()
    {
        std::lock_guard<std::mutex> lock(m_benchMutex);
        if (m_pendingSeqs.empty())
            return 0;
        const uint64_t seq = m_pendingSeqs.front();
        m_pendingSeqs.pop_front();
        return seq;
    }

    void forgetQueuedFrameMetadata()
    {
        std::lock_guard<std::mutex> lock(m_benchMutex);
        if (!m_pendingSeqs.empty())
            m_pendingSeqs.pop_back();
        if (m_settings.benchEncode && !m_benchQueueTimes.empty())
            m_benchQueueTimes.pop_back();
    }

    void recordEncodeLatency(size_t encodedBytes)
    {
        if (!m_settings.benchEncode)
            return;

        std::chrono::steady_clock::time_point queueTime;
        {
            std::lock_guard<std::mutex> lock(m_benchMutex);
            if (m_benchQueueTimes.empty())
                return;
            queueTime = m_benchQueueTimes.front();
            m_benchQueueTimes.pop_front();
        }

        const auto now = std::chrono::steady_clock::now();
        const double encodeMs =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(now - queueTime).count();

        std::lock_guard<std::mutex> lock(m_benchMutex);
        ++m_benchSamples;
        m_benchSumMs += encodeMs;
        if (encodeMs < m_benchMinMs)
            m_benchMinMs = encodeMs;
        if (encodeMs > m_benchMaxMs)
            m_benchMaxMs = encodeMs;

        std::printf("argus_sender[encode-bench]: frame=%llu encode_ms=%.3f packet_bytes=%zu\n",
                    static_cast<unsigned long long>(m_totalFrames + 1), encodeMs, encodedBytes);
    }

    void printEncodeBenchSummary()
    {
        if (!m_settings.benchEncode)
            return;

        std::lock_guard<std::mutex> lock(m_benchMutex);
        if (m_benchSummaryPrinted)
            return;
        m_benchSummaryPrinted = true;

        if (m_benchSamples == 0)
        {
            std::printf("argus_sender[encode-bench-summary]: no encoded frames sampled\n");
            return;
        }

        std::printf("argus_sender[encode-bench-summary]: n=%llu mean=%.3f ms min=%.3f ms max=%.3f ms\n",
                    static_cast<unsigned long long>(m_benchSamples), m_benchSumMs / static_cast<double>(m_benchSamples),
                    m_benchMinMs, m_benchMaxMs);
    }

    EncodedFrameSink* m_sink;
    const HevcEncoderSettings& m_settings;
    NvVideoEncoder* m_encoder;
    uint64_t m_totalFrames;
    uint64_t m_intervalFrames;
    uint64_t m_intervalBytes;
    std::chrono::steady_clock::time_point m_statsStart;
    std::mutex m_benchMutex;
    uint64_t m_benchSamples;
    double m_benchSumMs;
    double m_benchMinMs;
    double m_benchMaxMs;
    std::deque<std::chrono::steady_clock::time_point> m_benchQueueTimes;
    std::deque<uint64_t> m_pendingSeqs;
    bool m_benchSummaryPrinted;
    std::atomic<bool> m_error;
};

bool waitForEncoderInput(DirectHevcEncoder* encoder, struct v4l2_buffer* v4l2Buf, DmaBuffer** buffer)
{
    while (!encoder->hasError())
    {
        if (encoder->dequeueInput(*v4l2Buf, buffer))
            return true;
        if (stopRequested())
            return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

DirectHevcEncoder::DirectHevcEncoder(EncodedFrameSink* sink, const HevcEncoderSettings& settings)
    : m_impl(new Impl(sink, settings))
{
}

DirectHevcEncoder::~DirectHevcEncoder()
{
}

bool DirectHevcEncoder::initialize(uint32_t width, uint32_t height, uint32_t fps, uint32_t gop)
{
    return m_impl->initialize(width, height, fps, gop);
}

bool DirectHevcEncoder::queueInput(struct v4l2_buffer& v4l2Buf, DmaBuffer* buffer, uint64_t frameSeq)
{
    return m_impl->queueInput(v4l2Buf, buffer, frameSeq);
}

bool DirectHevcEncoder::dequeueInput(struct v4l2_buffer& v4l2Buf, DmaBuffer** buffer)
{
    return m_impl->dequeueInput(v4l2Buf, buffer);
}

bool DirectHevcEncoder::sendEos(struct v4l2_buffer& v4l2Buf)
{
    return m_impl->sendEos(v4l2Buf);
}

bool DirectHevcEncoder::hasError() const
{
    return m_impl->hasError();
}

void DirectHevcEncoder::shutdown()
{
    m_impl->shutdown();
}

} // namespace argus_sender

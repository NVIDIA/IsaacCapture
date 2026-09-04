// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "jetson_encoder.hpp"

#include "rgba_to_nv12.cuh"

#include <linux/videodev2.h>

#include <NvVideoEncoder.h>
#include <cstring>
#include <cuda_runtime.h>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>

namespace plugins
{
namespace sensing
{

namespace
{

constexpr uint32_t kOutputBuffers = 6;
constexpr uint32_t kCaptureBuffers = 6;

void check(int ret, const char* what)
{
    if (ret < 0)
        throw std::runtime_error(std::string("JetsonEncoder: ") + what + " failed");
}

void check_cuda(cudaError_t err, const char* what)
{
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("JetsonEncoder: ") + what + ": " + cudaGetErrorString(err));
}

} // namespace

struct JetsonEncoder::Impl
{
    EncoderConfig config;
    NvVideoEncoder* encoder = nullptr;

    // NV12 staging in device memory; the V4L2 output plane is host-mapped, so
    // each frame costs one device-to-host copy per plane.
    uint8_t* d_y = nullptr;
    uint8_t* d_uv = nullptr;
    size_t d_y_pitch = 0;
    size_t d_uv_pitch = 0;

    std::mutex mutex;
    std::deque<EncodedUnit> ready;
    bool eos_reached = false;

    ~Impl()
    {
        if (d_y)
            cudaFree(d_y);
        if (d_uv)
            cudaFree(d_uv);
        delete encoder;
    }

    // Capture-plane dequeue callback; runs on the encoder's own thread.
    static bool capture_dq(struct v4l2_buffer* v4l2_buf, NvBuffer* buffer, NvBuffer* /*shared*/, void* arg)
    {
        auto* self = static_cast<Impl*>(arg);

        if (!v4l2_buf)
            return false;

        if (buffer && buffer->planes[0].bytesused > 0)
        {
            const auto* data = static_cast<const uint8_t*>(buffer->planes[0].data);
            // V4L2 M2M copies the OUTPUT buffer's timestamp onto the CAPTURE
            // buffer it produces, so this unit carries its own frame's stamp
            // rather than whatever was submitted most recently.
            EncodedUnit unit;
            unit.data.assign(data, data + buffer->planes[0].bytesused);
            unit.timestamp_ns = static_cast<int64_t>(v4l2_buf->timestamp.tv_sec) * 1000000000LL +
                                static_cast<int64_t>(v4l2_buf->timestamp.tv_usec) * 1000LL;
            std::lock_guard<std::mutex> lock(self->mutex);
            self->ready.push_back(std::move(unit));
        }

        // A zero-length unit marks end of stream; stop the thread rather than
        // re-queueing, otherwise the dq thread spins on a finished encoder.
        if (buffer && buffer->planes[0].bytesused == 0)
        {
            std::lock_guard<std::mutex> lock(self->mutex);
            self->eos_reached = true;
            return false;
        }

        if (self->encoder->capture_plane.qBuffer(*v4l2_buf, nullptr) < 0)
            return false;

        return true;
    }
};

JetsonEncoder::JetsonEncoder(const EncoderConfig& config) : m_impl(std::make_unique<Impl>())
{
    if (config.width == 0 || config.height == 0)
        throw std::runtime_error("JetsonEncoder: width/height must be non-zero");
    if ((config.width % 2) != 0 || (config.height % 2) != 0)
        throw std::runtime_error("JetsonEncoder: width/height must be even for NV12");

    m_impl->config = config;

    m_impl->encoder = NvVideoEncoder::createVideoEncoder("enc0");
    if (!m_impl->encoder)
        throw std::runtime_error("JetsonEncoder: createVideoEncoder failed (is /dev/v4l2-nvenc present?)");

    auto* enc = m_impl->encoder;

    // Capture format must be set before the output format.
    const uint32_t bitstream_size = config.width * config.height * 3 / 2;
    check(enc->setCapturePlaneFormat(V4L2_PIX_FMT_H264, config.width, config.height, bitstream_size),
          "setCapturePlaneFormat");
    check(enc->setOutputPlaneFormat(V4L2_PIX_FMT_NV12M, config.width, config.height), "setOutputPlaneFormat");

    check(enc->setBitrate(config.bitrate_bps), "setBitrate");
    check(enc->setProfile(V4L2_MPEG_VIDEO_H264_PROFILE_HIGH), "setProfile");

    // Level is not inferred reliably, and 2560x1984@60 is 1,190,400 macroblocks
    // per second against Level 5.1's ceiling of 983,040 -- 21% over. An unset
    // or too-low level encodes fine here and then gets refused downstream.
    check(enc->setLevel(V4L2_MPEG_VIDEO_H264_LEVEL_5_2), "setLevel");

    if (config.peak_bitrate_bps > config.bitrate_bps)
    {
        check(enc->setRateControlMode(V4L2_MPEG_VIDEO_BITRATE_MODE_VBR), "setRateControlMode");
        check(enc->setPeakBitrate(config.peak_bitrate_bps), "setPeakBitrate");
    }
    else
    {
        check(enc->setRateControlMode(V4L2_MPEG_VIDEO_BITRATE_MODE_CBR), "setRateControlMode");
    }
    check(enc->setFrameRate(config.fps ? config.fps : 30, 1), "setFrameRate");

    const uint32_t gop = config.gop ? config.gop : (config.fps ? config.fps * 5 : 150);
    check(enc->setIDRInterval(gop), "setIDRInterval");
    check(enc->setIFrameInterval(gop), "setIFrameInterval");

    // Low-latency shape: no B-frames, SPS/PPS on every IDR so a receiver can
    // join mid-stream, and the max-performance clock preset.
    check(enc->setNumBFrames(0), "setNumBFrames");
    check(enc->setInsertSpsPpsAtIdrEnabled(true), "setInsertSpsPpsAtIdrEnabled");
    check(enc->setMaxPerfMode(1), "setMaxPerfMode");

    check(enc->output_plane.setupPlane(V4L2_MEMORY_MMAP, kOutputBuffers, true, false), "output setupPlane");
    check(enc->capture_plane.setupPlane(V4L2_MEMORY_MMAP, kCaptureBuffers, true, false), "capture setupPlane");

    check(enc->output_plane.setStreamStatus(true), "output setStreamStatus");
    check(enc->capture_plane.setStreamStatus(true), "capture setStreamStatus");

    enc->capture_plane.setDQThreadCallback(&Impl::capture_dq);
    enc->capture_plane.startDQThread(m_impl.get());

    // Prime the capture plane so the encoder always has somewhere to write.
    for (uint32_t i = 0; i < enc->capture_plane.getNumBuffers(); ++i)
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        std::memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        std::memset(planes, 0, sizeof(planes));
        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;
        check(enc->capture_plane.qBuffer(v4l2_buf, nullptr), "capture qBuffer");
    }

    check_cuda(cudaMallocPitch(reinterpret_cast<void**>(&m_impl->d_y), &m_impl->d_y_pitch, config.width, config.height),
               "cudaMallocPitch(Y)");
    check_cuda(
        cudaMallocPitch(reinterpret_cast<void**>(&m_impl->d_uv), &m_impl->d_uv_pitch, config.width, config.height / 2),
        "cudaMallocPitch(UV)");
}

JetsonEncoder::~JetsonEncoder()
{
    if (m_impl && m_impl->encoder)
    {
        m_impl->encoder->capture_plane.stopDQThread();
        m_impl->encoder->capture_plane.waitForDQThread(1000);
    }
}

bool JetsonEncoder::submit(uintptr_t rgba_device_ptr, std::size_t row_pitch_bytes, int64_t timestamp_ns)
{
    auto* enc = m_impl->encoder;
    const auto& config = m_impl->config;

    launch_rgba_to_nv12(reinterpret_cast<const uint8_t*>(rgba_device_ptr), static_cast<int>(row_pitch_bytes), m_impl->d_y,
                        static_cast<int>(m_impl->d_y_pitch), m_impl->d_uv, static_cast<int>(m_impl->d_uv_pitch),
                        static_cast<int>(config.width), static_cast<int>(config.height), config.full_range, nullptr);
    check_cuda(cudaGetLastError(), "rgba_to_nv12 launch");
    check_cuda(cudaStreamSynchronize(nullptr), "rgba_to_nv12 sync");

    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];
    std::memset(&v4l2_buf, 0, sizeof(v4l2_buf));
    std::memset(planes, 0, sizeof(planes));
    v4l2_buf.m.planes = planes;

    NvBuffer* buffer = nullptr;
    // Until every output buffer has been queued once, index i is free by
    // construction; after that a dequeue is what frees one.
    if (m_queued < enc->output_plane.getNumBuffers())
    {
        buffer = enc->output_plane.getNthBuffer(m_queued);
        v4l2_buf.index = m_queued;
        ++m_queued;
    }
    else if (enc->output_plane.dqBuffer(v4l2_buf, &buffer, nullptr, 0) < 0)
    {
        return false; // encoder still busy; drop this frame rather than block
    }

    check_cuda(cudaMemcpy2D(buffer->planes[0].data, buffer->planes[0].fmt.stride, m_impl->d_y, m_impl->d_y_pitch,
                            config.width, config.height, cudaMemcpyDeviceToHost),
               "cudaMemcpy2D(Y)");
    check_cuda(cudaMemcpy2D(buffer->planes[1].data, buffer->planes[1].fmt.stride, m_impl->d_uv, m_impl->d_uv_pitch,
                            config.width, config.height / 2, cudaMemcpyDeviceToHost),
               "cudaMemcpy2D(UV)");

    buffer->planes[0].bytesused = buffer->planes[0].fmt.stride * config.height;
    buffer->planes[1].bytesused = buffer->planes[1].fmt.stride * (config.height / 2);
    v4l2_buf.m.planes[0].bytesused = buffer->planes[0].bytesused;
    v4l2_buf.m.planes[1].bytesused = buffer->planes[1].bytesused;

    // Microsecond granularity is all a timeval carries; that is well below the
    // frame interval and keeps each unit attributed to its own capture.
    v4l2_buf.timestamp.tv_sec = static_cast<time_t>(timestamp_ns / 1000000000LL);
    v4l2_buf.timestamp.tv_usec = static_cast<suseconds_t>((timestamp_ns % 1000000000LL) / 1000LL);

    check(enc->output_plane.qBuffer(v4l2_buf, nullptr), "output qBuffer");
    return true;
}

EncodedUnit JetsonEncoder::poll()
{
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    if (m_impl->ready.empty())
        return {};

    auto out = std::move(m_impl->ready.front());
    m_impl->ready.pop_front();
    return out;
}

EncodedUnit JetsonEncoder::end_of_stream()
{
    auto* enc = m_impl->encoder;

    // A zero-length output buffer is the EOS marker for the V4L2 encoder.
    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];
    std::memset(&v4l2_buf, 0, sizeof(v4l2_buf));
    std::memset(planes, 0, sizeof(planes));
    v4l2_buf.m.planes = planes;

    NvBuffer* buffer = nullptr;
    if (enc->output_plane.dqBuffer(v4l2_buf, &buffer, nullptr, 10) >= 0)
    {
        v4l2_buf.m.planes[0].bytesused = 0;
        v4l2_buf.m.planes[1].bytesused = 0;
        enc->output_plane.qBuffer(v4l2_buf, nullptr);
    }

    enc->capture_plane.waitForDQThread(2000);

    EncodedUnit out;
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    // The tail is concatenated, so it carries the stamp of the last unit in it.
    for (auto& chunk : m_impl->ready)
    {
        out.data.insert(out.data.end(), chunk.data.begin(), chunk.data.end());
        out.timestamp_ns = chunk.timestamp_ns;
    }
    m_impl->ready.clear();
    return out;
}

} // namespace sensing
} // namespace plugins

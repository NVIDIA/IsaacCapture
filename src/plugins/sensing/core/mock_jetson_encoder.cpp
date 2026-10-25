// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Mock backend for JetsonEncoder: real H.264 from the desktop GPU's NVENC, through libavcodec's
// h264_nvenc, set up as the Jetson V4L2 encoder is (High profile, Level 5.2, VBR with a peak, an IDR
// every GOP with SPS/PPS in front of it, no B-frames). It runs the real RGBA-to-NV12 kernel and the same
// device-to-host copy as the V4L2 path, then a worker thread feeds the NV12 buffer to the encoder.
//
// Reproduced from the real encoder: a fixed pool of input buffers, so submit() returns false when the
// encoder falls behind rather than blocking; the capture stamp coming back at microsecond resolution (a
// V4L2 timeval); end_of_stream() returning the remaining tail as one concatenated unit. NVENC's own
// latency and unit sizes are what they are, and are not those of the Jetson engine.

#include "jetson_encoder.hpp"
#include "rgba_to_nv12.cuh"

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/opt.h>
}

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <cuda_runtime.h>
#include <deque>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace plugins
{
namespace sensing
{

namespace
{

constexpr uint32_t kInputBuffers = 6;

void check_cuda(cudaError_t err, const char* what)
{
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("mock JetsonEncoder: ") + what + ": " + cudaGetErrorString(err));
}

void check_av(int err, const char* what)
{
    if (err < 0)
    {
        char text[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(err, text, sizeof(text));
        throw std::runtime_error(std::string("mock JetsonEncoder: ") + what + ": " + text);
    }
}

} // namespace

struct JetsonEncoder::Impl
{
    struct Job
    {
        uint32_t buffer = 0;
        int64_t timestamp_ns = 0;
        bool keyframe = false;
    };

    explicit Impl(const EncoderConfig& cfg) : config(cfg)
    {
        codec = avcodec_find_encoder_by_name("h264_nvenc");
        if (!codec)
            throw std::runtime_error("mock JetsonEncoder: libavcodec has no h264_nvenc encoder");

        // A constructor that throws never runs the destructor, so free what was allocated here.
        try
        {
            open_context(config.bitrate_bps);
            alloc_frame();
            check_cuda(cudaMallocPitch(reinterpret_cast<void**>(&d_y), &d_y_pitch, config.width, config.height),
                       "cudaMallocPitch(Y)");
            check_cuda(cudaMallocPitch(reinterpret_cast<void**>(&d_uv), &d_uv_pitch, config.width, config.height / 2),
                       "cudaMallocPitch(UV)");

            // Host staging, standing in for the V4L2 output plane's mapped buffers.
            const size_t nv12_bytes = static_cast<size_t>(config.width) * config.height * 3 / 2;
            for (uint32_t i = 0; i < kInputBuffers; ++i)
            {
                host_buffers.emplace_back(nv12_bytes);
                free_buffers.push_back(i);
            }
            worker = std::thread([this] { run(); });
        }
        catch (...)
        {
            release();
            throw;
        }
    }

    ~Impl()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            closing = true;
        }
        wake.notify_all();
        if (worker.joinable())
            worker.join();
        release();
    }

    void set_option(const char* name, const char* value)
    {
        check_av(av_opt_set(context->priv_data, name, value, 0), (std::string("av_opt_set ") + name).c_str());
    }

    // A fresh encoder session at `bitrate`: its first frame is an IDR with parameter sets.
    void open_context(uint32_t bitrate)
    {
        avcodec_free_context(&context);
        context = avcodec_alloc_context3(codec);
        if (!context)
            throw std::runtime_error("mock JetsonEncoder: cannot allocate the encoder context");

        const uint32_t fps = config.fps ? config.fps : 30;
        const uint32_t gop = config.gop ? config.gop : fps * 5;
        context->width = static_cast<int>(config.width);
        context->height = static_cast<int>(config.height);
        context->pix_fmt = AV_PIX_FMT_NV12;
        // Microseconds, as a V4L2 timeval carries them.
        context->time_base = { 1, 1'000'000 };
        context->framerate = { static_cast<int>(fps), 1 };
        context->bit_rate = bitrate;
        context->gop_size = static_cast<int>(gop);
        context->max_b_frames = 0;
        // Parameter sets stay in the stream, in front of every IDR, so a receiver can join mid-stream.
        context->flags &= ~AV_CODEC_FLAG_GLOBAL_HEADER;
        set_option("profile", "high");
        set_option("level", "5.2");
        set_option("forced-idr", "1");
        set_option("zerolatency", "1");
        // Hand each unit back as soon as it is encoded rather than after a queue of frames.
        set_option("delay", "0");
        if (config.peak_bitrate_bps > config.bitrate_bps)
        {
            // The configured ratio between the ceiling and the target holds at any target.
            context->rc_max_rate = static_cast<int64_t>(bitrate) * config.peak_bitrate_bps / config.bitrate_bps;
            context->rc_buffer_size = static_cast<int>(context->rc_max_rate);
            set_option("rc", "vbr");
        }
        else
        {
            set_option("rc", "cbr");
        }
        check_av(avcodec_open2(context, codec, nullptr), "cannot open h264_nvenc");
        applied_bitrate = bitrate;
    }

    void alloc_frame()
    {
        frame = av_frame_alloc();
        packet = av_packet_alloc();
        if (!frame || !packet)
            throw std::runtime_error("mock JetsonEncoder: cannot allocate a frame or packet");
        frame->format = AV_PIX_FMT_NV12;
        frame->width = context->width;
        frame->height = context->height;
        check_av(av_frame_get_buffer(frame, 0), "cannot allocate the input frame");
    }

    void release()
    {
        cudaFree(d_y);
        cudaFree(d_uv);
        av_packet_free(&packet);
        av_frame_free(&frame);
        avcodec_free_context(&context);
    }

    // Moves every packet the encoder has ready into `out`. Worker thread, or end_of_stream() once the
    // worker is idle.
    void collect(std::deque<EncodedUnit>& out)
    {
        for (;;)
        {
            const int err = avcodec_receive_packet(context, packet);
            if (err == AVERROR(EAGAIN) || err == AVERROR_EOF)
                return;
            check_av(err, "avcodec_receive_packet");
            EncodedUnit unit;
            unit.data.assign(packet->data, packet->data + packet->size);
            unit.timestamp_ns = packet->pts * 1000;
            unit.keyframe = (packet->flags & AV_PKT_FLAG_KEY) != 0;
            av_packet_unref(packet);
            out.push_back(std::move(unit));
        }
    }

    // Worker thread, which owns the codec. The libavcodec NVENC wrapper takes its rate control at
    // open, so a new target drains the running session into `out` and opens another at that rate: the
    // stream continues from an IDR with parameter sets.
    void apply_bitrate(std::deque<EncodedUnit>& out)
    {
        uint32_t target = 0;
        {
            std::lock_guard<std::mutex> lock(mutex);
            target = pending_bitrate;
            pending_bitrate = 0;
        }
        if (target == 0 || target == applied_bitrate)
            return;
        check_av(avcodec_send_frame(context, nullptr), "avcodec_send_frame(flush)");
        collect(out);
        open_context(target);
    }

    void encode(const Job& job)
    {
        check_av(av_frame_make_writable(frame), "av_frame_make_writable");
        const uint8_t* nv12 = host_buffers[job.buffer].data();
        const size_t width = config.width;
        for (uint32_t row = 0; row < config.height; ++row)
            std::memcpy(frame->data[0] + static_cast<size_t>(row) * frame->linesize[0], nv12 + row * width, width);
        const uint8_t* uv = nv12 + width * config.height;
        for (uint32_t row = 0; row < config.height / 2; ++row)
            std::memcpy(frame->data[1] + static_cast<size_t>(row) * frame->linesize[1], uv + row * width, width);
        frame->pts = job.timestamp_ns / 1000;
        // With forced-idr, an I picture is an IDR with its parameter sets.
        frame->pict_type = job.keyframe ? AV_PICTURE_TYPE_I : AV_PICTURE_TYPE_NONE;
        check_av(avcodec_send_frame(context, frame), "avcodec_send_frame");
    }

    void run()
    {
        for (;;)
        {
            Job job;
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [&] { return closing || !jobs.empty(); });
                // Frames still queued at teardown are abandoned, as the hardware encoder would.
                if (closing || jobs.empty())
                    return;
                job = jobs.front();
                jobs.pop_front();
                busy = true;
            }

            std::deque<EncodedUnit> produced;
            std::string error;
            try
            {
                apply_bitrate(produced);
                encode(job);
                collect(produced);
            }
            catch (const std::exception& e)
            {
                error = e.what();
            }

            {
                std::lock_guard<std::mutex> lock(mutex);
                free_buffers.push_back(job.buffer);
                for (auto& unit : produced)
                    ready.push_back(std::move(unit));
                if (!error.empty() && failure.empty())
                    failure = error;
                busy = false;
            }
            idle.notify_all();
        }
    }

    EncoderConfig config;

    const AVCodec* codec = nullptr;
    AVCodecContext* context = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* packet = nullptr;

    uint8_t* d_y = nullptr;
    uint8_t* d_uv = nullptr;
    size_t d_y_pitch = 0;
    size_t d_uv_pitch = 0;

    std::vector<std::vector<uint8_t>> host_buffers;

    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable idle;
    std::deque<uint32_t> free_buffers;
    std::deque<Job> jobs;
    std::deque<EncodedUnit> ready; // what poll() hands out
    std::string failure;
    bool busy = false;
    bool closing = false;
    bool flushed = false;
    bool keyframe_requested = false; // guarded by `mutex`
    uint32_t pending_bitrate = 0; // guarded by `mutex`; 0 when nothing is waiting
    uint32_t applied_bitrate = 0; // worker thread only
    std::thread worker;
};

JetsonEncoder::JetsonEncoder(const EncoderConfig& config)
{
    if (config.width == 0 || config.height == 0)
        throw std::runtime_error("JetsonEncoder: width/height must be non-zero");
    if ((config.width % 2) != 0 || (config.height % 2) != 0)
        throw std::runtime_error("JetsonEncoder: width/height must be even for NV12");
    m_impl = std::make_unique<Impl>(config);
}

JetsonEncoder::~JetsonEncoder() = default;

bool JetsonEncoder::submit(uintptr_t rgba_device_ptr, std::size_t row_pitch_bytes, int64_t timestamp_ns)
{
    auto& impl = *m_impl;
    const auto& config = impl.config;

    uint32_t buffer = 0;
    {
        std::lock_guard<std::mutex> lock(impl.mutex);
        if (!impl.failure.empty())
            throw std::runtime_error(impl.failure);
        if (impl.free_buffers.empty())
            return false; // encoder still busy; drop this frame rather than block
        buffer = impl.free_buffers.front();
        impl.free_buffers.pop_front();
    }

    launch_rgba_to_nv12(reinterpret_cast<const uint8_t*>(rgba_device_ptr), static_cast<int>(row_pitch_bytes), impl.d_y,
                        static_cast<int>(impl.d_y_pitch), impl.d_uv, static_cast<int>(impl.d_uv_pitch),
                        static_cast<int>(config.width), static_cast<int>(config.height), config.full_range, nullptr);
    check_cuda(cudaGetLastError(), "rgba_to_nv12 launch");
    check_cuda(cudaStreamSynchronize(nullptr), "rgba_to_nv12 sync");

    uint8_t* host = impl.host_buffers[buffer].data();
    check_cuda(
        cudaMemcpy2D(host, config.width, impl.d_y, impl.d_y_pitch, config.width, config.height, cudaMemcpyDeviceToHost),
        "cudaMemcpy2D(Y)");
    check_cuda(cudaMemcpy2D(host + static_cast<size_t>(config.width) * config.height, config.width, impl.d_uv,
                            impl.d_uv_pitch, config.width, config.height / 2, cudaMemcpyDeviceToHost),
               "cudaMemcpy2D(UV)");

    {
        std::lock_guard<std::mutex> lock(impl.mutex);
        impl.jobs.push_back({ buffer, timestamp_ns, impl.keyframe_requested });
        impl.keyframe_requested = false;
    }
    impl.wake.notify_one();
    return true;
}

EncodedUnit JetsonEncoder::poll()
{
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    if (!m_impl->failure.empty())
        throw std::runtime_error(m_impl->failure);
    if (m_impl->ready.empty())
        return {};
    auto out = std::move(m_impl->ready.front());
    m_impl->ready.pop_front();
    return out;
}

std::vector<EncodedUnit> JetsonEncoder::drain()
{
    auto& impl = *m_impl;
    std::unique_lock<std::mutex> lock(impl.mutex);
    impl.idle.wait(lock, [&] { return impl.jobs.empty() && !impl.busy; });
    if (!impl.failure.empty())
        throw std::runtime_error(impl.failure);
    if (!impl.flushed)
    {
        impl.flushed = true;
        check_av(avcodec_send_frame(impl.context, nullptr), "avcodec_send_frame(flush)");
        impl.collect(impl.ready);
    }
    std::vector<EncodedUnit> units(
        std::make_move_iterator(impl.ready.begin()), std::make_move_iterator(impl.ready.end()));
    impl.ready.clear();
    return units;
}

EncodedUnit JetsonEncoder::end_of_stream()
{
    // The tail is concatenated, so it carries the stamp of the last unit in it.
    EncodedUnit out;
    for (auto& chunk : drain())
    {
        out.data.insert(out.data.end(), chunk.data.begin(), chunk.data.end());
        out.timestamp_ns = chunk.timestamp_ns;
    }
    return out;
}

void JetsonEncoder::request_keyframe()
{
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    m_impl->keyframe_requested = true;
}

bool JetsonEncoder::set_bitrate(uint32_t bitrate_bps)
{
    if (bitrate_bps == 0)
        return false;
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    m_impl->pending_bitrate = bitrate_bps;
    return true;
}

const char* JetsonEncoder::backend_name()
{
    return "NVENC (libavcodec h264_nvenc)";
}

} // namespace sensing
} // namespace plugins

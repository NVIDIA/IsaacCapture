// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "livekit_video_output.hpp"

#include "rgba_box_downscale.cuh"
#include "rgba_to_nv12.cuh"

#include <atomic>
#include <condition_variable>
#include <cuda_runtime.h>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace plugins
{
namespace sensing
{

namespace
{

void check_cuda(cudaError_t err, const char* what)
{
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("LiveKit output: ") + what + ": " + cudaGetErrorString(err));
}

// The track's colours are BT.601 limited range: that is what the video pipeline assumes when it
// decodes NV12, so it is not a per-sensor setting.
constexpr bool kFullRange = false;

} // namespace

struct LiveKitVideoOutput::Impl
{
    struct Slot
    {
        uint8_t* ptr = nullptr;
        size_t pitch = 0;
        uint64_t timestamp_ns = 0;
        uint64_t capture_tsc_ns = 0;
        uint64_t sequence = 0;
    };

    Impl(std::unique_ptr<VideoTrackSink> track, const LiveKitOutputConfig& cfg) : sink(std::move(track)), config(cfg)
    {
        validate_geometry(config.source_width, config.source_height, config.scale_div);
        if (config.fps == 0)
            throw std::runtime_error("LiveKit output: the publication rate must be at least 1 fps");
        out_width = config.source_width / config.scale_div;
        out_height = config.source_height / config.scale_div;
        if (sink->width() != out_width || sink->height() != out_height)
            throw std::runtime_error("LiveKit output: the track size does not match the output size");
        period_ns = 1'000'000'000ull / config.fps;

        // A constructor that throws never runs the destructor, so free what was allocated here.
        try
        {
            check_cuda(cudaSetDevice(config.gpu_id), "cudaSetDevice");
            for (auto& slot : slots)
            {
                check_cuda(cudaMallocPitch(reinterpret_cast<void**>(&slot.ptr), &slot.pitch,
                                           static_cast<size_t>(config.source_width) * 4, config.source_height),
                           "cudaMallocPitch(snapshot)");
            }
            if (config.scale_div > 1)
            {
                check_cuda(cudaMallocPitch(reinterpret_cast<void**>(&d_scaled), &scaled_pitch,
                                           static_cast<size_t>(out_width) * 4, out_height),
                           "cudaMallocPitch(reduced)");
            }
            check_cuda(
                cudaMallocPitch(reinterpret_cast<void**>(&d_y), &y_pitch, out_width, out_height), "cudaMallocPitch(Y)");
            check_cuda(cudaMallocPitch(reinterpret_cast<void**>(&d_uv), &uv_pitch, out_width, out_height / 2),
                       "cudaMallocPitch(UV)");
            check_cuda(cudaStreamCreateWithFlags(&snapshot_stream, cudaStreamNonBlocking), "cudaStreamCreate");
            check_cuda(cudaStreamCreateWithFlags(&work_stream, cudaStreamNonBlocking), "cudaStreamCreate");

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
            stopping = true;
        }
        wake.notify_all();
        if (worker.joinable())
            worker.join();
        release();
    }

    void release()
    {
        for (auto& slot : slots)
            cudaFree(slot.ptr);
        cudaFree(d_scaled);
        cudaFree(d_y);
        cudaFree(d_uv);
        if (snapshot_stream)
            cudaStreamDestroy(snapshot_stream);
        if (work_stream)
            cudaStreamDestroy(work_stream);
    }

    void fail(const std::string& message)
    {
        std::lock_guard<std::mutex> lock(failure_mutex);
        if (!failed_flag.load())
        {
            failure_message = message;
            failed_flag.store(true);
        }
    }

    // Polling thread.
    void submit(const FrameView& view)
    {
        ++offered;
        if (failed_flag.load())
            return;

        // Pace on the capture clock, so poll jitter does not drop a frame that is due. A tenth of a
        // period of slack lets 60 fps capture feed a 30 fps track every second capture.
        if (have_last && view.timestamp_ns < last_accepted_ns + period_ns * 9 / 10)
        {
            ++paced_out;
            return;
        }

        try
        {
            std::lock_guard<std::mutex> lock(mutex);
            int target = -1;
            for (int i = 0; i < 2; ++i)
            {
                if (i != inflight && i != pending)
                {
                    target = i;
                    break;
                }
            }
            if (target < 0)
            {
                // The worker holds one slot and the other waits: the newer capture takes the waiting one.
                target = pending;
                ++replaced;
            }

            // Complete before returning: the camera may rewrite its slot at the second following poll.
            Slot& slot = slots[target];
            check_cuda(cudaMemcpy2DAsync(slot.ptr, slot.pitch, reinterpret_cast<const void*>(view.ptr), view.pitch,
                                         static_cast<size_t>(config.source_width) * 4, config.source_height,
                                         cudaMemcpyDeviceToDevice, snapshot_stream),
                       "snapshot copy");
            check_cuda(cudaStreamSynchronize(snapshot_stream), "snapshot sync");

            slot.timestamp_ns = view.timestamp_ns;
            slot.capture_tsc_ns = view.capture_tsc_ns;
            slot.sequence = view.sequence;
            pending = target;
        }
        catch (const std::exception& e)
        {
            fail(e.what());
            return;
        }
        ++snapshots;
        have_last = true;
        last_accepted_ns = view.timestamp_ns;
        wake.notify_one();
    }

    // Worker thread.
    void run()
    {
        try
        {
            check_cuda(cudaSetDevice(config.gpu_id), "cudaSetDevice(worker)");
            for (;;)
            {
                int index = -1;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    wake.wait(lock, [&] { return stopping || pending >= 0; });
                    if (stopping)
                        return;
                    index = pending;
                    pending = -1;
                    inflight = index;
                }

                publish(slots[index]);

                {
                    std::lock_guard<std::mutex> lock(mutex);
                    inflight = -1;
                }
                ++published;
            }
        }
        catch (const std::exception& e)
        {
            fail(e.what());
        }
    }

    void publish(const Slot& slot)
    {
        const uint8_t* rgba = slot.ptr;
        size_t pitch = slot.pitch;
        if (config.scale_div > 1)
        {
            launch_rgba_box_downscale(rgba, static_cast<int>(pitch), d_scaled, static_cast<int>(scaled_pitch),
                                      static_cast<int>(out_width), static_cast<int>(out_height),
                                      static_cast<int>(config.scale_div), work_stream);
            rgba = d_scaled;
            pitch = scaled_pitch;
        }
        launch_rgba_to_nv12(rgba, static_cast<int>(pitch), d_y, static_cast<int>(y_pitch), d_uv,
                            static_cast<int>(uv_pitch), static_cast<int>(out_width), static_cast<int>(out_height),
                            kFullRange, work_stream);
        check_cuda(cudaGetLastError(), "conversion kernels");

        // Straight into the track's buffer: tightly packed rows, luma then chroma.
        uint8_t* host = sink->frame_buffer();
        check_cuda(
            cudaMemcpy2DAsync(host, out_width, d_y, y_pitch, out_width, out_height, cudaMemcpyDeviceToHost, work_stream),
            "luma copy");
        check_cuda(cudaMemcpy2DAsync(host + static_cast<size_t>(out_width) * out_height, out_width, d_uv, uv_pitch,
                                     out_width, out_height / 2, cudaMemcpyDeviceToHost, work_stream),
                   "chroma copy");
        check_cuda(cudaStreamSynchronize(work_stream), "frame copy");

        VideoFrameInfo info;
        info.timestamp_us = slot.timestamp_ns / 1000;
        info.frame_id = static_cast<uint32_t>(slot.sequence);
        info.capture_tsc_ns = slot.capture_tsc_ns;
        sink->submit(info);
    }

    std::unique_ptr<VideoTrackSink> sink;
    LiveKitOutputConfig config;
    uint32_t out_width = 0;
    uint32_t out_height = 0;
    uint64_t period_ns = 0;

    Slot slots[2];
    uint8_t* d_scaled = nullptr;
    size_t scaled_pitch = 0;
    uint8_t* d_y = nullptr;
    uint8_t* d_uv = nullptr;
    size_t y_pitch = 0;
    size_t uv_pitch = 0;
    cudaStream_t snapshot_stream = nullptr;
    cudaStream_t work_stream = nullptr;

    // Guards pending, inflight, stopping and the snapshot slots' contents.
    std::mutex mutex;
    std::condition_variable wake;
    int pending = -1;
    int inflight = -1;
    bool stopping = false;
    std::thread worker;

    // Polling thread only.
    bool have_last = false;
    uint64_t last_accepted_ns = 0;

    std::atomic<uint64_t> offered{ 0 };
    std::atomic<uint64_t> paced_out{ 0 };
    std::atomic<uint64_t> replaced{ 0 };
    std::atomic<uint64_t> snapshots{ 0 };
    std::atomic<uint64_t> published{ 0 };

    std::mutex failure_mutex;
    std::string failure_message;
    std::atomic<bool> failed_flag{ false };
};

void LiveKitVideoOutput::validate_geometry(uint32_t width, uint32_t height, uint32_t divisor)
{
    if (divisor == 0)
        throw std::runtime_error("LiveKit output: the scale divisor must be at least 1");
    if (width == 0 || height == 0)
        throw std::runtime_error("LiveKit output: the source size is empty");
    if (width % divisor != 0 || height % divisor != 0)
    {
        throw std::runtime_error("LiveKit output: divisor " + std::to_string(divisor) + " does not divide " +
                                 std::to_string(width) + "x" + std::to_string(height) + " exactly");
    }
    if ((width / divisor) % 2 != 0 || (height / divisor) % 2 != 0)
    {
        throw std::runtime_error("LiveKit output: " + std::to_string(width / divisor) + "x" +
                                 std::to_string(height / divisor) + " is not even; NV12 needs even dimensions");
    }
}

LiveKitVideoOutput::LiveKitVideoOutput(std::unique_ptr<VideoTrackSink> sink, const LiveKitOutputConfig& config)
    : m_impl(std::make_unique<Impl>(std::move(sink), config))
{
}

LiveKitVideoOutput::~LiveKitVideoOutput() = default;

void LiveKitVideoOutput::submit(const FrameView& view)
{
    m_impl->submit(view);
}

LiveKitOutputStats LiveKitVideoOutput::stats() const
{
    LiveKitOutputStats s;
    s.offered = m_impl->offered.load();
    s.paced_out = m_impl->paced_out.load();
    s.replaced = m_impl->replaced.load();
    s.snapshots = m_impl->snapshots.load();
    s.published = m_impl->published.load();
    return s;
}

bool LiveKitVideoOutput::failed() const
{
    return m_impl->failed_flag.load();
}

std::string LiveKitVideoOutput::failure() const
{
    std::lock_guard<std::mutex> lock(m_impl->failure_mutex);
    return m_impl->failure_message;
}

std::string LiveKitVideoOutput::describe()
{
    return m_impl->sink->describe();
}

} // namespace sensing
} // namespace plugins

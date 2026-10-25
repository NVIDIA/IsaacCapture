// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "livekit_video_output.hpp"

#include "jetson_encoder.hpp"
#include "rgba_box_downscale.cuh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cuda_runtime.h>
#include <deque>
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

// Congestion control never drives the encoder below this.
constexpr uint32_t kMinBitrateBps = 100'000;
// A target is applied only when it differs from the current one by this fraction, and this long after
// the last change.
constexpr double kMinBitrateChange = 0.25;
constexpr std::chrono::seconds kMinBitrateInterval{ 3 };

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
        if (config.bitrate_bps < kMinBitrateBps)
        {
            throw std::runtime_error("LiveKit output: the bitrate must be at least " + std::to_string(kMinBitrateBps) +
                                     " bps");
        }
        // Congestion control moves the target between the floor and the ceiling; the configured target
        // is never below the floor, so the range is never inverted.
        max_bitrate = std::max(config.bitrate_bps, config.peak_bitrate_bps);

        // A constructor that throws never runs the destructor, so free what was allocated here.
        try
        {
            check_cuda(cudaSetDevice(config.gpu_id), "cudaSetDevice");

            // First, so a machine without a usable hardware encoder fails before anything is allocated.
            EncoderConfig encoder_config;
            encoder_config.width = out_width;
            encoder_config.height = out_height;
            encoder_config.bitrate_bps = config.bitrate_bps;
            encoder_config.peak_bitrate_bps = config.peak_bitrate_bps;
            encoder_config.fps = config.fps;
            encoder_config.gop = config.gop;
            encoder_config.full_range = config.full_range;
            encoder = std::make_unique<JetsonEncoder>(encoder_config);
            applied_bitrate = config.bitrate_bps;

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
            }
            // A capture still waiting is overwritten by this newer one.
            if (pending >= 0)
                ++replaced;

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
                bool flush_now = false;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    // Frames in the encoder come out a little later: poll for them soon, and otherwise
                    // wake now and then to pick up the receiving side's requests.
                    const auto wait = in_encoder.empty() ? std::chrono::milliseconds(100) : std::chrono::milliseconds(2);
                    wake.wait_for(lock, wait, [&] { return stopping || pending >= 0 || flush_requested; });
                    if (stopping)
                        return;
                    index = pending;
                    pending = -1;
                    inflight = index;
                    flush_now = flush_requested;
                }

                apply_feedback();
                if (index >= 0)
                {
                    encode(slots[index]);
                    std::lock_guard<std::mutex> lock(mutex);
                    inflight = -1;
                }
                if (flush_now)
                {
                    for (auto& unit : encoder->drain())
                        publish_unit(unit);
                    std::lock_guard<std::mutex> lock(mutex);
                    flush_requested = false;
                    flushed.notify_all();
                }
                else
                {
                    for (auto unit = encoder->poll(); !unit.empty(); unit = encoder->poll())
                        publish_unit(unit);
                }
            }
        }
        catch (const std::exception& e)
        {
            fail(e.what());
            std::lock_guard<std::mutex> lock(mutex);
            flushed.notify_all();
        }
    }

    // Worker thread: what the receiving side asked for since the last frame.
    void apply_feedback()
    {
        const EncoderFeedback feedback = sink->take_feedback();
        if (feedback.keyframe_requested)
        {
            ++keyframes_requested;
            keyframe_wanted = true;
        }
        if (feedback.target_bitrate_bps)
        {
            const uint32_t target = std::clamp(*feedback.target_bitrate_bps, kMinBitrateBps, max_bitrate);
            // Congestion control re-estimates all the time. Follow it only for a real change, and not
            // more than every few seconds: on the mock a change reopens the encoder session.
            const auto now = std::chrono::steady_clock::now();
            const double change = std::abs(static_cast<double>(target) - applied_bitrate) / applied_bitrate;
            if (change >= kMinBitrateChange && now - last_bitrate_change >= kMinBitrateInterval &&
                encoder->set_bitrate(target))
            {
                applied_bitrate = target;
                last_bitrate_change = now;
            }
        }
    }

    // Worker thread.
    void encode(const Slot& slot)
    {
        const uint8_t* rgba = slot.ptr;
        size_t pitch = slot.pitch;
        if (config.scale_div > 1)
        {
            launch_rgba_box_downscale(rgba, static_cast<int>(pitch), d_scaled, static_cast<int>(scaled_pitch),
                                      static_cast<int>(out_width), static_cast<int>(out_height),
                                      static_cast<int>(config.scale_div), work_stream);
            check_cuda(cudaGetLastError(), "reduction kernel");
            // The encoder reads on the default stream, which a non-blocking stream does not order against.
            check_cuda(cudaStreamSynchronize(work_stream), "reduction sync");
            rgba = d_scaled;
            pitch = scaled_pitch;
        }

        if (keyframe_wanted)
        {
            encoder->request_keyframe();
            keyframe_wanted = false;
        }
        const int64_t stamp_ns = static_cast<int64_t>(slot.timestamp_ns);
        if (!encoder->submit(reinterpret_cast<uintptr_t>(rgba), pitch, stamp_ns))
        {
            // No input buffer: dropped before encoding, so no reference is lost. A keyframe already
            // requested of the encoder applies to the next frame it takes.
            ++encoder_busy;
            return;
        }
        in_encoder.push_back({ stamp_ns / 1000, static_cast<uint32_t>(slot.sequence), slot.capture_tsc_ns });
    }

    // Worker thread: one encoded unit becomes one published frame, with the sequence and capture time
    // of the capture it came from.
    void publish_unit(const EncodedUnit& unit)
    {
        const int64_t stamp_us = unit.timestamp_ns / 1000;
        // The encoder keeps submission order; a frame it never returned was dropped inside it.
        while (!in_encoder.empty() && in_encoder.front().stamp_us < stamp_us)
            in_encoder.pop_front();
        if (in_encoder.empty() || in_encoder.front().stamp_us != stamp_us)
            throw std::runtime_error("LiveKit output: the encoder returned a unit for a frame it was not given");
        const FrameMeta meta = in_encoder.front();
        in_encoder.pop_front();

        // After a refusal the stream is only usable from its next keyframe.
        if (awaiting_keyframe && !unit.keyframe)
            return;

        EncodedVideoFrame frame;
        frame.data = unit.data.data();
        frame.size = unit.data.size();
        frame.keyframe = unit.keyframe;
        frame.timestamp_us = static_cast<uint64_t>(stamp_us);
        frame.frame_id = meta.sequence;
        frame.capture_tsc_ns = meta.capture_tsc_ns;
        if (sink->submit(frame))
        {
            awaiting_keyframe = false;
            ++published;
        }
        else
        {
            ++rejected;
            awaiting_keyframe = true;
            keyframe_wanted = true;
        }
    }

    std::unique_ptr<VideoTrackSink> sink;
    LiveKitOutputConfig config;
    uint32_t out_width = 0;
    uint32_t out_height = 0;
    uint64_t period_ns = 0;

    struct FrameMeta
    {
        int64_t stamp_us = 0;
        uint32_t sequence = 0;
        uint64_t capture_tsc_ns = 0;
    };

    std::unique_ptr<JetsonEncoder> encoder;
    uint32_t max_bitrate = 0;

    // Worker thread only.
    std::deque<FrameMeta> in_encoder; ///< Submitted to the encoder, no unit back yet.
    bool keyframe_wanted = false;
    bool awaiting_keyframe = false;
    uint32_t applied_bitrate = 0;
    std::chrono::steady_clock::time_point last_bitrate_change = std::chrono::steady_clock::now();

    Slot slots[2];
    uint8_t* d_scaled = nullptr;
    size_t scaled_pitch = 0;
    cudaStream_t snapshot_stream = nullptr;
    cudaStream_t work_stream = nullptr;

    // Guards pending, inflight, stopping and the snapshot slots' contents.
    std::mutex mutex;
    std::condition_variable wake;
    int pending = -1;
    int inflight = -1;
    bool stopping = false;
    bool flush_requested = false;
    std::condition_variable flushed;
    std::thread worker;

    // Polling thread only.
    bool have_last = false;
    uint64_t last_accepted_ns = 0;

    std::atomic<uint64_t> offered{ 0 };
    std::atomic<uint64_t> paced_out{ 0 };
    std::atomic<uint64_t> replaced{ 0 };
    std::atomic<uint64_t> snapshots{ 0 };
    std::atomic<uint64_t> published{ 0 };
    std::atomic<uint64_t> encoder_busy{ 0 };
    std::atomic<uint64_t> rejected{ 0 };
    std::atomic<uint64_t> keyframes_requested{ 0 };

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
    s.encoder_busy = m_impl->encoder_busy.load();
    s.rejected = m_impl->rejected.load();
    s.keyframes_requested = m_impl->keyframes_requested.load();
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
    std::string text = std::string(JetsonEncoder::backend_name()) + ", pre-encoded passthrough";
    const std::string transport = m_impl->sink->describe();
    if (!transport.empty())
        text += "; transport: " + transport;
    return text;
}

void LiveKitVideoOutput::flush()
{
    std::unique_lock<std::mutex> lock(m_impl->mutex);
    m_impl->flush_requested = true;
    m_impl->wake.notify_one();
    m_impl->flushed.wait(lock, [&] { return !m_impl->flush_requested || m_impl->failed_flag.load(); });
}

} // namespace sensing
} // namespace plugins

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "stream_worker.hpp"

#include <gpu_ops/box_downscale.hpp>
#include <ipc_client/cuda_ipc_client.hpp>
#include <nvenc/nvenc_encoder.hpp>
#include <video_types/video_types.hpp>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <cuda_runtime_api.h>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace camera_bridge
{

namespace
{

struct PlaneLayout
{
    size_t y_offset = 0;
    size_t y_pitch = 0;
    size_t uv_offset = 0;
    size_t uv_pitch = 0;
};

PlaneLayout read_layout(const IpcStream& stream)
{
    PlaneLayout layout;
    bool have_y = false;
    bool have_uv = false;
    for (const auto& plane : stream.layout.value("planes", nlohmann::json::array()))
    {
        const std::string name = plane.value("name", std::string());
        if (name == "y")
        {
            layout.y_offset = plane.at("offset").get<size_t>();
            layout.y_pitch = plane.at("pitch").get<size_t>();
            have_y = true;
        }
        else if (name == "uv")
        {
            layout.uv_offset = plane.at("offset").get<size_t>();
            layout.uv_pitch = plane.at("pitch").get<size_t>();
            have_uv = true;
        }
    }
    if (stream.pixel_format != "nv12" || !have_y || !have_uv)
        throw std::runtime_error("stream '" + stream.name + "' is not NV12 with y and uv planes; only NV12 is supported");
    return layout;
}

constexpr uint32_t kStartBitrateBps = 2'000'000;

uint64_t unsigned_field(const nlohmann::json& metadata, const char* key)
{
    const auto it = metadata.find(key);
    return (it != metadata.end() && it->is_number_unsigned()) ? it->get<uint64_t>() : 0;
}

} // namespace

struct StreamWorker::Impl
{
    Impl(const WorkerConfig& cfg, CudaContext& ctx, LiveKitSession& session)
        : config(cfg), context(ctx), client(cfg.socket_path, { cfg.stream })
    {
        client.connect();
        if (client.streams().size() != 1)
            throw std::runtime_error("the IPC publisher has no stream named '" + cfg.stream + "'");
        const IpcStream& stream = client.streams().front();
        layout = read_layout(stream);
        source = { static_cast<uint32_t>(stream.width), static_cast<uint32_t>(stream.height) };
        if (!box_downscale_supported(source.width, source.height, cfg.scale_div))
            throw std::runtime_error("stream '" + cfg.stream + "' is " + std::to_string(source.width) + "x" +
                                     std::to_string(source.height) + " and cannot be shrunk by " +
                                     std::to_string(cfg.scale_div) + " to an even size");
        geometry = { source.width / cfg.scale_div, source.height / cfg.scale_div };

        PublishSpec spec;
        spec.name = cfg.track;
        spec.width = geometry.width;
        spec.height = geometry.height;
        spec.max_bitrate_bps = cfg.bitrate_bps;
        spec.max_fps = cfg.fps;
        sink = session.publish_video(spec);
    }

    ~Impl()
    {
        running = false;
        if (thread.joinable())
            thread.join();
        if (copy_stream != nullptr)
            cudaStreamDestroy(copy_stream);
    }

    void run(WorkerStats& stats)
    {
        try
        {
            loop(stats);
        }
        catch (const std::exception& error)
        {
            std::lock_guard<std::mutex> lock(failure_mutex);
            failure = error.what();
        }
    }

    void loop(WorkerStats& stats)
    {
        context.make_current();
        if (cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking) != cudaSuccess)
            throw std::runtime_error("cannot create a CUDA stream");

        EncoderConfig encoder_config;
        encoder_config.geometry = geometry;
        encoder_config.fps_numerator = config.fps;
        // Start low: a fixed high rate sent into a connection whose bandwidth is not yet known queues behind the
        // sender's pacer and ages every frame. The transport's target raises it toward the configured maximum.
        encoder_config.bitrate_bps = std::min(config.bitrate_bps, kStartBitrateBps);
        encoder_config.idr_period = config.idr_period;
        // NVENC copies the source's levels; the stream must say which range they are in or players rescale them.
        encoder_config.full_range = config.source_full_range;
        NvencEncoder encoder(context, encoder_config);

        const IpcStream& stream = client.streams().front();
        bool need_idr = true;
        uint32_t own_frame_id = 0;
        uint64_t last_session = 0;
        FrameNotice notice;
        std::optional<FrameNotice> pending; // a newer notice seen while checking the last slot; used next
        while (running)
        {
            if (pending)
            {
                notice = std::move(*pending);
                pending.reset();
            }
            else if (!client.next_frame(notice, std::chrono::milliseconds(200)))
                continue;
            // Encode the newest picture: anything queued behind it is older, and a late frame is worse than a skipped
            // one.
            FrameNotice newer;
            while (client.next_frame(newer, std::chrono::milliseconds(0)))
            {
                notice = std::move(newer);
                ++stats.skipped;
            }

            const IpcSlot& slot = client.slot(notice);
            const auto* base = static_cast<const uint8_t*>(slot.device_ptr);
            NvencEncoder::Surface surface;
            if (!encoder.acquire(surface))
                throw std::runtime_error("no free encoder input surface");
            // The picture goes from the IPC slot to the encoder input on the GPU: a plain copy at full size, or a block
            // average that shrinks it. The encode below waits for this stream.
            const auto copy_status = [&]
            {
                if (config.scale_div > 1)
                {
                    const Nv12Planes in{ base + layout.y_offset, base + layout.uv_offset, layout.y_pitch,
                                         layout.uv_pitch };
                    const Nv12PlanesOut out{ surface.view.y, surface.view.uv, surface.view.y_pitch,
                                             surface.view.uv_pitch };
                    return box_downscale_nv12(in, source.width, source.height, out, config.scale_div, copy_stream);
                }
                const auto y =
                    cudaMemcpy2DAsync(surface.view.y, surface.view.y_pitch, base + layout.y_offset, layout.y_pitch,
                                      source.width, source.height, cudaMemcpyDeviceToDevice, copy_stream);
                if (y != cudaSuccess)
                    return y;
                return cudaMemcpy2DAsync(surface.view.uv, surface.view.uv_pitch, base + layout.uv_offset, layout.uv_pitch,
                                         source.width, source.height / 2, cudaMemcpyDeviceToDevice, copy_stream);
            }();
            if (copy_status != cudaSuccess)
            {
                encoder.release(surface);
                throw std::runtime_error(std::string("GPU copy from the IPC slot failed: ") +
                                         cudaGetErrorString(copy_status));
            }

            // The publisher overwrites a slot without notice once its ring wraps. The copy must have finished before
            // that, and a newer notice for the same slot, or as many newer notices as the ring has other slots, means
            // it may not have. Such a picture is dropped, never encoded.
            if (cudaStreamSynchronize(copy_stream) != cudaSuccess)
            {
                encoder.release(surface);
                throw std::runtime_error("GPU copy from the IPC slot did not complete");
            }
            {
                bool overwritten = false;
                std::size_t newer_count = 0;
                FrameNotice newer;
                while (client.next_frame(newer, std::chrono::milliseconds(0)))
                {
                    ++newer_count;
                    if (newer.slot == notice.slot && newer.slot_generation > notice.slot_generation)
                        overwritten = true;
                    pending = std::move(newer);
                }
                if (stream.slots.size() > 1 && newer_count + 1 >= stream.slots.size())
                    overwritten = true;
                if (overwritten)
                {
                    encoder.release(surface);
                    ++stats.slot_overruns;
                    continue;
                }
            }

            const uint64_t session_id = unsigned_field(notice.metadata, "session_id");
            if (session_id != last_session)
            {
                // A restarted source: start the track's picture sequence over at an IDR.
                need_idr = true;
                last_session = session_id;
            }
            const auto feedback = sink->take_feedback();
            const bool force_idr = need_idr || feedback.keyframe_requested;
            if (feedback.target_bitrate_bps)
            {
                // The congestion controller's target for this track, with headroom. A fixed rate above what the
                // connection can carry only fills the sender's pacing queue and ages every frame behind it. Never
                // above the configured rate, never below a floor that keeps the picture usable.
                constexpr uint32_t kMinBitrateBps = 500'000;
                stats.target_bps = *feedback.target_bitrate_bps;
                const uint64_t shaded = static_cast<uint64_t>(*feedback.target_bitrate_bps) * 9 / 10;
                const auto wanted =
                    static_cast<uint32_t>(std::clamp<uint64_t>(shaded, kMinBitrateBps, config.bitrate_bps));
                const uint32_t current = encoder.bitrate();
                // Only a real change reconfigures: each one resets rate control inside the encoder.
                if (wanted > current + current / 10 || wanted + wanted / 10 < current)
                {
                    encoder.set_bitrate(wanted);
                    ++stats.rate_changes;
                }
            }
            stats.bitrate_bps = encoder.bitrate();

            // The frame-ready instant: the picture is on the encoder's input, before any encoding delay. Both clocks
            // are read together so the capture bases can move the sensor time between them.
            timespec ready_realtime{};
            clock_gettime(CLOCK_REALTIME, &ready_realtime);
            const int64_t ready_realtime_ns =
                static_cast<int64_t>(ready_realtime.tv_sec) * 1'000'000'000 + ready_realtime.tv_nsec;
            const int64_t ready_monotonic_ns = monotonic_ns();

            const auto begin = std::chrono::steady_clock::now();
            EncodedPacket packet = encoder.encode(surface, copy_stream, 0, force_idr);
            stats.encode_us_total += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count());
            need_idr = false;

            const uint64_t sequence = unsigned_field(notice.metadata, "stream_sequence");
            EncodedVideoFrame frame;
            frame.data = packet.data.data();
            frame.size = packet.data.size();
            frame.keyframe = packet.keyframe;
            frame.timestamp_us = static_cast<uint64_t>(ready_monotonic_ns / 1000);
            // The sensor start-of-frame is wall-clock time on the sender; only a realtime (Unix) sender clock may be
            // used as Unix time. Any other domain, or none, means the publisher cannot vouch for a capture time.
            const auto wall_domain = notice.metadata.find("sender_wall_clock");
            const bool wall_is_realtime = wall_domain != notice.metadata.end() && wall_domain->is_string() &&
                                          wall_domain->get<std::string>() == "realtime";
            const uint64_t sensor_sof_ns = wall_is_realtime ? unsigned_field(notice.metadata, "sensor_sof_ns") : 0;
            frame.user_timestamp_us = config.timestamp_basis == TimestampBasis::MonotonicUs ?
                                          static_cast<uint64_t>(ready_monotonic_ns / 1000) :
                                          0;
            if (config.timestamp_basis == TimestampBasis::CaptureMonotonicUs && config.chrony_clock != nullptr)
            {
                // The receiver's timing record: the sensor start of frame on the sender's CLOCK_REALTIME with the
                // sender's stated uncertainty. A conversion that cannot be vouched for publishes no capture time; the
                // frame-ready time is never offered in its place.
                const auto valid = notice.metadata.find("sensor_capture_valid");
                const bool sender_valid = valid != notice.metadata.end() && valid->is_boolean() && valid->get<bool>();
                const auto result = config.chrony_clock->qualify(
                    sender_valid, static_cast<int64_t>(unsigned_field(notice.metadata, "sensor_capture_realtime_ns")),
                    static_cast<int64_t>(unsigned_field(notice.metadata, "sensor_capture_uncertainty_ns")));
                ++stats.chrony_verdicts[static_cast<std::size_t>(result.verdict)];
                stats.last_uncertainty_ns = result.uncertainty_ns;
                stats.last_generation = result.generation;
                // A precision miss still has a converted instant with a stated bound, so the frame keeps its capture
                // time; only a conversion that cannot be vouched for publishes none.
                if (result.usable())
                    frame.user_timestamp_us = static_cast<uint64_t>(result.capture_monotonic_ns) / 1000;
            }
            if (sensor_sof_ns != 0)
            {
                const int64_t age_ns = ready_realtime_ns - static_cast<int64_t>(sensor_sof_ns);
                if (age_ns >= 0 && age_ns <= kMaxCaptureAgeNs)
                {
                    const auto age_us = static_cast<uint32_t>(age_ns / 1000);
                    stats.capture_to_ready_us_total += age_us;
                    ++stats.capture_to_ready_samples;
                    uint32_t seen = stats.capture_to_ready_max_us.load();
                    while (age_us > seen && !stats.capture_to_ready_max_us.compare_exchange_weak(seen, age_us))
                    {
                    }
                }
            }
            // The source's own sequence keeps frames lost before the encoder visible to the consumer; a source without
            // one is numbered here.
            frame.frame_id = sequence != 0 ? static_cast<uint32_t>(sequence) : ++own_frame_id;
            frame.capture_tsc = unsigned_field(notice.metadata, "sof_tsc_ns");

            ++stats.frames;
            if (packet.keyframe)
                ++stats.keyframes;
            if (!sink->submit(frame))
            {
                // The SDK refused a unit, which can break the reference chain: follow with a keyframe.
                need_idr = true;
                ++stats.rejected;
            }
        }
        (void)stream;
    }

    WorkerConfig config;
    CudaContext& context;
    CudaIpcClient client;
    PlaneLayout layout;
    Geometry source;
    Geometry geometry;
    std::unique_ptr<VideoTrackSink> sink;
    cudaStream_t copy_stream = nullptr;
    std::atomic<bool> running{ false };
    std::thread thread;
    mutable std::mutex failure_mutex;
    std::string failure;
};

StreamWorker::StreamWorker(const WorkerConfig& config, CudaContext& context, LiveKitSession& session)
    : m_config(config), m_impl(std::make_unique<Impl>(config, context, session))
{
}

StreamWorker::~StreamWorker() = default;

void StreamWorker::start()
{
    m_impl->running = true;
    m_impl->thread = std::thread([this] { m_impl->run(m_stats); });
}

void StreamWorker::stop()
{
    m_impl->running = false;
    if (m_impl->thread.joinable())
        m_impl->thread.join();
}

std::string StreamWorker::failure() const
{
    std::lock_guard<std::mutex> lock(m_impl->failure_mutex);
    return m_impl->failure;
}

} // namespace camera_bridge

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "stream_worker.hpp"

#include <gpu_ops/gpu_step.hpp>
#include <ipc_client/cuda_ipc_client.hpp>
#include <nvenc/nvenc_encoder.hpp>
#include <video_types/slot_age.hpp>
#include <video_types/video_types.hpp>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <cuda_runtime_api.h>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace camera_bridge
{

namespace
{

constexpr auto kOverwriteReportInterval = std::chrono::seconds(5);

/// The loud error for a picture that may mix two frames. Rate limited per stream; the stats line keeps the totals.
void report_slot_overwrite_risk(const WorkerConfig& config,
                                const FrameNotice& notice,
                                std::size_t slot_count,
                                bool newer_notice,
                                bool late_copy,
                                int64_t age_ns,
                                int64_t deadline_ns,
                                uint64_t dropped_total)
{
    std::cerr << "\n!!!!!!!!!! SLOT OVERWRITE RISK on stream '" << config.stream << "': picture DROPPED !!!!!!!!!!\n"
              << "  The receiver's ring (" << slot_count << " slots) may have rewritten slot " << notice.slot
              << " (generation " << notice.slot_generation << ") while this bridge was still reading it, so the\n"
              << "  pixels could be a mix of two frames under the first frame's ID and timestamp.\n";
    if (newer_notice)
        std::cerr << "  Evidence: a newer notice for the same slot, or a full ring of newer notices, arrived.\n";
    if (late_copy)
        std::cerr << "  Evidence: the copy finished " << static_cast<double>(age_ns) / 1e6
                  << " ms after the notice; the ring wraps in " << static_cast<double>(deadline_ns) / 1e6 << " ms at "
                  << config.fps << " fps.\n";
    std::cerr
        << "  Cause: this consumer is reading too slowly (starved GPU or CPU), or the receiver is delivering frames\n"
        << "  faster than " << config.fps << " fps. Fix the stall; do not trust recordings of this period.\n"
        << "  Pictures dropped for this reason so far: " << dropped_total << " (reported at most every "
        << kOverwriteReportInterval.count() << " s)\n\n"
        << std::flush;
}

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

// Bytes a plane needs from the slot start through the end of its last row, or false on overflow.
bool plane_extent(size_t offset, size_t pitch, size_t rows, size_t row_bytes, size_t& end)
{
    size_t span = 0;
    if (rows == 0 || row_bytes > pitch || __builtin_mul_overflow(rows - 1, pitch, &span) ||
        __builtin_add_overflow(span, row_bytes, &span) || __builtin_add_overflow(span, offset, &end))
        return false;
    return true;
}

// Every GPU read starts from the advertised offsets and pitches, so each plane must fit inside every slot.
void check_plane_extents(const IpcStream& stream, const PlaneLayout& layout)
{
    const auto width = static_cast<size_t>(stream.width);
    const auto height = static_cast<size_t>(stream.height);
    size_t y_end = 0;
    size_t uv_end = 0;
    if (stream.width <= 0 || stream.height <= 0 || height % 2 != 0 ||
        !plane_extent(layout.y_offset, layout.y_pitch, height, width, y_end) ||
        !plane_extent(layout.uv_offset, layout.uv_pitch, height / 2, width, uv_end))
        throw std::runtime_error("stream '" + stream.name + "' advertises an invalid NV12 plane layout");
    const size_t needed = std::max(y_end, uv_end);
    for (const IpcSlot& slot : stream.slots)
    {
        if (slot.nbytes < needed)
            throw std::runtime_error("stream '" + stream.name + "' advertises NV12 planes that extend past its " +
                                     std::to_string(slot.nbytes) + "-byte slots");
    }
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
        check_plane_extents(stream, layout);
        source = { static_cast<uint32_t>(stream.width), static_cast<uint32_t>(stream.height) };
        // The steps asked for run in order at the source size, then the stream's own downscale shrinks the result.
        for (const auto& step_spec : cfg.gpu_steps)
            pipeline.add(make_gpu_step(step_spec));
        if (cfg.scale_div > 1)
            pipeline.add(make_box_downscale_step(cfg.scale_div));
        const std::string unusable = pipeline.prepare({ source.width, source.height });
        if (!unusable.empty())
            throw std::runtime_error("stream '" + cfg.stream + "' (" + std::to_string(source.width) + "x" +
                                     std::to_string(source.height) + "): " + unusable);
        geometry = { pipeline.output_size().width, pipeline.output_size().height };

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
        // Declared after the encoder so it runs first on every exit, including a throw: the encoder's teardown frees
        // the surfaces a queued copy may still be writing, and destroying a stream does not wait for its work.
        struct CopyDrain
        {
            cudaStream_t stream;
            ~CopyDrain()
            {
                const cudaError_t status = cudaStreamSynchronize(stream);
                if (status != cudaSuccess)
                    std::cerr
                        << "[error] the GPU copy stream did not drain before teardown: " << cudaGetErrorString(status)
                        << "\n";
            }
        } copy_drain{ copy_stream };

        const IpcStream& stream = client.streams().front();
        bool need_idr = true;
        const int64_t deadline_ns = config.max_notice_age_ms != 0 ?
                                        static_cast<int64_t>(config.max_notice_age_ms) * 1'000'000 :
                                        slot_read_deadline_ns(config.fps, stream.slots.size());
        auto last_overwrite_report = std::chrono::steady_clock::now() - kOverwriteReportInterval;
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
            while (running && client.next_frame(newer, std::chrono::milliseconds(0)))
            {
                notice = std::move(newer);
                ++stats.skipped;
            }
            if (!running)
                break;

            const IpcSlot& slot = client.slot(notice);
            const auto* base = static_cast<const uint8_t*>(slot.device_ptr);
            NvencEncoder::Surface surface;
            if (!encoder.acquire(surface))
                throw std::runtime_error("no free encoder input surface");
            // The picture goes from the IPC slot to the encoder input on the GPU, through the configured series of
            // GPU steps (a plain copy when there are none). This is the place a per-frame image operation such as lens
            // rectification runs: see GpuStep in gpu_ops/gpu_step.hpp. The encode below waits for this stream.
            const Nv12Planes in{ base + layout.y_offset, base + layout.uv_offset, layout.y_pitch, layout.uv_pitch };
            const Nv12PlanesOut out{ surface.view.y, surface.view.uv, surface.view.y_pitch, surface.view.uv_pitch };
            const cudaError_t copy_status = pipeline.enqueue(in, out, copy_stream);
            if (copy_status != cudaSuccess)
            {
                // The first plane may already be queued; the surface goes back only after the stream has drained.
                cudaStreamSynchronize(copy_stream);
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
            // The copy is done: measure its age now, before the notice drain below adds its own time.
            const int64_t copy_done_ns = monotonic_ns();
            {
                bool overwritten = false;
                std::size_t newer_count = 0;
                FrameNotice newer;
                while (running && client.next_frame(newer, std::chrono::milliseconds(0)))
                {
                    ++newer_count;
                    if (newer.slot == notice.slot && newer.slot_generation > notice.slot_generation)
                        overwritten = true;
                    pending = std::move(newer);
                }
                if (!running)
                {
                    // Stopping: the copy has completed, so the surface can go back and the worker can leave.
                    encoder.release(surface);
                    break;
                }
                if (stream.slots.size() > 1 && newer_count + 1 >= stream.slots.size())
                    overwritten = true;
                // If the copy finished later than the publisher needs to wrap its ring, the slot may have been
                // overwritten during it, whether or not the replacement notice has arrived yet.
                const SlotAge age = classify_slot_age(notice.timestamp_ns, copy_done_ns, deadline_ns);
                if (age == SlotAge::Unknown && deadline_ns > 0)
                    ++stats.unchecked_slot_reads;
                const bool stale = age == SlotAge::Stale;
                if (overwritten || stale)
                {
                    encoder.release(surface);
                    if (overwritten)
                        ++stats.slot_overruns;
                    else
                        ++stats.stale_slot_reads;
                    const auto now = std::chrono::steady_clock::now();
                    if (now - last_overwrite_report >= kOverwriteReportInterval)
                    {
                        last_overwrite_report = now;
                        report_slot_overwrite_risk(config, notice, stream.slots.size(), overwritten, stale,
                                                   copy_done_ns - static_cast<int64_t>(notice.timestamp_ns),
                                                   deadline_ns, stats.slot_overruns + stats.stale_slot_reads);
                    }
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
                constexpr uint32_t kFloorBps = 500'000;
                // A configured cap below the floor is the cap itself; clamp needs lo <= hi.
                const uint32_t kMinBitrateBps = std::min(kFloorBps, config.bitrate_bps);
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
    GpuPipeline pipeline;
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

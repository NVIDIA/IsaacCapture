// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "stream_worker.hpp"

#include <gpu_ops/box_downscale.hpp>
#include <ipc_client/cuda_ipc_client.hpp>
#include <nvenc/nvenc_encoder.hpp>
#include <video_types/video_types.hpp>

#include <chrono>
#include <ctime>
#include <cuda_runtime_api.h>
#include <mutex>
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
        encoder_config.bitrate_bps = config.bitrate_bps;
        encoder_config.idr_period = config.idr_period;
        NvencEncoder encoder(context, encoder_config);

        const IpcStream& stream = client.streams().front();
        bool need_idr = true;
        uint32_t own_frame_id = 0;
        uint64_t last_session = 0;
        FrameNotice notice;
        while (running)
        {
            if (!client.next_frame(notice, std::chrono::milliseconds(200)))
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

            const uint64_t session_id = unsigned_field(notice.metadata, "session_id");
            if (session_id != last_session)
            {
                // A restarted source: start the track's picture sequence over at an IDR.
                need_idr = true;
                last_session = session_id;
            }
            const auto feedback = sink->take_feedback();
            const bool force_idr = need_idr || feedback.keyframe_requested;

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
            frame.timestamp_us = static_cast<uint64_t>(monotonic_ns() / 1000);
            // The application timestamp on the configured basis; zero tells consumers the frame has no time the
            // publisher can vouch for (for the capture bases, no usable sensor start-of-frame).
            timespec realtime{};
            clock_gettime(CLOCK_REALTIME, &realtime);
            const int64_t realtime_ns = static_cast<int64_t>(realtime.tv_sec) * 1'000'000'000 + realtime.tv_nsec;
            frame.user_timestamp_us = user_timestamp_us(
                config.timestamp_basis, unsigned_field(notice.metadata, "sensor_sof_ns"), realtime_ns, monotonic_ns());
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

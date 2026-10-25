// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sensing_camera.hpp"

#include "frame_sink.hpp"

#include <algorithm>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

#if SENSING_WITH_LIVEKIT
#    include "livekit_video_output.hpp"

#    include <sensing_livekit/livekit_session.hpp>
#else
// Without LiveKit the members are never set; empty types let the camera's special members compile.
namespace plugins
{
namespace sensing
{
class LiveKitSession
{
};
class LiveKitVideoOutput
{
};
} // namespace sensing
} // namespace plugins
#endif

namespace plugins
{
namespace sensing
{

namespace
{

SiplConfig make_sipl_config(const SensingConfig& config)
{
    SiplConfig sipl{};
    sipl.platform_config_json = config.platform_config_json;
    sipl.platform_config_name = config.platform_config_name;
    sipl.link_masks = config.link_masks;
    sipl.nito_dir = config.nito_dir;
    sipl.gpu_id = config.gpu_id;
    sipl.full_range = config.full_range;
    sipl.swap_uv = config.swap_uv;
    sipl.isp0_buffers = config.isp0_buffers;
    return sipl;
}

EncoderConfig make_encoder_config(const SensingConfig& config, const SensorInfo& sensor)
{
    EncoderConfig encoder{};
    encoder.width = sensor.width;
    encoder.height = sensor.height;
    encoder.bitrate_bps = config.bitrate_bps;
    encoder.peak_bitrate_bps = config.peak_bitrate_bps;
    encoder.fps = static_cast<uint32_t>(sensor.fps + 0.5);
    encoder.gop = config.gop;
    encoder.full_range = config.full_range;
    return encoder;
}

} // namespace

SensingCamera::SensingCamera(const SensingConfig& config,
                             const std::vector<StreamConfig>& streams,
                             std::unique_ptr<FrameSink> sink)
    : m_config(config), m_sink(std::move(sink))
{
    if (streams.empty())
        throw std::runtime_error("SensingCamera: no streams requested");

    m_camera = std::make_unique<SiplCamera>(make_sipl_config(config));
    const auto& sensors = m_camera->sensors();

#if SENSING_WITH_LIVEKIT
    // Everything that needs no network is checked first, so a bad name or divisor fails before any
    // connection is made.
    std::set<std::string> livekit_tracks;
    for (const auto& stream_config : streams)
    {
        if (stream_config.livekit_track.empty())
            continue;
        if (!livekit_tracks.insert(stream_config.livekit_track).second)
            throw std::runtime_error("LiveKit track name '" + stream_config.livekit_track + "' is used twice");
        for (const auto& sensor : sensors)
        {
            if (sensor.id == stream_config.sensor_id)
                LiveKitVideoOutput::validate_geometry(sensor.width, sensor.height, config.livekit.scale_div);
        }
    }
#endif

    m_streams.reserve(streams.size());
    for (const auto& stream_config : streams)
    {
        const auto found = std::find_if(
            sensors.begin(), sensors.end(), [&](const SensorInfo& s) { return s.id == stream_config.sensor_id; });
        if (found == sensors.end())
        {
            std::ostringstream oss;
            oss << "sensor=" << stream_config.sensor_id << " is not a pipeline in '" << config.platform_config_name
                << "'. Available:";
            for (const auto& s : sensors)
                oss << ' ' << s.id;
            oss << ". Note this is the SIPL pipeline index, not the GMSL link index.";
            throw std::runtime_error(oss.str());
        }

        Stream stream;
        stream.sensor_id = found->id;
        stream.width = found->width;
        stream.height = found->height;

        if (!stream_config.output_path.empty())
            stream.encoder = std::make_unique<JetsonEncoder>(make_encoder_config(config, *found));

        if (!stream_config.ipc_socket_path.empty())
        {
            CudaIpcConfig ipc{};
            ipc.socket_path = stream_config.ipc_socket_path;
            ipc.width = found->width;
            ipc.height = found->height;
            ipc.sensor_id = found->id;
            ipc.gpu_id = config.gpu_id;
            stream.publisher = std::make_unique<CudaIpcPublisher>(ipc);
        }

        if (!stream_config.livekit_track.empty())
        {
#if SENSING_WITH_LIVEKIT
            LiveKitOutputConfig output;
            output.source_width = found->width;
            output.source_height = found->height;
            output.scale_div = config.livekit.scale_div;
            output.fps = config.livekit.fps;
            output.gpu_id = config.gpu_id;
            output.bitrate_bps = config.livekit.bitrate_bps;
            output.peak_bitrate_bps = config.livekit.bitrate_bps + config.livekit.bitrate_bps / 2;
            output.gop = config.gop;
            output.full_range = config.full_range;
            if (!m_livekit)
                m_livekit = LiveKitSession::connect(config.livekit.url, config.livekit.token);

            PublishSpec spec;
            spec.name = stream_config.livekit_track;
            spec.width = found->width / output.scale_div;
            spec.height = found->height / output.scale_div;
            spec.max_bitrate_bps = output.peak_bitrate_bps;
            spec.max_fps = static_cast<double>(std::min<double>(output.fps, found->fps));
            stream.livekit = std::make_unique<LiveKitVideoOutput>(m_livekit->publish_video(spec), output);
            std::cout << "LiveKit track '" << spec.name << "': " << spec.width << "x" << spec.height << " @ up to "
                      << spec.max_fps << " fps" << std::endl;
#else
            throw std::runtime_error("this build has no LiveKit support; configure with -DSENSING_LIVEKIT=ON");
#endif
        }

        m_streams.push_back(std::move(stream));

        std::cout << "Sensor " << found->id << " (" << found->name << "): " << found->width << "x" << found->height
                  << " @ " << found->fps << " fps" << std::endl;
    }

    m_camera->start();
}

SensingCamera::~SensingCamera()
{
    if (m_camera)
        m_camera->stop();
}

int64_t SensingCamera::take_capture_tsc(Stream& stream, int64_t timestamp_ns)
{
    // The encoder returns units in submission order, so the match is at the
    // front. Anything older than the match was dropped by the encoder and is
    // discarded with it, which is what keeps this bounded.
    //
    // The stamp rides a V4L2 timeval, which holds microseconds, so the unit
    // comes back with the submitted stamp truncated. Compare at that resolution.
    constexpr int64_t kNsPerUs = 1000;
    while (!stream.pending_stamps.empty())
    {
        auto entry = stream.pending_stamps.front();
        stream.pending_stamps.pop_front();
        if (entry.first / kNsPerUs == timestamp_ns / kNsPerUs)
            return entry.second;
    }
    return 0;
}

void SensingCamera::update()
{
#if SENSING_WITH_LIVEKIT
    // The room is gone for good: stop rather than keep capturing for nobody.
    if (m_livekit && m_livekit->failed())
        throw std::runtime_error(m_livekit->failure());
#endif

    for (auto& stream : m_streams)
    {
        // Accept consumers and reap released slots even on a frameless tick,
        // so a viewer can attach before the camera produces anything.
        if (stream.publisher)
            stream.publisher->poll();

        auto view = m_camera->latest(stream.sensor_id);
        if (view.has_value() && (!stream.have_last_sequence || view->sequence != stream.last_sequence))
        {
            if (stream.have_last_sequence && view->sequence > stream.last_sequence + 1)
                stream.missed_captures += view->sequence - stream.last_sequence - 1;
            stream.last_sequence = view->sequence;
            stream.have_last_sequence = true;

            // Publish before encoding: the IPC consumer is the latency-
            // sensitive path, and the encoder submit below is pipelined anyway.
            if (stream.publisher)
                stream.publisher->publish(view->ptr, view->pitch, view->timestamp_ns);

#if SENSING_WITH_LIVEKIT
            // A GPU snapshot only; the conversion and the network are on the output's own thread.
            if (stream.livekit)
                stream.livekit->submit(*view);
#endif

            if (stream.encoder)
            {
                if (stream.encoder->submit(view->ptr, view->pitch, static_cast<int64_t>(view->timestamp_ns)))
                {
                    stream.pending_stamps.emplace_back(
                        static_cast<int64_t>(view->timestamp_ns), static_cast<int64_t>(view->capture_tsc_ns));
                }
                else
                {
                    ++stream.encoder_drops;
                }
            }
            else
            {
                // No encoder means nothing would ever reach the sink, so an
                // ipc-only stream emits its metadata here instead.
                dispatch(
                    stream, {}, static_cast<int64_t>(view->timestamp_ns), static_cast<int64_t>(view->capture_tsc_ns));
            }
        }

#if SENSING_WITH_LIVEKIT
        if (stream.livekit && stream.livekit->failed())
            throw std::runtime_error("sensor " + std::to_string(stream.sensor_id) + ": " + stream.livekit->failure());
#endif

        // Submission and output are decoupled: the V4L2 encoder needs several
        // input frames before it emits the first unit, so drain independently.
        // Each unit carries its own frame's stamp back from the output plane.
        if (stream.encoder)
        {
            for (auto unit = stream.encoder->poll(); !unit.empty(); unit = stream.encoder->poll())
            {
                const int64_t tsc = take_capture_tsc(stream, unit.timestamp_ns);
                dispatch(stream, std::move(unit.data), unit.timestamp_ns, tsc);
            }
        }
    }
}

void SensingCamera::flush()
{
    for (auto& stream : m_streams)
    {
#if SENSING_WITH_LIVEKIT
        if (stream.livekit)
            stream.livekit->flush();
#endif
        if (!stream.encoder)
            continue;
        auto unit = stream.encoder->end_of_stream();
        if (!unit.empty())
        {
            const int64_t tsc = take_capture_tsc(stream, unit.timestamp_ns);
            dispatch(stream, std::move(unit.data), unit.timestamp_ns, tsc);
        }
    }
}

void SensingCamera::dispatch(Stream& stream, std::vector<uint8_t> h264, int64_t timestamp_ns, int64_t capture_tsc_ns)
{
    SensingFrame frame;
    frame.sensor_id = stream.sensor_id;
    frame.h264_data = std::move(h264);
    frame.metadata.sensor_id = stream.sensor_id;
    frame.metadata.sequence_number = stream.frame_count;
    frame.metadata.capture_tsc_ns = static_cast<uint64_t>(capture_tsc_ns);

    // CLOCK_MONOTONIC, stamped at YUV->RGBA conversion, so it includes capture
    // and convert latency. The device clock behind it is SIPL's frameCaptureTSC,
    // on a timebase every sensor on the rig shares -- that is the one a consumer
    // must pair the two eyes on.
    frame.sample_time_local_common_clock_ns = timestamp_ns;
    frame.sample_time_raw_device_clock_ns = capture_tsc_ns;

    ++stream.frame_count;
    m_sink->on_frame(frame);
}

void SensingCamera::print_stats() const
{
    for (const auto& stream : m_streams)
    {
        std::cout << "  sensor " << stream.sensor_id << ": " << stream.frame_count << " frames";
        if (stream.missed_captures)
            std::cout << " | " << stream.missed_captures << " captures missed";
        if (stream.encoder_drops)
            std::cout << " | " << stream.encoder_drops << " encoder drops";
        if (stream.publisher)
        {
            std::cout << " | ipc " << stream.publisher->published_count() << " published, "
                      << stream.publisher->dropped_count() << " dropped"
                      << (stream.publisher->has_consumer() ? "" : ", no consumer");
        }
#if SENSING_WITH_LIVEKIT
        if (stream.livekit)
        {
            const auto s = stream.livekit->stats();
            std::cout << " | livekit " << s.published << " published, " << s.paced_out << " paced out, " << s.replaced
                      << " replaced";
            if (s.encoder_busy)
                std::cout << ", " << s.encoder_busy << " encoder busy";
            if (s.rejected)
                std::cout << ", " << s.rejected << " rejected";
            if (s.keyframes_requested)
                std::cout << ", " << s.keyframes_requested << " keyframes requested";
            const std::string encoder = stream.livekit->describe();
            if (!encoder.empty())
                std::cout << " (" << encoder << ")";
        }
#endif
        std::cout << std::endl;
    }
}

} // namespace sensing
} // namespace plugins

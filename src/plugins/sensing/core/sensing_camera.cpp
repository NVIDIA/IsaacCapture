// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sensing_camera.hpp"

#include "livekit_video_output.hpp"

#include <sensing_livekit/livekit_session.hpp>

#include <algorithm>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace plugins
{
namespace sensing
{

namespace
{

LiveKitOutputConfig make_output_config(const SensingConfig& config, const StreamConfig& stream, const CameraInfo& sensor)
{
    LiveKitOutputConfig output;
    output.source_width = sensor.width;
    output.source_height = sensor.height;
    output.scale_div = config.livekit.scale_div;
    output.fps = config.livekit.fps;
    output.gpu_id = config.gpu_id;
    output.bitrate_bps = config.livekit.bitrate_bps;
    output.peak_bitrate_bps = config.livekit.bitrate_bps + config.livekit.bitrate_bps / 2;
    output.input = sensor.pixels;
    output.gop = config.gop;
    output.full_range = config.full_range;
    output.timestamp_basis = config.livekit.timestamp_basis;
    output.direct_host = stream.direct_host;
    output.direct_port = stream.direct_port;
    return output;
}

} // namespace

SensingCamera::SensingCamera(const SensingConfig& config,
                             std::unique_ptr<CameraCapture> camera,
                             const std::vector<StreamConfig>& streams)
    : m_config(config)
{
    if (streams.empty())
        throw std::runtime_error("SensingCamera: no streams requested");

    m_camera = std::move(camera);
    if (!m_camera)
        throw std::runtime_error("SensingCamera: no camera backend");
    const auto& sensors = m_camera->cameras();

    // Everything that needs no network is checked first, so a bad name or divisor fails before any
    // connection is made.
    std::set<std::string> tracks;
    for (const auto& stream_config : streams)
    {
        if (!tracks.insert(stream_config.livekit_track).second)
            throw std::runtime_error("LiveKit track name '" + stream_config.livekit_track + "' is used twice");
        const auto found = std::find_if(
            sensors.begin(), sensors.end(), [&](const CameraInfo& s) { return s.id == stream_config.sensor_id; });
        if (found == sensors.end())
        {
            std::ostringstream oss;
            oss << "sensor=" << stream_config.sensor_id << " is not a pipeline in '" << config.platform_config_name
                << "'. Available:";
            for (const auto& s : sensors)
                oss << ' ' << s.id;
            oss << ". Note this is the backend's own camera index (a SIPL pipeline index is not the GMSL link index).";
            throw std::runtime_error(oss.str());
        }
        LiveKitVideoOutput::validate_geometry(found->width, found->height, config.livekit.scale_div);
    }

    m_livekit = LiveKitSession::connect(config.livekit.url, config.livekit.token);
    m_streams.reserve(streams.size());
    for (const auto& stream_config : streams)
    {
        const auto found = std::find_if(
            sensors.begin(), sensors.end(), [&](const CameraInfo& s) { return s.id == stream_config.sensor_id; });
        const LiveKitOutputConfig output = make_output_config(config, stream_config, *found);

        PublishSpec spec;
        spec.name = stream_config.livekit_track;
        spec.width = found->width / output.scale_div;
        spec.height = found->height / output.scale_div;
        spec.max_bitrate_bps = output.peak_bitrate_bps;
        spec.max_fps = std::min<double>(output.fps, found->fps);

        Stream stream;
        stream.sensor_id = found->id;
        stream.livekit = std::make_unique<LiveKitVideoOutput>(m_livekit->publish_video(spec), output);
        m_streams.push_back(std::move(stream));

        std::cout << "Sensor " << found->id << " (" << found->name << ") -> LiveKit track '" << spec.name
                  << "': " << spec.width << "x" << spec.height << " @ up to " << spec.max_fps << " fps" << std::endl;
    }

    m_camera->start();
}

SensingCamera::~SensingCamera()
{
    if (m_camera)
        m_camera->stop();
}

void SensingCamera::update()
{
    // The room is gone for good: stop rather than keep capturing for nobody.
    if (m_livekit->failed())
        throw std::runtime_error(m_livekit->failure());

    for (auto& stream : m_streams)
    {
        auto view = m_camera->acquire_latest(stream.sensor_id);
        if (view.has_value() && (!stream.have_last_sequence || view->sequence != stream.last_sequence))
        {
            if (stream.have_last_sequence && view->sequence > stream.last_sequence + 1)
                stream.missed_captures += view->sequence - stream.last_sequence - 1;
            stream.last_sequence = view->sequence;
            stream.have_last_sequence = true;
            ++stream.frame_count;

            // A GPU snapshot only; the conversion and the network are on the output's own thread.
            stream.livekit->submit(*view);
        }

        if (stream.livekit->failed())
            throw std::runtime_error("sensor " + std::to_string(stream.sensor_id) + ": " + stream.livekit->failure());
    }
}

void SensingCamera::flush()
{
    for (auto& stream : m_streams)
        stream.livekit->flush();
}

void SensingCamera::print_stats() const
{
    for (const auto& stream : m_streams)
    {
        const auto s = stream.livekit->stats();
        std::cout << "  sensor " << stream.sensor_id << ": " << stream.frame_count << " frames";
        if (stream.missed_captures)
            std::cout << " | " << stream.missed_captures << " captures missed";
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
        std::cout << std::endl;
    }
}

} // namespace sensing
} // namespace plugins

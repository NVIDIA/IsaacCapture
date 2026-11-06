// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sensing_camera.hpp"

#include "video_output.hpp"

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

VideoOutputConfig make_output_config(const SensingConfig& config, const StreamConfig& stream, const CameraInfo& sensor)
{
    VideoOutputConfig output;
    output.source_width = sensor.width;
    output.source_height = sensor.height;
    output.scale_div = config.encode.scale_div;
    output.fps = config.encode.fps;
    output.gpu_id = config.gpu_id;
    output.bitrate_bps = config.encode.bitrate_bps;
    output.peak_bitrate_bps = config.encode.bitrate_bps + config.encode.bitrate_bps / 2;
    output.input = sensor.pixels;
    output.gop = config.gop;
    output.full_range = config.full_range;
    output.timestamp_basis = config.encode.timestamp_basis;
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

    // Everything that needs no hardware is checked first, so a bad name or divisor fails before the cameras start.
    std::set<std::string> tracks;
    for (const auto& stream_config : streams)
    {
        if (!tracks.insert(stream_config.name).second)
            throw std::runtime_error("stream name '" + stream_config.name + "' is used twice");
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
        VideoOutput::validate_geometry(found->width, found->height, config.encode.scale_div);
        if (found->pixels == PixelKind::DmaBufNv12 && config.encode.scale_div != 1)
            throw std::runtime_error("sensor " + std::to_string(found->id) +
                                     " delivers DMA-BUF frames, which cannot be reduced: use --scale-div=1");
    }

    m_streams.reserve(streams.size());
    for (const auto& stream_config : streams)
    {
        const auto found = std::find_if(
            sensors.begin(), sensors.end(), [&](const CameraInfo& s) { return s.id == stream_config.sensor_id; });
        const VideoOutputConfig output = make_output_config(config, stream_config, *found);
        const uint32_t width = found->width / output.scale_div;
        const uint32_t height = found->height / output.scale_div;
        const double max_fps = std::min<double>(output.fps, found->fps);

        Stream stream;
        stream.sensor_id = found->id;
        stream.track = stream_config.name;
        stream.output = std::make_unique<VideoOutput>(output);
        m_streams.push_back(std::move(stream));

        std::cout << "Sensor " << found->id << " (" << found->name << ") -> stream '" << stream_config.name
                  << "': " << width << "x" << height << " @ up to " << max_fps << " fps, direct link "
                  << (stream_config.direct_host.empty() ?
                          std::string("not set yet") :
                          stream_config.direct_host + ":" + std::to_string(stream_config.direct_port))
                  << std::endl;
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
            stream.output->submit(*view);
        }

        if (stream.output->failed())
            throw std::runtime_error("sensor " + std::to_string(stream.sensor_id) + ": " + stream.output->failure());
    }
}

std::vector<StreamFigures> SensingCamera::figures() const
{
    std::vector<StreamFigures> out;
    for (const auto& stream : m_streams)
    {
        StreamFigures figures;
        figures.track = stream.track;
        figures.sensor_id = stream.sensor_id;
        figures.frames = stream.frame_count;
        figures.published = stream.output->stats().published;
        figures.direct = stream.output->direct_status();
        out.push_back(std::move(figures));
    }
    return out;
}

bool SensingCamera::set_direct_target(const std::string& track, const std::string& host, uint16_t port)
{
    for (auto& stream : m_streams)
    {
        if (stream.track == track)
        {
            stream.output->set_direct_target(host, port);
            return true;
        }
    }
    return false;
}

void SensingCamera::flush()
{
    for (auto& stream : m_streams)
        stream.output->flush();
}

void SensingCamera::print_stats() const
{
    for (const auto& stream : m_streams)
    {
        const auto s = stream.output->stats();
        std::cout << "  sensor " << stream.sensor_id << ": " << stream.frame_count << " frames";
        if (stream.missed_captures)
            std::cout << " | " << stream.missed_captures << " captures missed";
        std::cout << " | encoded " << s.published << ", " << s.paced_out << " paced out, " << s.replaced << " replaced";
        if (s.encoder_busy)
            std::cout << ", " << s.encoder_busy << " encoder busy";
        if (s.keyframes_requested)
            std::cout << ", " << s.keyframes_requested << " keyframes requested";
        const DirectLinkStatus direct = stream.output->direct_status();
        if (direct.host.empty())
            std::cout << " | direct: no destination";
        else if (direct.connected)
            std::cout << " | direct: " << direct.host << ":" << direct.port << " connected";
        else
            std::cout << " | direct: " << direct.host << ":" << direct.port << " NOT connected"
                      << (direct.error.empty() ? "" : " (" + direct.error + ")");
        const std::string encoder = stream.output->describe();
        if (!encoder.empty())
            std::cout << " (" << encoder << ")";
        std::cout << std::endl;
    }
}

} // namespace sensing
} // namespace plugins

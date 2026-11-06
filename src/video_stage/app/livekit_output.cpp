// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "livekit_output.hpp"

#include <stdexcept>

namespace video_stage
{

namespace sensing = plugins::sensing;

SharedLiveKit::SharedLiveKit(const std::string& url, const std::string& token)
{
    m_session = sensing::LiveKitSession::connect(url, token);
}

std::unique_ptr<sensing::VideoTrackSink> SharedLiveKit::publish(const sensing::PublishSpec& spec)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_session->publish_video(spec);
}

bool SharedLiveKit::failed() const
{
    return m_session->failed();
}

std::string SharedLiveKit::failure() const
{
    return m_session->failure();
}

LiveKitOutput::LiveKitOutput(SharedLiveKit& livekit,
                             const LiveKitOutputConfig& config,
                             MappingSource mapping,
                             std::function<void()> request_idr)
    : m_livekit(livekit), m_config(config), m_mapping(std::move(mapping)), m_request_idr(std::move(request_idr))
{
}

void LiveKitOutput::break_publication()
{
    m_track.reset(); // destroying the sink unpublishes the track
    m_awaiting_keyframe = true;
    m_idr_requested_for_gap = false;
}

void LiveKitOutput::write(const EncodedPacket& packet, const FrameMeta& meta, const Geometry& geometry)
{
    if (m_livekit.failed())
        throw std::runtime_error("LiveKit room lost: " + m_livekit.failure());

    // A subscriber's keyframe or rate request reaches the encoder through the pipeline.
    if (m_track && m_track->take_feedback().keyframe_requested)
        m_request_idr();

    const auto upstream = m_mapping();
    if (!upstream)
    {
        ++m_unmapped;
        break_publication();
        return;
    }
    if (upstream->mapping.uncertainty_ns > m_config.max_upstream_uncertainty_ns)
    {
        ++m_over_budget;
        break_publication();
        return;
    }
    // This milestone publishes one fixed picture size: a change is refused rather than reconfiguring the track.
    if (m_track && !(geometry == m_track_geometry))
        throw std::runtime_error("the stream's picture size changed; publishing a different size is not supported");
    // The Orin's clock jumped: end this publication so consumers see a break instead of a silent offset change.
    if (m_track && upstream->epoch != m_epoch)
    {
        ++m_republished;
        break_publication();
    }
    m_epoch = upstream->epoch;

    if (m_awaiting_keyframe && !packet.keyframe)
    {
        // Resume at a keyframe: ask for one once per gap instead of waiting out the encoder's own interval.
        if (!m_idr_requested_for_gap)
        {
            m_idr_requested_for_gap = true;
            m_request_idr();
        }
        return;
    }
    if (!m_track)
    {
        sensing::PublishSpec spec;
        spec.name = m_config.track_name;
        spec.width = geometry.width;
        spec.height = geometry.height;
        spec.max_bitrate_bps = m_config.max_bitrate_bps;
        spec.max_fps = m_config.fps;
        m_track = m_livekit.publish(spec);
        m_track_geometry = geometry;
    }
    m_awaiting_keyframe = false;

    const int64_t pc_ns = upstream->mapping.to_local_ns(static_cast<int64_t>(meta.source_timestamp_ns));
    sensing::EncodedVideoFrame frame;
    frame.data = packet.data.data();
    frame.size = packet.data.size();
    frame.keyframe = packet.keyframe;
    frame.timestamp_us = static_cast<uint64_t>(pc_ns / 1000);
    frame.user_timestamp_us = frame.timestamp_us;
    frame.frame_id = static_cast<uint32_t>(meta.frame_number);
    if (m_track->submit(frame))
    {
        ++m_published;
        return;
    }
    // The SDK refused the frame, which can break the reference chain: continue from a keyframe.
    ++m_refused;
    m_awaiting_keyframe = true;
    m_idr_requested_for_gap = true;
    m_request_idr();
}

} // namespace video_stage

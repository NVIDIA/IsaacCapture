// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "livekit_output.hpp"

#include <iostream>
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

void LiveKitOutput::end_session()
{
    break_publication();
}

void LiveKitOutput::write(const EncodedPacket& packet, const FrameMeta& meta, const Geometry& geometry)
{
    const int64_t write_start_ns = monotonic_ns();
    if (m_livekit.failed())
        throw std::runtime_error("LiveKit room lost: " + m_livekit.failure());

    // A subscriber's keyframe or rate request reaches the encoder through the pipeline.
    if (m_track)
    {
        const auto feedback = m_track->take_feedback();
        if (feedback.keyframe_requested)
            m_request_idr();
        // This worker runs at the rate its source encodes; it does not adapt it to downstream congestion. Say so once
        // instead of silently discarding the request.
        if (feedback.target_bitrate_bps && !m_rate_feedback_noted)
        {
            m_rate_feedback_noted = true;
            std::cerr << "stream " << m_config.track_name << ": downstream asked for "
                      << *feedback.target_bitrate_bps / 1000
                      << " kbit/s; this worker forwards its source's rate and does not adapt it" << std::endl;
        }
    }

    // Video keeps flowing whatever the timing quality: a viewer that loses the picture every time the link's clock
    // mapping wobbles is worse than one that sees video without a latency figure. What a frame may claim about time
    // depends on the mapping: a qualified one carries the Orin's capture time on this PC's clock; otherwise the frame
    // carries no timestamp, and a consumer that needs one (a recording) says so instead of using a guessed time.
    const auto upstream = m_mapping();
    bool qualified = false;
    if (meta.source_timestamp_ns == 0)
        ++m_no_source_stamp; // the sender could not tie this unit to a capture: no capture time to map
    else if (!upstream)
        ++m_unmapped;
    else if (upstream->mapping.uncertainty_ns > m_config.max_upstream_uncertainty_ns)
        ++m_over_budget;
    else
        qualified = true;

    // This milestone publishes one fixed picture size: a change is refused rather than reconfiguring the track.
    if (m_track && !(geometry == m_track_geometry))
        throw std::runtime_error("the stream's picture size changed; publishing a different size is not supported");
    // The Orin's clock jumped: end this publication so consumers see a break instead of a silent offset change.
    if (upstream)
    {
        if (m_track && upstream->epoch != m_epoch)
        {
            ++m_republished;
            break_publication();
        }
        m_epoch = upstream->epoch;
    }

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

    // Without a qualified mapping the transport timestamp is the arrival time here and the frame has no capture time.
    const int64_t pc_ns =
        qualified ? upstream->mapping.to_local_ns(static_cast<int64_t>(meta.source_timestamp_ns)) : meta.received_ns;
    sensing::EncodedVideoFrame frame;
    frame.data = packet.data.data();
    frame.size = packet.data.size();
    frame.keyframe = packet.keyframe;
    frame.timestamp_us = static_cast<uint64_t>(pc_ns / 1000);
    frame.user_timestamp_us = qualified ? frame.timestamp_us : 0;
    frame.frame_id = static_cast<uint32_t>(meta.frame_number);
    const int64_t submit_start_ns = monotonic_ns();
    const bool accepted = m_track->submit(frame);
    const int64_t now_ns = monotonic_ns();
    if (accepted)
    {
        ++m_published;
        m_intake_to_write_ns.add(write_start_ns - meta.received_ns);
        m_submit_ns.add(now_ns - submit_start_ns);
        if (qualified)
        {
            m_uplink_ns.add(meta.received_ns - pc_ns);
            m_source_to_publish_ns.add(now_ns - pc_ns);
            m_upstream_uncertainty_ns = upstream->mapping.uncertainty_ns;
        }
        return;
    }
    // The SDK refused the frame, which can break the reference chain: continue from a keyframe.
    ++m_refused;
    m_awaiting_keyframe = true;
    m_idr_requested_for_gap = true;
    m_request_idr();
}

} // namespace video_stage

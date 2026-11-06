// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "control_channel.hpp"
#include "pipeline.hpp"

#include <sensing_livekit/livekit_session.hpp>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace video_stage
{

/// One LiveKit participant shared by every stream of the worker, as the Orin producer shares one for its cameras: one
/// identity, one room, one track per stream. Publishing is serialised because streams run on their own threads.
class SharedLiveKit
{
public:
    SharedLiveKit(const std::string& url, const std::string& token);

    std::unique_ptr<plugins::sensing::VideoTrackSink> publish(const plugins::sensing::PublishSpec& spec);
    bool failed() const;
    std::string failure() const;

private:
    plugins::sensing::LiveKitSdkReference m_sdk;
    std::unique_ptr<plugins::sensing::LiveKitSession> m_session;
    std::mutex m_mutex;
};

struct LiveKitOutputConfig
{
    std::string track_name;
    uint64_t max_bitrate_bps = 12'000'000;
    double fps = 30.0;
    /// Frames are withheld while the Orin-to-PC mapping is less certain than this. The recorder's and the page's own
    /// uncertainty covers only the leg from this worker to them; this leg is on top of it.
    int64_t max_upstream_uncertainty_ns = 5'000'000;
};

/**
 * @brief Publishes the processed access units as an ordinary LiveKit camera track.
 *
 * The recorder and the web page subscribe to it exactly as they do to the Orin's own track, so it follows the same
 * conventions: per-frame application timestamp and frame id in the packet trailer, and the publisher's own
 * monotonic clock answered on the clock-sync topic. The Orin's frame-ready stamp is converted onto this machine's
 * monotonic clock through the control-link mapping before publication, so consumers map one clock, never two.
 *
 * Timing breaks are explicit, because consumers cannot see this leg: a frame with no mapping, or a mapping worse than
 * the budget, is withheld; and when the mapping's epoch changes (the Orin's clock jumped) the track is unpublished and
 * published again at the next keyframe, which a recorder reports as a lost track instead of silently continuing a
 * synchronized take through the jump. The picture size is fixed per stream: a change is an error.
 */
class LiveKitOutput final : public PacketSink
{
public:
    /// Where the Orin-to-PC clock mapping comes from; empty when there is none (no session, or a stale mapping).
    using MappingSource = std::function<std::optional<UpstreamMapping>()>;

    /// `request_idr` is called when the track needs a keyframe (a subscriber joined, a frame was refused, or a gap).
    LiveKitOutput(SharedLiveKit& livekit,
                  const LiveKitOutputConfig& config,
                  MappingSource mapping,
                  std::function<void()> request_idr);

    void write(const EncodedPacket& packet, const FrameMeta& meta, const Geometry& geometry) override;

    /// The source session ended: unpublish, so a sender restart is never carried through as one continuous take. The
    /// next session publishes a fresh track at its first keyframe.
    void end_session();

    uint64_t published() const
    {
        return m_published.load();
    }
    /// Frames the sender sent without a capture time (it could not tie the unit to a capture).
    uint64_t no_source_stamp() const
    {
        return m_no_source_stamp.load();
    }
    uint64_t unmapped() const
    {
        return m_unmapped.load();
    }
    uint64_t refused() const
    {
        return m_refused.load();
    }
    uint64_t republished() const
    {
        return m_republished.load();
    }
    uint64_t over_budget() const
    {
        return m_over_budget.load();
    }
    /// Orin frame-ready to arrival at this PC (Orin encode, its queue and the link), per published frame.
    const DurationStats& uplink_ns() const
    {
        return m_uplink_ns;
    }
    /// Orin frame-ready to the moment this worker hands the frame to LiveKit: everything before the SFU and the page.
    const DurationStats& source_to_publish_ns() const
    {
        return m_source_to_publish_ns;
    }
    /// From the intake of the record to the start of this publish call: the worker's own time (queue, decode, warp,
    /// encode).
    const DurationStats& intake_to_write_ns() const
    {
        return m_intake_to_write_ns;
    }
    /// The LiveKit SDK call that takes the encoded frame.
    const DurationStats& submit_ns() const
    {
        return m_submit_ns;
    }
    int64_t upstream_uncertainty_ns() const
    {
        return m_upstream_uncertainty_ns.load();
    }

private:
    void break_publication(); ///< unpublish; the next keyframe publishes a fresh track

    SharedLiveKit& m_livekit;
    LiveKitOutputConfig m_config;
    MappingSource m_mapping;
    std::function<void()> m_request_idr;
    std::unique_ptr<plugins::sensing::VideoTrackSink> m_track;
    Geometry m_track_geometry;
    uint64_t m_epoch = 0;
    bool m_awaiting_keyframe = true;
    bool m_idr_requested_for_gap = false;
    bool m_rate_feedback_noted = false;
    // Counted on the output's thread and read by the statistics thread.
    std::atomic<uint64_t> m_published{ 0 };
    std::atomic<uint64_t> m_unmapped{ 0 };
    std::atomic<uint64_t> m_no_source_stamp{ 0 };
    std::atomic<uint64_t> m_refused{ 0 };
    std::atomic<uint64_t> m_republished{ 0 };
    std::atomic<uint64_t> m_over_budget{ 0 };
    DurationStats m_intake_to_write_ns;
    DurationStats m_submit_ns;
    DurationStats m_uplink_ns;
    DurationStats m_source_to_publish_ns;
    std::atomic<int64_t> m_upstream_uncertainty_ns{ 0 };
};

} // namespace video_stage

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/sensing_livekit/livekit_session.hpp"

#include <livekit/encoded_video_source.h>
#include <livekit/livekit.h>
#include <log_bridge/logger.hpp>

#include <atomic>
#include <chrono>
#include <future>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace plugins
{
namespace sensing
{

namespace
{

std::mutex g_sdk_mutex;
int g_sdk_users = 0;

const char* reason_name(livekit::DisconnectReason reason)
{
    using R = livekit::DisconnectReason;
    switch (reason)
    {
    case R::ClientInitiated:
        return "client initiated";
    case R::DuplicateIdentity:
        return "another participant joined with the same identity";
    case R::ServerShutdown:
        return "the server is shutting down";
    case R::ParticipantRemoved:
        return "the participant was removed from the room";
    case R::RoomDeleted:
        return "the room was deleted";
    default:
        return "connection lost";
    }
}

} // namespace

LiveKitSdkReference::LiveKitSdkReference()
{
    std::lock_guard<std::mutex> lock(g_sdk_mutex);
    if (g_sdk_users == 0)
        livekit::initialize(livekit::LogLevel::Warn);
    ++g_sdk_users;
}

LiveKitSdkReference::~LiveKitSdkReference()
{
    std::lock_guard<std::mutex> lock(g_sdk_mutex);
    if (--g_sdk_users == 0)
        livekit::shutdown();
}

struct LiveKitSessionImpl : livekit::RoomDelegate
{
    ~LiveKitSessionImpl() override
    {
        // Leave the room while the SDK is still up: the reference below is released after the room.
        room.disconnect();
    }

    void onDisconnected(livekit::Room&, const livekit::DisconnectedEvent& event) override
    {
        if (event.reason == livekit::DisconnectReason::ClientInitiated)
            return;
        fail(std::string("LiveKit room lost: ") + reason_name(event.reason));
    }

    void onReconnecting(livekit::Room&, const livekit::ReconnectingEvent&) override
    {
        logger->warn("LiveKit connection interrupted; the SDK is reconnecting");
    }

    void onReconnected(livekit::Room&, const livekit::ReconnectedEvent&) override
    {
        logger->info("LiveKit connection restored");
    }

    void fail(const std::string& message)
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!failed.load())
        {
            failure = message;
            failed.store(true);
            logger->error("{}", message);
        }
    }

    // Declared before the room, so destroyed after it.
    LiveKitSdkReference sdk;
    livekit::Room room;
    std::atomic<bool> failed{ false };
    mutable std::mutex mutex;
    std::string failure;
    std::shared_ptr<spdlog::logger> logger = isaaccapture::Logger::get("isaaccapture.plugins.sensing.LiveKitSession");
};

namespace
{

/// A track fed with H.264 access units that were encoded upstream.
class LiveKitVideoTrack : public VideoTrackSink
{
public:
    LiveKitVideoTrack(std::shared_ptr<LiveKitSessionImpl> session, const PublishSpec& spec)
        : m_session(std::move(session)), m_width(spec.width), m_height(spec.height)
    {
        m_source = std::make_shared<livekit::EncodedVideoSource>(
            livekit::VideoCodec::H264, static_cast<int>(spec.width), static_cast<int>(spec.height));
        m_track = livekit::LocalVideoTrack::createLocalVideoTrack(spec.name, m_source);

        livekit::TrackPublishOptions options;
        options.source = livekit::TrackSource::SOURCE_CAMERA;
        options.video_codec = livekit::VideoCodec::H264;
        // Pre-encoded access units go through as they are: no software or SDK-selected encoder.
        options.video_encoder = livekit::VideoEncoderBackend::PreEncoded;
        options.simulcast = false;
        options.video_encoding = livekit::VideoEncodingOptions{ spec.max_bitrate_bps, spec.max_fps };
        options.frame_metadata_features = livekit::FrameMetadataFeatures{ true, true, true };

        auto participant = m_session->room.localParticipant().lock();
        if (!participant)
            throw std::runtime_error("LiveKit: no local participant to publish '" + spec.name + "'");
        try
        {
            participant->publishTrack(m_track, options);
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error("LiveKit: cannot publish track '" + spec.name + "': " + e.what());
        }
        m_participant = participant;
    }

    ~LiveKitVideoTrack() override
    {
        try
        {
            if (auto participant = m_participant.lock())
                participant->unpublishTrack(m_track->sid());
        }
        catch (const std::exception& e)
        {
            m_session->logger->warn("LiveKit: unpublish failed: {}", e.what());
        }
    }

    uint32_t width() const override
    {
        return m_width;
    }
    uint32_t height() const override
    {
        return m_height;
    }

    bool submit(const EncodedVideoFrame& frame) override
    {
        if (m_session->failed.load())
            throw std::runtime_error(m_session->failure);

        livekit::EncodedVideoSource::Frame out;
        out.is_keyframe = frame.keyframe;
        out.width = m_width;
        out.height = m_height;
        out.timestamp_us = static_cast<int64_t>(frame.timestamp_us);
        out.data.assign(frame.data, frame.data + frame.size);
        livekit::VideoFrameMetadata metadata;
        metadata.user_timestamp_us = frame.timestamp_us;
        metadata.frame_id = frame.frame_id;
        std::vector<uint8_t> tsc(8);
        for (int i = 0; i < 8; ++i)
            tsc[i] = static_cast<uint8_t>((frame.capture_tsc_ns >> (8 * i)) & 0xFF);
        metadata.user_data = std::move(tsc);
        out.metadata = std::move(metadata);

        try
        {
            return m_source->captureFrame(out);
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error(std::string("LiveKit: cannot submit a frame: ") + e.what());
        }
    }

    EncoderFeedback take_feedback() override
    {
        EncoderFeedback feedback;
        try
        {
            const auto pending = m_source->takeFeedback();
            feedback.keyframe_requested = pending.keyframe_requested;
            if (pending.rate_control && pending.rate_control->target_bitrate_bps > 0)
                feedback.target_bitrate_bps = static_cast<uint32_t>(pending.rate_control->target_bitrate_bps);
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error(std::string("LiveKit: cannot read encoder feedback: ") + e.what());
        }
        return feedback;
    }

    // Called from the polling thread, which must not wait on the network: this returns the last answer
    // and starts, or collects, a stats request that was asked for earlier.
    std::string describe() override
    {
        try
        {
            if (m_stats.valid() && m_stats.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
            {
                for (const auto& entry : m_stats.get())
                {
                    const auto* out = std::get_if<livekit::RtcOutboundRtpStats>(&entry.stats);
                    if (!out || out->outbound.frame_width == 0)
                        continue;
                    const auto& o = out->outbound;
                    m_description = "encoder " + o.encoder_implementation + ", " + std::to_string(o.frame_width) + "x" +
                                    std::to_string(o.frame_height) + " @ " +
                                    std::to_string(static_cast<int>(o.frames_per_second)) + " fps, " +
                                    std::to_string(static_cast<long>(o.target_bitrate / 1000)) + " kbps target";
                    break;
                }
            }
            if (!m_stats.valid())
                m_stats = m_track->getStats();
        }
        catch (const std::exception&)
        {
        }
        return m_description;
    }

private:
    std::shared_ptr<LiveKitSessionImpl> m_session;
    uint32_t m_width;
    uint32_t m_height;
    std::shared_ptr<livekit::EncodedVideoSource> m_source;
    std::shared_ptr<livekit::LocalVideoTrack> m_track;
    std::weak_ptr<livekit::LocalParticipant> m_participant;
    decltype(std::declval<livekit::LocalVideoTrack&>().getStats()) m_stats;
    std::string m_description;
};

} // namespace

LiveKitSession::LiveKitSession(std::shared_ptr<LiveKitSessionImpl> impl) : m_impl(std::move(impl))
{
}

LiveKitSession::~LiveKitSession() = default;

std::unique_ptr<LiveKitSession> LiveKitSession::connect(const std::string& url, const std::string& token)
{
    auto impl = std::make_shared<LiveKitSessionImpl>();
    impl->room.setDelegate(impl.get());

    livekit::RoomOptions options;
    options.auto_subscribe = false; // this process only publishes
    options.dynacast = false;
    if (!impl->room.connect(url, token, options))
    {
        throw std::runtime_error("LiveKit: cannot join the room at " + url +
                                 " (check the URL, that the server is reachable and that the token is valid)");
    }
    impl->logger->info("Joined LiveKit room at {}", url);
    return std::unique_ptr<LiveKitSession>(new LiveKitSession(std::move(impl)));
}

std::unique_ptr<VideoTrackSink> LiveKitSession::publish_video(const PublishSpec& spec)
{
    return std::make_unique<LiveKitVideoTrack>(m_impl, spec);
}

bool LiveKitSession::failed() const
{
    return m_impl->failed.load();
}

std::string LiveKitSession::failure() const
{
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return m_impl->failure;
}

} // namespace sensing
} // namespace plugins

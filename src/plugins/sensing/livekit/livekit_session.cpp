// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/sensing_livekit/livekit_session.hpp"

#include <livekit/livekit.h>
#include <log_bridge/logger.hpp>

#include <atomic>
#include <chrono>
#include <future>
#include <mutex>
#include <optional>
#include <stdexcept>
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
    if (g_sdk_users++ == 0)
        livekit::initialize(livekit::LogLevel::Warn);
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

/// A track fed from one NV12 buffer.
class LiveKitVideoTrack : public VideoTrackSink
{
public:
    LiveKitVideoTrack(std::shared_ptr<LiveKitSessionImpl> session, const PublishSpec& spec)
        : m_session(std::move(session)),
          m_width(spec.width),
          m_height(spec.height),
          m_frame(livekit::VideoFrame::create(
              static_cast<int>(spec.width), static_cast<int>(spec.height), livekit::VideoBufferType::NV12))
    {
        m_source = std::make_shared<livekit::VideoSource>(static_cast<int>(spec.width), static_cast<int>(spec.height));
        m_track = livekit::LocalVideoTrack::createLocalVideoTrack(spec.name, m_source);

        livekit::TrackPublishOptions options;
        options.source = livekit::TrackSource::SOURCE_CAMERA;
        options.video_codec = livekit::VideoCodec::H264;
        options.simulcast = false;
        options.video_encoding = livekit::VideoEncodingOptions{ spec.max_bitrate_bps, spec.max_fps };
        options.frame_metadata_features = livekit::FrameMetadataFeatures{ true, true, true };
        // Under bandwidth pressure give up frame rate, not picture size or aspect.
        options.degradation_preference = livekit::DegradationPreference::MaintainResolution;

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
    uint8_t* frame_buffer() override
    {
        return m_frame.data();
    }

    void submit(const VideoFrameInfo& info) override
    {
        if (m_session->failed.load())
            throw std::runtime_error(m_session->failure);

        livekit::VideoCaptureOptions options;
        options.timestamp_us = static_cast<int64_t>(info.timestamp_us);
        livekit::VideoFrameMetadata metadata;
        metadata.user_timestamp_us = info.timestamp_us;
        metadata.frame_id = info.frame_id;
        std::vector<uint8_t> tsc(8);
        for (int i = 0; i < 8; ++i)
            tsc[i] = static_cast<uint8_t>((info.capture_tsc_ns >> (8 * i)) & 0xFF);
        metadata.user_data = std::move(tsc);
        options.metadata = std::move(metadata);

        try
        {
            m_source->captureFrame(m_frame, options);
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error(std::string("LiveKit: cannot submit a frame: ") + e.what());
        }
    }

    std::string describe() override
    {
        try
        {
            auto stats = m_track->getStats();
            if (stats.wait_for(std::chrono::milliseconds(500)) != std::future_status::ready)
                return {};
            for (const auto& entry : stats.get())
            {
                if (const auto* out = std::get_if<livekit::RtcOutboundRtpStats>(&entry.stats))
                {
                    const auto& o = out->outbound;
                    if (o.frame_width == 0)
                        continue;
                    return "encoder " + o.encoder_implementation + ", " + std::to_string(o.frame_width) + "x" +
                           std::to_string(o.frame_height) + " @ " +
                           std::to_string(static_cast<int>(o.frames_per_second)) + " fps, " +
                           std::to_string(static_cast<long>(o.target_bitrate / 1000)) + " kbps target";
                }
            }
        }
        catch (const std::exception&)
        {
        }
        return {};
    }

private:
    std::shared_ptr<LiveKitSessionImpl> m_session;
    uint32_t m_width;
    uint32_t m_height;
    livekit::VideoFrame m_frame;
    std::shared_ptr<livekit::VideoSource> m_source;
    std::shared_ptr<livekit::LocalVideoTrack> m_track;
    std::weak_ptr<livekit::LocalParticipant> m_participant;
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

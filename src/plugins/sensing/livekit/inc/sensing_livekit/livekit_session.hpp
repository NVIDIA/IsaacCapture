// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "video_track_sink.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace plugins
{
namespace sensing
{

struct LiveKitSessionImpl;

/**
 * @brief Keeps the LiveKit SDK initialised while it is held.
 *
 * The SDK is process-wide and must be shut down only when nothing in the process uses it, so every
 * user holds one of these and the last release shuts it down.
 */
class LiveKitSdkReference
{
public:
    LiveKitSdkReference();
    ~LiveKitSdkReference();
    LiveKitSdkReference(const LiveKitSdkReference&) = delete;
    LiveKitSdkReference& operator=(const LiveKitSdkReference&) = delete;
};

struct PublishSpec
{
    /// Name subscribers see for the track.
    std::string name;
    uint32_t width = 0;
    uint32_t height = 0;
    /// Upper bound the encoder may use; WebRTC adapts below it.
    uint64_t max_bitrate_bps = 0;
    double max_fps = 30.0;
};

/**
 * @brief One participant in one LiveKit room.
 *
 * The room and the participant's identity come from the join token. A terminal loss of the room
 * (the server went away, the participant was removed) sets failed(); the SDK's own bounded
 * reconnection runs first and is not wrapped in another supervisor.
 */
class LiveKitSession
{
public:
    /// @throws std::runtime_error if the room cannot be joined. The token is never logged.
    static std::unique_ptr<LiveKitSession> connect(const std::string& url, const std::string& token);

    /// Unpublish every track (destroy them first), then leave the room.
    ~LiveKitSession();

    LiveKitSession(const LiveKitSession&) = delete;
    LiveKitSession& operator=(const LiveKitSession&) = delete;

    /// Publishes an H.264 camera track fed from NV12 frames.
    /// @throws std::runtime_error if publication fails.
    std::unique_ptr<VideoTrackSink> publish_video(const PublishSpec& spec);

    /// True once the room has been lost for good.
    bool failed() const;
    std::string failure() const;

private:
    explicit LiveKitSession(std::shared_ptr<LiveKitSessionImpl> impl);
    std::shared_ptr<LiveKitSessionImpl> m_impl;
};

} // namespace sensing
} // namespace plugins

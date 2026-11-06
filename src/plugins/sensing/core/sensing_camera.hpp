// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "camera_capture.hpp"
#include "livekit_video_output.hpp"
#include "sensing_types.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace plugins
{
namespace sensing
{

class LiveKitSession;

/**
 * @brief Multi-sensor SENSING camera publisher.
 *
 * One CameraCapture owns every camera of the rig (SIPL's camera object is process-wide, so there is no
 * per-camera capture object to hold). Each update() polls every sensor's latest frame and hands a new frame
 * to that sensor's LiveKit track. Geometry is per sensor and comes from the capture backend.
 */
/// Live figures for one stream, for the management channel.
struct StreamFigures
{
    std::string track;
    uint32_t sensor_id = 0;
    uint64_t frames = 0;
    uint64_t published = 0;
    DirectLinkStatus direct;
};

class SensingCamera
{
public:
    /// `camera` is the opened backend; its camera ids are what StreamConfig::sensor_id names.
    SensingCamera(const SensingConfig& config,
                  std::unique_ptr<CameraCapture> camera,
                  const std::vector<StreamConfig>& streams);
    ~SensingCamera();

    SensingCamera(const SensingCamera&) = delete;
    SensingCamera& operator=(const SensingCamera&) = delete;
    SensingCamera(SensingCamera&&) = delete;
    SensingCamera& operator=(SensingCamera&&) = delete;

    /** @brief Poll every sensor and submit newly captured frames. Throws once the room is lost. */
    void update();

    /** @brief Flush each track's queued frames. */
    void flush();

    /** @brief Per-stream figures. Call from the thread that calls update(). */
    std::vector<StreamFigures> figures() const;

    /**
     * @brief Send stream `track` to host:port from now on (empty host stops the direct link).
     * @return false if no stream has that track name. Thread-safe against update().
     */
    bool set_direct_target(const std::string& track, const std::string& host, uint16_t port);

    /** @brief Print per-sensor frame counts to stdout. */
    void print_stats() const;

private:
    struct Stream
    {
        uint32_t sensor_id = 0;
        std::string track;
        std::unique_ptr<LiveKitVideoOutput> livekit;
        /// SIPL publish counter of the frame already handled; skips re-reads.
        uint64_t last_sequence = 0;
        bool have_last_sequence = false;
        uint64_t frame_count = 0;
        /// Captures the poll loop never saw, from gaps in the SIPL sequence.
        uint64_t missed_captures = 0;
    };

    SensingConfig m_config;
    std::unique_ptr<CameraCapture> m_camera;
    /// Declared before the streams so it is destroyed after their tracks have been unpublished.
    std::unique_ptr<LiveKitSession> m_livekit;
    std::vector<Stream> m_streams;
};

} // namespace sensing
} // namespace plugins

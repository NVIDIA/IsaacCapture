// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sensing_types.hpp"
#include "sipl_camera.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace plugins
{
namespace sensing
{

class LiveKitSession;
class LiveKitVideoOutput;

/// Live figures for one stream, for the management channel.
struct StreamFigures
{
    std::string track;
    uint32_t sensor_id = 0;
    uint64_t frames = 0;
    uint64_t published = 0;
};

/**
 * @brief Multi-sensor SENSING camera publisher.
 *
 * One SiplCamera owns every sensor, because INvSIPLCamera::GetInstance() is process-wide -- there is no
 * per-sensor capture object to hold. Each update() polls every sensor's latest-frame mailbox and hands a new
 * frame to that sensor's LiveKit track. Geometry is per sensor and comes from the platform config.
 */
class SensingCamera
{
public:
    SensingCamera(const SensingConfig& config, const std::vector<StreamConfig>& streams);
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
    std::unique_ptr<SiplCamera> m_camera;
    /// Declared before the streams so it is destroyed after their tracks have been unpublished.
    std::unique_ptr<LiveKitSession> m_livekit;
    std::vector<Stream> m_streams;
};

} // namespace sensing
} // namespace plugins

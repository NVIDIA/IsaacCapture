// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sipl_camera.hpp"

#include <sensing_livekit/video_track_sink.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace plugins
{
namespace sensing
{

struct LiveKitOutputConfig
{
    uint32_t source_width = 0;
    uint32_t source_height = 0;
    /// Uniform output-only reduction; 1 publishes the sensor's native size.
    uint32_t scale_div = 1;
    /// Publication rate cap, independent of the capture rate.
    uint32_t fps = 30;
    int gpu_id = 0;
};

struct LiveKitOutputStats
{
    uint64_t offered = 0; ///< Captures handed to submit().
    uint64_t paced_out = 0; ///< Skipped to hold the publication rate.
    uint64_t replaced = 0; ///< Snapshots overwritten by a newer capture before the worker took them.
    uint64_t snapshots = 0; ///< GPU snapshots taken.
    uint64_t published = 0; ///< Frames the SDK accepted. Not proof that a viewer received them.
};

/**
 * @brief Feeds one sensor's frames to a video track, off the polling thread.
 *
 * submit() runs on the caller's polling loop. It takes a snapshot of the borrowed camera frame in
 * GPU memory and finishes that copy before returning, because the camera may overwrite its slot at
 * the second following poll. Everything else -- the optional reduction, the RGBA-to-NV12
 * conversion, the host copy and the SDK call -- happens on this output's own worker thread, so a slow
 * network cannot hold up capture or the other sensors.
 *
 * Two snapshot slots bound the memory: one the worker is publishing, one waiting. A newer capture
 * replaces the waiting one; the worker never waits for it.
 */
class LiveKitVideoOutput
{
public:
    /// @throws std::runtime_error if the geometry is unusable (see validate_geometry) or CUDA fails.
    LiveKitVideoOutput(std::unique_ptr<VideoTrackSink> sink, const LiveKitOutputConfig& config);

    /// Stops the worker; a snapshot not yet published is dropped.
    ~LiveKitVideoOutput();

    LiveKitVideoOutput(const LiveKitVideoOutput&) = delete;
    LiveKitVideoOutput& operator=(const LiveKitVideoOutput&) = delete;

    /// Never blocks on the network. A view older than the publication rate allows is skipped.
    void submit(const FrameView& view);

    LiveKitOutputStats stats() const;

    bool failed() const;
    std::string failure() const;

    /// The sink's description of its encoder, for diagnostics.
    std::string describe();

    /**
     * Output size after dividing: both source dimensions must divide exactly and leave positive,
     * even results. Nothing is rounded or stretched.
     * @throws std::runtime_error naming the problem.
     */
    static void validate_geometry(uint32_t width, uint32_t height, uint32_t divisor);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace sensing
} // namespace plugins

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <livekit_publisher/livekit_session.hpp>
#include <nvenc/cuda_context.hpp>
#include <video_types/timestamp_basis.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

namespace camera_bridge
{

struct WorkerConfig
{
    /// The stream's name in camera_receiver's IPC catalog.
    std::string stream;
    /// The LiveKit track name egocap's streams file expects.
    std::string track;
    std::string socket_path;
    uint32_t fps = 30;
    uint32_t bitrate_bps = 8'000'000;
    uint32_t idr_period = 30;
    /// Integer factor the picture is shrunk by on the GPU before encoding (1 keeps the source size).
    uint32_t scale_div = 1;
    /// What each frame's user_timestamp means; the consumer's streams file must name the same basis.
    TimestampBasis timestamp_basis = TimestampBasis::CaptureUnixUs;
};

struct WorkerStats
{
    std::atomic<uint64_t> frames{ 0 };
    std::atomic<uint64_t> skipped{ 0 };
    std::atomic<uint64_t> keyframes{ 0 };
    std::atomic<uint64_t> rejected{ 0 };
    /// Pictures dropped because the publisher may have overwritten the IPC slot during the copy.
    std::atomic<uint64_t> slot_overruns{ 0 };
    std::atomic<uint64_t> encode_us_total{ 0 };
    /// The bitrate the encoder is configured for now, and how many times congestion feedback changed it.
    std::atomic<uint32_t> bitrate_bps{ 0 };
    std::atomic<uint64_t> rate_changes{ 0 };
    /// Sensor start-of-frame to frame-ready, from frames whose sensor time is on a realtime sender clock. Only as
    /// good as the agreement between the sender's wall clock and this host's.
    std::atomic<uint64_t> capture_to_ready_us_total{ 0 };
    std::atomic<uint64_t> capture_to_ready_samples{ 0 };
    std::atomic<uint32_t> capture_to_ready_max_us{ 0 };
    /// The latest target the transport reported (0 until it reports one).
    std::atomic<uint32_t> target_bps{ 0 };
};

/**
 * @brief One camera stream from CUDA IPC to a LiveKit track on its own thread.
 *
 * Each frame is copied on the GPU from the IPC slot into an NVENC input surface, encoded to H.264 and submitted as a
 * pre-encoded access unit. The pixels never reach host memory. Streams share nothing but the CUDA context and the
 * LiveKit room, so one stalling does not delay another.
 */
class StreamWorker
{
public:
    /// Connects to the IPC socket and publishes the track. Throws std::runtime_error with the reason on failure.
    StreamWorker(const WorkerConfig& config, CudaContext& context, LiveKitSession& session);
    ~StreamWorker();

    StreamWorker(const StreamWorker&) = delete;
    StreamWorker& operator=(const StreamWorker&) = delete;

    void start();
    void stop();

    const WorkerConfig& config() const
    {
        return m_config;
    }
    const WorkerStats& stats() const
    {
        return m_stats;
    }
    /// The reason the worker ended, or empty while it runs or after a requested stop.
    std::string failure() const;

private:
    struct Impl;
    WorkerConfig m_config;
    WorkerStats m_stats;
    std::unique_ptr<Impl> m_impl;
};

} // namespace camera_bridge

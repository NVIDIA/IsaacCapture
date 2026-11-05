// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "jetson_encoder.hpp"
#include "sensing_types.hpp"
#include "sipl_camera.hpp"

#include <sensing_cuda_ipc/cuda_ipc_publisher.hpp>

#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

namespace plugins
{
namespace sensing
{

class FrameSink;

/**
 * @brief Multi-sensor SENSING camera manager.
 *
 * One SiplCamera owns every sensor, because INvSIPLCamera::GetInstance() is
 * process-wide -- there is no per-sensor capture object to hold. Each update()
 * polls every sensor's latest-frame mailbox and fans the new frame out to
 * whichever destinations that stream configured: an H.264 encoder, a CUDA IPC
 * socket, or both. A stream may have an encoder or not; the CUDA path
 * deliberately needs neither.
 *
 * Geometry is per sensor and comes from the platform config, not from the CLI.
 */
class SensingCamera
{
public:
    SensingCamera(const SensingConfig& config, const std::vector<StreamConfig>& streams, std::unique_ptr<FrameSink> sink);
    ~SensingCamera();

    SensingCamera(const SensingCamera&) = delete;
    SensingCamera& operator=(const SensingCamera&) = delete;
    SensingCamera(SensingCamera&&) = delete;
    SensingCamera& operator=(SensingCamera&&) = delete;

    /** @brief Poll every sensor and dispatch newly encoded frames. */
    void update();

    /** @brief Flush each encoder's queued packets into the sink. */
    void flush();

    /** @brief Print per-sensor frame counts to stdout. */
    void print_stats() const;

private:
    struct Stream
    {
        uint32_t sensor_id = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        /// Null when the stream requested no H.264 output.
        std::unique_ptr<JetsonEncoder> encoder;
        /// Null when the stream requested no CUDA IPC socket.
        std::unique_ptr<CudaIpcPublisher> publisher;
        /// SIPL publish counter of the frame already handled; skips re-reads.
        uint64_t last_sequence = 0;
        bool have_last_sequence = false;
        uint64_t frame_count = 0;
        /// Frames the encoder refused because no input buffer was free.
        uint64_t encoder_drops = 0;
        /// Captures the poll loop never saw, from gaps in the SIPL sequence.
        uint64_t missed_captures = 0;
        /// (monotonic, TSC) for frames submitted to the encoder but not yet
        /// emitted. The encoder returns the monotonic stamp it was given, which
        /// is how a delayed unit recovers the capture TSC of its own frame
        /// rather than of whatever frame happens to be current.
        std::deque<std::pair<int64_t, int64_t>> pending_stamps;
    };

    void dispatch(Stream& stream, std::vector<uint8_t> h264, int64_t timestamp_ns, int64_t capture_tsc_ns);
    int64_t take_capture_tsc(Stream& stream, int64_t timestamp_ns);

    SensingConfig m_config;
    std::unique_ptr<SiplCamera> m_camera;
    std::vector<Stream> m_streams;
    std::unique_ptr<FrameSink> m_sink;
};

} // namespace sensing
} // namespace plugins
